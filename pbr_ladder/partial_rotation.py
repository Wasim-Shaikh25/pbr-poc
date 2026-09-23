# Proposal 1 test: the space-filling vs memory-gain trade-off in the rotation.
# (weight-only, CPU numpy)
#
# THEORY (Gersho-Gray rate-distortion decomposition):
#   VQ's advantage over optimal scalar quantization on any source =
#     shape gain (scalar+companding already gets this)
#   + memory gain  (correlation BETWEEN the D components -- only VQ gets it)
#   + space-filling gain (<=1.53 dB as D->inf; ~0.65 dB at D=8 via E8).
# Full incoherence rotation makes the D=8 block ~i.i.d. Gaussian -> memory gain
# = 0, leaving VQ only the tiny D=8 space-filling gain.  That is WHY memoryless
# D=8 VQ measured -0.43 dB vs Lloyd-Max scalar at full rotation.
#
# Here we DIAL DOWN the input-side rotation Ri so inter-column correlation
# survives, and watch the VQ-over-scalar output-SQNR gain reappear.  Rotation
# "reach" is controlled by a BLOCK-DIAGONAL orthogonal Ri with block size b:
#   b = 1   -> identity (no input rotation, full native correlation kept)
#   b < D=8 -> a vector's 8 cols straddle block boundaries -> correlation kept
#   b >= 8  -> within-vector cols fully mixed -> memory gain destroyed
#   b = I   -> one full random rotation (== the current pipeline, alpha=1)
# Ro (output side) is kept as a fixed full random rotation throughout, so we
# isolate the column-side (Ri) effect that governs the D=8 vector grouping.
#
# Usage:
#   python partial_rotation.py [weight_key] [acts.npy] [stages]
#   python partial_rotation.py model.layers.12.mlp.gate_proj.weight acts_L12_gate.npy 256,256,256
import sys, numpy as np
from safetensors import safe_open

CK   = "./qwen05b/model.safetensors"
KEY  = sys.argv[1] if len(sys.argv) > 1 else "model.layers.12.mlp.gate_proj.weight"
ACTF = sys.argv[2] if len(sys.argv) > 2 else "acts_L12_gate.npy"
STAGES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "256,256,256").split(",")]
D  = 8
CH = 40000
rng = np.random.default_rng(0)

with safe_open(CK, "pt") as f:
    W = f.get_tensor(KEY).float().numpy().astype(np.float64)
A = np.load(ACTF).astype(np.float64)
O, I = W.shape
H = A.T @ A / len(A)
WH = W @ H
sig = float(np.sum(W * WH))

def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q * np.sign(np.diag(r))
Ro = ortho(O, O)                                       # fixed full output rotation

def block_ortho(n, b, seed):
    """block-diagonal orthogonal; b=1 -> identity, b=n -> single full rotation."""
    r = np.random.default_rng(seed); M = np.eye(n); i = 0
    while i < n:
        bs = min(b, n - i)
        if bs > 1:
            q, rr = np.linalg.qr(r.normal(size=(bs, bs)))
            M[i:i+bs, i:i+bs] = q * np.sign(np.diag(rr))
        i += bs
    return M

def eucl_assign(Vc, C):
    return ((Vc**2).sum(1)[:, None] - 2*Vc @ C.T + (C**2).sum(1)[None]).argmin(1)

def kmeans(Vv, k, it=10):
    C = Vv[rng.choice(len(Vv), k, replace=False)].copy()
    for _ in range(it):
        a = np.concatenate([eucl_assign(Vv[i:i+CH], C) for i in range(0, len(Vv), CH)])
        for j in range(k):
            m = a == j
            if m.any(): C[j] = Vv[m].mean(0)
    return C

def rvq(V):
    books, R = [], V.copy()
    for k in STAGES:
        C = kmeans(R, k)
        a = np.concatenate([eucl_assign(R[i:i+CH], C) for i in range(0, len(R), CH)])
        books.append(C); R = R - C[a]
    q = np.zeros_like(V); R = V.copy()
    for C in books:
        a = np.concatenate([eucl_assign(R[i:i+CH], C) for i in range(0, len(R), CH)])
        q += C[a]; R = R - C[a]
    return q

def lloyd1d(x, levels, it=40):
    x = x.reshape(-1); C = np.quantile(x, (np.arange(levels) + 0.5) / levels)
    for _ in range(it):
        mids = (C[:-1] + C[1:]) / 2; a = np.searchsorted(mids, x)
        for j in range(levels):
            m = a == j
            if m.any(): C[j] = x[m].mean()
        C.sort()
    return C

def osqnr_from_Vq(Vq, s, Ri):
    Wq = Ro.T @ (Vq.reshape(O, I) * s) @ Ri.T
    E = W - Wq
    return 10 * np.log10(sig / float(np.sum(E * (E @ H))))

bpw = sum(np.log2(k) for k in STAGES) / D
levels = int(round(2 ** (bpw)))
bsizes = [b for b in [1, 2, 4, 7, 8, 14, 28, 56, 112, 224, 448, 896] if b <= I]

print(f"tensor {KEY}  W{W.shape}  acts{A.shape}  stages={STAGES}  bpw={bpw:.3f}  scalar={levels}-lvl")
print(f"{'block b':>8} {'corr':>7} {'VQ dB':>8} {'scalar dB':>10} {'VQ-scalar':>10}")
print("-" * 48)
rows = []
for b in bsizes:
    Ri = block_ortho(I, b, seed=1000 + b)
    Wr = Ro @ W @ Ri
    s = np.sqrt((Wr ** 2).mean(1, keepdims=True)) + 1e-12
    V = (Wr / s).reshape(-1, D)
    # mean |correlation| between adjacent columns within a vector (memory proxy)
    Vc = (Wr / s)                                     # (O, I)
    cc = np.corrcoef(Vc[:, :D].T)                     # 8x8 within first block
    corr = float(np.mean(np.abs(cc[np.triu_indices(D, 1)])))
    vq = osqnr_from_Vq(rvq(V), s, Ri)
    Cs = lloyd1d(V, levels); mids = (Cs[:-1] + Cs[1:]) / 2
    Vq_s = Cs[np.searchsorted(mids, V.reshape(-1))].reshape(V.shape)
    sc = osqnr_from_Vq(Vq_s, s, Ri)
    rows.append((b, corr, vq, sc, vq - sc))
    print(f"{b:>8} {corr:>7.3f} {vq:>8.2f} {sc:>10.2f} {vq-sc:>+10.2f}")

print("\nInterpretation:")
print("  gain>0  => VQ memory gain has reappeared (scalar cannot follow)")
print("  gain~=-0.43 at b=896 reproduces the full-rotation dead end")
best = max(rows, key=lambda r: r[4])
print(f"  best VQ-over-scalar gain {best[4]:+.2f} dB at block b={best[0]} (corr {best[1]:.3f})")
print("  NOTE: small b keeps outliers/importance-spread that GPTQ (not in this")
print("        test) relies on rotation to flatten -- compare ABSOLUTE VQ dB too.")
try:
    import csv
    with open("partial_rotation_result.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["block_b", "within_vec_corr", "vq_dB", "scalar_dB", "gain_dB"])
        w.writerows(rows)
    print("  wrote partial_rotation_result.csv")
except Exception as e:
    print("  (csv skip:", e, ")")
