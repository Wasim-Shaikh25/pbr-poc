# Proposal 2 test: does Hessian-Mahalanobis VQ assignment beat plain Euclidean
# at MATCHED bpw on a real rotated tensor?  (weight-only, CPU numpy)
#
# The current pipeline (beam_vq.py) assigns codewords by plain Euclidean
# ||v-c||^2 on rotated weights.  The TRUE objective is the Hessian-weighted
# output error  tr(E Hr E^T),  Hr = Ri^T (A^T A) Ri  (rotated input Hessian).
# With D=8 columns per vector, the correct *local* metric is the 8x8 diagonal
# block of Hr for that column-position.  We compare, at identical bits:
#   (0) Lloyd-Max 3-bit SCALAR              -- anchor (what VQ must beat)
#   (1) Euclidean RVQ                       -- current pipeline
#   (2) Euclidean-fit books + Mahal assign  -- isolates the *assignment* metric
#   (3) Mahalanobis-consistent RVQ          -- books fit AND assigned under Hr
#       (centroid update is still the plain mean -- the minimiser of a fixed
#        quadratic metric is the arithmetic mean, so only the ASSIGN step changes)
#
# Output SQNR is measured in the ORIGINAL weight space via the trace form
# tr(E H E^T) = sum(E * (E @ H)) so it never materialises A @ W^T (memory-safe,
# and identical up to the constant len(A) to beam_vq's osqnr ratio).
#
# Usage:
#   python mahalanobis_assign.py [weight_key] [acts.npy] [stages] [damp]
#   python mahalanobis_assign.py model.layers.12.mlp.gate_proj.weight acts_L12_gate.npy 256,256,256 0.01
import sys, numpy as np
from safetensors import safe_open

CK   = "./qwen05b/model.safetensors"
KEY  = sys.argv[1] if len(sys.argv) > 1 else "model.layers.12.mlp.gate_proj.weight"
ACTF = sys.argv[2] if len(sys.argv) > 2 else "acts_L12_gate.npy"
STAGES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "256,256,256").split(",")]
DAMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01   # rel. diagonal damping on Hr block
D  = 8
CH = 40000                      # HARD CONSTRAINT: chunk every (N x K) distance matrix
rng = np.random.default_rng(0)

# ---------- load ----------
with safe_open(CK, "pt") as f:
    W = f.get_tensor(KEY).float().numpy().astype(np.float64)     # (O, I)
A = np.load(ACTF).astype(np.float64)                             # (tokens, I)
O, I = W.shape
P = I // D                                                       # column-blocks per row
assert I % D == 0, f"I={I} not divisible by D={D}"
H = A.T @ A / len(A)                                             # (I, I) input Hessian

# ---------- rotation (full random-QR, both sides -- matches beam_vq) ----------
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q * np.sign(np.diag(r))
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri
s  = np.sqrt((Wr ** 2).mean(1, keepdims=True)) + 1e-12           # per-row scale
V  = (Wr / s).reshape(-1, D)                                     # (O*P, D) pooled vectors
Vg = V.reshape(O, P, D)                                          # [o, p, :] = block p of row o

# ---------- rotated Hessian and its 8x8 diagonal blocks ----------
Hr = Ri.T @ H @ Ri
Hblocks = np.empty((P, D, D))
for p in range(P):
    blk = Hr[p*D:(p+1)*D, p*D:(p+1)*D].copy()
    blk = 0.5 * (blk + blk.T)                                    # symmetrise
    blk += DAMP * np.mean(np.diag(blk)) * np.eye(D)              # damp for stability
    Hblocks[p] = blk
conds = np.array([np.linalg.cond(Hblocks[p]) for p in range(P)])
print(f"tensor {KEY}  W{W.shape}  acts{A.shape}  P={P} blocks/row  stages={STAGES}")
print(f"rotated 8x8 Hessian-block cond: median {np.median(conds):.2f}  p90 {np.percentile(conds,90):.2f}")

# ---------- distances ----------
def eucl_assign(Vc, C):                                          # (n,D),(K,D)->(n,)
    return ((Vc**2).sum(1)[:, None] - 2*Vc @ C.T + (C**2).sum(1)[None]).argmin(1)

def mahal_assign(R, C, Hp):                                      # (n,D),(K,D),(D,D)->(n,)
    RH = R @ Hp                                                  # (n,D)
    rHr = np.einsum('ni,ni->n', RH, R)                           # (n,)
    CH = C @ Hp
    cHc = np.einsum('ki,ki->k', CH, C)                           # (K,)
    d = rHr[:, None] - 2.0 * (RH @ C.T) + cHc[None, :]           # (n,K)  n<=O<40000
    return d.argmin(1)

# ---------- codebook fitting ----------
def kmeans_eucl(Vv, k, it=10):
    C = Vv[rng.choice(len(Vv), k, replace=False)].copy()
    for _ in range(it):
        a = np.concatenate([eucl_assign(Vv[i:i+CH], C) for i in range(0, len(Vv), CH)])
        for j in range(k):
            m = a == j
            if m.any(): C[j] = Vv[m].mean(0)
    return C

def kmeans_mahal(Rg, k, it=10):                                  # Rg:(O,P,D) residuals
    flat = Rg.reshape(-1, D)
    C = flat[rng.choice(len(flat), k, replace=False)].copy()
    for _ in range(it):
        Aall = np.empty((O, P), np.int64)
        for p in range(P):
            Aall[:, p] = mahal_assign(Rg[:, p, :], C, Hblocks[p])   # n=O=4864 rows
        fa = Aall.reshape(-1)
        cnt = np.bincount(fa, minlength=k).astype(np.float64)
        nz = cnt > 0
        for d in range(D):
            C[nz, d] = np.bincount(fa, weights=flat[:, d], minlength=k)[nz] / cnt[nz]
    return C

# fit both codebook sets (same K, same #stages => identical stored bits)
books_e, R = [], V.copy()
for k in STAGES:
    C = kmeans_eucl(R, k)
    a = np.concatenate([eucl_assign(R[i:i+CH], C) for i in range(0, len(R), CH)])
    books_e.append(C); R = R - C[a]

books_m, Rg = [], Vg.copy()
for k in STAGES:
    C = kmeans_mahal(Rg, k)
    for p in range(P):
        a = mahal_assign(Rg[:, p, :], C, Hblocks[p])
        Rg[:, p, :] -= C[a]
    books_m.append(C)

# ---------- decoders ----------
def decode_eucl(books):
    q = np.zeros_like(V); r = V.copy()
    for C in books:
        a = np.concatenate([eucl_assign(r[i:i+CH], C) for i in range(0, len(r), CH)])
        q += C[a]; r -= C[a]
    return q                                                     # (O*P, D)

def decode_mahal(books):
    qg = np.zeros_like(Vg); rg = Vg.copy()
    for C in books:
        for p in range(P):
            a = mahal_assign(rg[:, p, :], C, Hblocks[p])
            qg[:, p, :] += C[a]; rg[:, p, :] -= C[a]
    return qg.reshape(-1, D)

# ---------- scalar Lloyd-Max anchor (3 bit) ----------
def lloyd1d(x, levels=8, it=40):
    x = x.reshape(-1)
    C = np.quantile(x, (np.arange(levels) + 0.5) / levels)
    for _ in range(it):
        mids = (C[:-1] + C[1:]) / 2
        a = np.searchsorted(mids, x)
        for j in range(levels):
            m = a == j
            if m.any(): C[j] = x[m].mean()
        C.sort()
    return C
bits_scalar = sum(np.log2(k) for k in STAGES) / D                # match VQ bpw
levels = int(round(2 ** bits_scalar))
Cs = lloyd1d(V, levels)
mids = (Cs[:-1] + Cs[1:]) / 2
Vq_s = Cs[np.searchsorted(mids, V.reshape(-1))].reshape(V.shape)

# ---------- output SQNR in original space (trace form, memory-safe) ----------
WH = W @ H
sig = float(np.sum(W * WH))
def osqnr(Vq):
    Wq = Ro.T @ (Vq.reshape(O, I) * s) @ Ri.T
    E = W - Wq
    return 10 * np.log10(sig / float(np.sum(E * (E @ H))))

bpw = sum(np.log2(k) for k in STAGES) / D
sq_scalar = osqnr(Vq_s)
sq_e   = osqnr(decode_eucl(books_e))
sq_ea  = osqnr(decode_mahal(books_e))       # Euclidean books, Mahalanobis assign
sq_m   = osqnr(decode_mahal(books_m))       # fully Mahalanobis-consistent RVQ

print(f"\n=== output SQNR @ {bpw:.3f} bpw (all identical bits) ===")
print(f"  Lloyd-Max scalar ({levels}-lvl)   : {sq_scalar:6.2f} dB   [anchor]")
print(f"  (1) Euclidean RVQ               : {sq_e:6.2f} dB   ({sq_e-sq_scalar:+.2f} vs scalar)")
print(f"  (2) Eucl books + Mahal assign   : {sq_ea:6.2f} dB   ({sq_ea-sq_e:+.2f} vs Eucl RVQ)")
print(f"  (3) Mahalanobis-consistent RVQ  : {sq_m:6.2f} dB   ({sq_m-sq_e:+.2f} vs Eucl RVQ)")
print(f"\nKEY DELTAS:  assign-only {sq_ea-sq_e:+.2f} dB | full-Mahal {sq_m-sq_e:+.2f} dB | "
      f"Mahal-vs-scalar {sq_m-sq_scalar:+.2f} dB")
