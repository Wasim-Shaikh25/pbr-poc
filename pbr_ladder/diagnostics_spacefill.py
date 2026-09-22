# Space-filling-gain diagnostic (both experts' #1 go/no-go).
# On a rotated real tensor, at matched bpw, compare:
#   - optimal SCALAR (Lloyd-Max, per rotated weight)  -> captures shape gain
#   - our k-means RVQ (D=8)                            -> shape + whatever space-filling it gets
# The gap = space-filling gain actually realized. Ceiling at D=8 is ~0.65 dB (E8).
# Also: isotropy check -- condition number of rotated 8x8 Hessian blocks (is a
# Hessian-Mahalanobis assignment metric worth anything, or ~Euclidean already?).
#
# Usage: python diagnostics_spacefill.py <weight_key> <acts.npy> [bpw]
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
BPW = int(sys.argv[3]) if len(sys.argv) > 3 else 3
D = 8
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
H = A.T @ A / len(A)
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n))); return q*np.sign(np.diag(r))
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri
s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
Vn = (Wr / s)                                   # normalized rotated weights

def lloyd_max(x, levels, it=30):
    """1-D Lloyd-Max on samples x -> quantized x (optimal scalar for this marginal)."""
    q = np.quantile(x, np.linspace(0, 1, levels+2)[1:-1]); c = q.copy()
    for _ in range(it):
        edges = (c[:-1]+c[1:])/2
        idx = np.searchsorted(edges, x)
        for j in range(levels):
            m = idx==j
            if m.any(): c[j] = x[m].mean()
    edges = (c[:-1]+c[1:])/2
    return c[np.searchsorted(edges, x)]

def assign(V, C, ch=40000):
    return np.concatenate([((V[i:i+ch]**2).sum(1)[:,None]-2*V[i:i+ch]@C.T+(C**2).sum(1)[None]).argmin(1)
                           for i in range(0, len(V), ch)])
def kmeans(V, k, it=12):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m=a==j
            if m.any(): C[j]=V[m].mean(0)
    return C
def rvq_recon(V, stages_k):
    q = np.zeros_like(V); r = V.copy()
    for k in stages_k:
        C = kmeans(r, k); a = assign(r, C); q += C[a]; r -= C[a]
    return q

# --- optimal scalar at BPW bits/weight = 2^BPW levels per weight ---
levels = 2**BPW
Vq_scalar = lloyd_max(Vn.reshape(-1), levels).reshape(O, I)
Wq_scalar = Ro.T @ (Vq_scalar*s) @ Ri.T

# --- our k-means RVQ at BPW bpw = BPW stages of K=256 (each stage=1 bpw over D=8) ---
V8 = Vn.reshape(-1, D)
Vq_vq = rvq_recon(V8, [256]*BPW).reshape(O, I)
Wq_vq = Ro.T @ (Vq_vq*s) @ Ri.T

sc, vq = osqnr(Wq_scalar), osqnr(Wq_vq)
print(f"tensor {key}  matched {BPW} bpw")
print(f"  optimal scalar (Lloyd-Max, {levels} levels): {sc:.2f} dB   <- shape gain")
print(f"  our k-means RVQ (D=8, {BPW}x256)          : {vq:.2f} dB")
print(f"  space-filling gain REALIZED by VQ over scalar: {vq-sc:+.2f} dB   (D=8 ceiling ~0.65 dB)")
if vq-sc < 0.2:
    print("  => VQ is NOT beating optimal scalar: memoryless D=8 VQ is a dead end here.")
elif vq-sc < 0.65:
    print("  => VQ captures part of the 0.65 dB ceiling; near the D=8 limit -> need trellis / higher-D / partial rotation.")
else:
    print("  => VQ EXCEEDS the i.i.d. D=8 ceiling -> rotation left exploitable correlation (memory gain).")

# --- isotropy check: condition number of rotated 8x8 Hessian diagonal blocks ---
Hr = Ri.T @ H @ Ri
conds = []
for j in range(0, I - D + 1, D):
    blk = Hr[j:j+D, j:j+D]
    ev = np.linalg.eigvalsh(blk); conds.append(ev[-1]/max(ev[0], 1e-12))
conds = np.array(conds)
print(f"\nrotated 8x8 Hessian-block condition number: median {np.median(conds):.2f}, "
      f"p90 {np.percentile(conds,90):.2f}")
print("  => " + ("~isotropic: Hessian-Mahalanobis assignment ~= Euclidean, little to gain."
                  if np.median(conds) < 3 else
                  "anisotropic: a Hessian-aware assignment metric could help."))
