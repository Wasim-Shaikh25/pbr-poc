# "Adaptive-first, rotate-second" done coherently: split off the high-importance
# input channels (original basis, where outliers are still concentrated) and keep
# them at HIGH bits UNROTATED; rotate + low-bit VQ only the bulk. Compare to a flat
# rotated recipe at the SAME average bpw. Metric: real output SQNR.
#
# Usage: python outlier_split.py <weight_key> <acts.npy> [outlier_frac] [bulk_stages] [outlier_bits]
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key = sys.argv[1]; actf = sys.argv[2]
P = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01      # outlier column fraction
BULK_STAGES = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "256,256").split(",")]
OBITS = int(sys.argv[5]) if len(sys.argv) > 5 else 8       # bits for outlier channels
D, K = 8, 256

with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64)
O, I = W.shape; H = A.T @ A / len(A); hdiag = np.diag(H)
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q*np.sign(np.diag(r))).astype(np.float64)
def assign(V, C): return ((V**2).sum(1)[:,None]-2*V@C.T+(C**2).sum(1)[None]).argmin(1)
def kmeans(V, k, it=10):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m = a==j
            if m.any(): C[j] = V[m].mean(0)
    return C
def rtn(M, bits):
    L = 2**bits; s = np.abs(M).max(1, keepdims=True)/((L-1)/2) + 1e-12
    return (np.clip(np.round(M/s-.5), -L/2, L/2-1)+.5)*s
def quant_rot_vq(Wsub, Hsub, stages, seed_off=0):
    """rotate (rows Ro, cols Ri) + global VQ + GPTQ feedback on the given submatrix."""
    o, i = Wsub.shape
    Ro, Ri = ortho(o, o+seed_off), ortho(i, i+seed_off+7)
    Wr = Ro@Wsub@Ri; Hr = Ri.T@Hsub@Ri; Hr = Hr + 0.01*np.mean(np.diag(Hr))*np.eye(i)
    U = np.linalg.cholesky(np.linalg.inv(Hr)).T
    s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
    pool = (Wr/s).reshape(-1, D); books=[]; R=pool.copy()
    for k in stages:
        C = kmeans(R, k); a = assign(R, C); books.append(C); R = R-C[a]
    T = Wr.copy(); Q = np.zeros_like(T)
    for j in range(0, i, D):
        b = slice(j, j+D); r = T[:,b]/s; q = np.zeros_like(r)
        for C in books:
            a = assign(r, C); q += C[a]; r -= C[a]
        Q[:,b] = q*s
        if j+D < i: T[:,j+D:] -= (T[:,b]-Q[:,b])@np.linalg.solve(U[b,b], U[b,j+D:])
    return Ro.T@Q@Ri.T
def index_bpw(stages): return sum(np.log2(k) for k in stages)/D

# ---- flat rotated baseline over ALL columns ----
Wq_flat = quant_rot_vq(W, H, BULK_STAGES)
sqnr_flat = osqnr(Wq_flat); bpw_flat = index_bpw(BULK_STAGES)

# ---- outlier-split: top-P% input channels by original-basis importance -> 8-bit unrotated ----
n_out = max(D, int(round(P*I/D))*D)   # keep multiple of D so bulk width stays divisible
out_cols = np.argsort(-hdiag)[:n_out]
bulk_cols = np.setdiff1d(np.arange(I), out_cols)
Wq = np.zeros_like(W)
Wq[:, out_cols] = rtn(W[:, out_cols], OBITS)                       # outliers: high-bit, unrotated
Hb = H[np.ix_(bulk_cols, bulk_cols)]
Wq[:, bulk_cols] = quant_rot_vq(W[:, bulk_cols], Hb, BULK_STAGES)  # bulk: rotate + low-bit VQ
sqnr_split = osqnr(Wq)
# honest bpw: bulk index over (I-n_out) cols + outlier bits over n_out cols + which-cols index (I-bit mask)
idx_bits = n_out * int(np.ceil(np.log2(I)))   # store which columns are outliers, as indices
bpw_split = (index_bpw(BULK_STAGES)*(I-n_out) + OBITS*n_out + idx_bits) / I

print(f"tensor {key}  ({O}x{I})   outlier frac {P:.3%} = {n_out} cols   bulk stages {BULK_STAGES}   outlier {OBITS}-bit")
print(f"\n{'scheme':<34}{'bpw':>8}{'out SQNR dB':>13}")
print(f"{'flat rotated ('+','.join(map(str,BULK_STAGES))+')':<34}{bpw_flat:8.3f}{sqnr_flat:13.2f}")
print(f"{'outlier-split (adaptive->rotate)':<34}{bpw_split:8.3f}{sqnr_split:13.2f}")
print(f"\nsplit vs flat: {sqnr_split-sqnr_flat:+.2f} dB   for {bpw_split-bpw_flat:+.3f} bpw")
print(f"=> {(sqnr_split-sqnr_flat)/max(bpw_split-bpw_flat,1e-9):+.1f} dB per extra bpw "
      f"(compare: adding a whole VQ stage buys ~{index_bpw([256]):.0f} bpw for ~4-6 dB)")
