# Does mixed-precision allocation help AFTER rotation? (the user's "allocate then
# rotate/shrink" question, done the only way that's coherent: rotate FIRST, then give
# each ROTATED column the bits it needs.) If gain over flat-on-rotated ~0, rotation has
# already flattened importance and there is nothing left for per-column bits to exploit.
#
# Usage: python mixed_after_rotation.py <weight_key> <acts.npy> [target_bpw]
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
BITS = [1, 2, 3, 4, 6, 8]
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n))); return q*np.sign(np.diag(r))
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri                                    # rotated weights
Ar = A @ Ri                                         # so output error = Ar @ (Wr - Q).T
Hr = Ar.T @ Ar / len(Ar); hd = np.clip(np.diag(Hr), 1e-12, None)
def osqnr_r(Qr): return 10*np.log10(((Ar@Wr.T)**2).sum()/((Ar@(Wr-Qr).T)**2).sum())

def qcol(x, b):
    lo, hi = x.min(), x.max()
    if hi <= lo: return x.copy()
    L = (1 << b) - 1; s = (hi - lo) / L
    return lo + np.round((x - lo) / s) * s

# per-rotated-column distortion at each bit-width (Hessian-weighted)
Dmat = np.zeros((I, len(BITS))); Qc = {}
for i in range(I):
    col = Wr[:, i]
    for k, b in enumerate(BITS):
        q = qcol(col, b); Qc[(i, k)] = q
        Dmat[i, k] = hd[i] * ((col - q) ** 2).sum()

# greedy allocation to TARGET avg bits, on ROTATED columns
lvl = np.zeros(I, dtype=int); budget = TARGET * I
def tot(l): return sum(BITS[k] for k in l)
while True:
    cur = tot(lvl); bi, bg = -1, -1.0
    for i in range(I):
        k = lvl[i]
        if k + 1 >= len(BITS): continue
        if cur - BITS[k] + BITS[k+1] > budget: continue
        g = (Dmat[i, k] - Dmat[i, k+1]) / (BITS[k+1] - BITS[k])
        if g > bg: bg, bi = g, i
    if bi < 0: break
    lvl[bi] += 1
bpw = tot(lvl) / I
Qmix = np.empty_like(Wr)
for i in range(I): Qmix[:, i] = Qc[(i, lvl[i])]
sq_mix_rot = osqnr_r(Qmix)

# flat scalar on rotated, same (rounded) bpw
bflat = int(round(bpw)); Qflat = np.empty_like(Wr)
for i in range(I): Qflat[:, i] = qcol(Wr[:, i], bflat)
sq_flat_rot = osqnr_r(Qflat)

hist = {b: int((np.array([BITS[k] for k in lvl]) == b).sum()) for b in BITS}
print(f"tensor {key}  ({O}x{I})  target {TARGET} bpw   [ALL on ROTATED weights]")
print(f"rotated-col bit histogram: " + "  ".join(f"{b}b:{hist[b]}" for b in BITS))
print(f"\n{'method (on rotated W)':<40}{'bpw':>7}{'SQNR dB':>10}")
print(f"{'flat scalar':<40}{float(bflat):7.3f}{sq_flat_rot:10.2f}")
print(f"{'mixed precision (bits by need)':<40}{bpw:7.3f}{sq_mix_rot:10.2f}")
print(f"\nmixed - flat, AFTER rotation: {sq_mix_rot - sq_flat_rot:+.2f} dB")
print("~0 => rotation already flattened importance; per-column bits add nothing (order cannot help).")
