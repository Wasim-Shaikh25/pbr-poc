# NEW: make rotation + mixed-precision work TOGETHER via BLOCK-LOCAL rotation.
# Global rotation flattens all importance (nothing left to allocate). Instead rotate
# only WITHIN input-column groups (block-diagonal Ri): tames within-group outliers but
# PRESERVES between-group variance -> mixed precision then allocates bits across groups.
# The two attack different structure and can stack. Bar to beat: global-rot + flat VQ.
#
# Usage: python block_rot_mixed.py <weight_key> <acts.npy> [target_bpw] [group_size]
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
G = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BITS = [1, 2, 3, 4, 6]
D = 8
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n))); return q*np.sign(np.diag(r))
def assign(V, C, ch=40000):
    return np.concatenate([((V[i:i+ch]**2).sum(1)[:,None]-2*V[i:i+ch]@C.T+(C**2).sum(1)[None]).argmin(1) for i in range(0,len(V),ch)])
def kmeans(V, k, it=8):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = V[m].mean(0)
    return C
def vq_recon(V, nst):                                  # nst stages of 256 = nst bpw
    q = np.zeros_like(V); r = V.copy()
    for _ in range(nst):
        C = kmeans(r, 256); a = assign(r, C); q += C[a]; r -= C[a]
    return q
def qscalar(x, b):
    lo, hi = x.min(), x.max()
    if hi <= lo: return x.copy()
    L = (1 << b) - 1; s = (hi - lo) / L
    return lo + np.round((x - lo) / s) * s

Ro = ortho(O, 1)
# ---- baselines: GLOBAL input rotation ----
Rg = ortho(I, 2)
WrG = Ro @ W @ Rg; sG = np.sqrt((WrG**2).mean(1, keepdims=True)) + 1e-12
# global-rot + flat scalar @ round(TARGET)
bflat = int(round(TARGET))
QgS = np.empty_like(WrG)
for i in range(I): QgS[:, i] = qscalar(WrG[:, i], bflat)
sq_gflat_s = osqnr(Ro.T @ QgS @ Rg.T)
# global-rot + flat VQ @ TARGET
VG = (WrG / sG).reshape(-1, D); QgV = vq_recon(VG, int(round(TARGET))).reshape(O, I) * sG
sq_gflat_vq = osqnr(Ro.T @ QgV @ Rg.T)

# ---- NEW: BLOCK-diagonal input rotation ----
assert I % G == 0, f"I={I} not divisible by group {G}"
ngrp = I // G
Rb = np.zeros((I, I))
for gi in range(ngrp):
    Rb[gi*G:(gi+1)*G, gi*G:(gi+1)*G] = ortho(G, 100 + gi)
WrB = Ro @ W @ Rb; sB = np.sqrt((WrB**2).mean(1, keepdims=True)) + 1e-12
ArB = A @ Rb                                            # output err = ArB @ (WrB - Q).T

# per-group importance (variance surviving block rotation) + reverse water-filling via greedy
grp_cols = [range(gi*G, (gi+1)*G) for gi in range(ngrp)]
gvar = np.array([WrB[:, list(c)].var() for c in grp_cols])
print(f"between-group variance spread after BLOCK rotation: "
      f"min {gvar.min():.3e}  max {gvar.max():.3e}  ratio {gvar.max()/gvar.min():.2f}x")

# distortion of quantizing group gi at b bits (scalar), output-domain contribution
def grp_scalar_recon(gi, b):
    cols = list(grp_cols[gi]); Q = np.empty((O, G))
    for j, c in enumerate(cols): Q[:, j] = qscalar(WrB[:, c], b)
    return Q, cols
def grp_vq_recon(gi, nst):
    cols = list(grp_cols[gi])
    sub = WrB[:, cols] / sB
    Vv = sub.reshape(-1, D); q = vq_recon(Vv, nst).reshape(O, G) * sB
    return q, cols
def grp_out_err(Q, cols):
    return ((ArB[:, cols] @ (WrB[:, cols] - Q).T) ** 2).sum()

def mixed_alloc(recon_fn):
    # precompute distortion per group per bit
    Dm = np.zeros((ngrp, len(BITS))); store = {}
    for gi in range(ngrp):
        for k, b in enumerate(BITS):
            Q, cols = recon_fn(gi, b); store[(gi, k)] = Q
            Dm[gi, k] = grp_out_err(Q, cols)
    lvl = np.zeros(ngrp, dtype=int); budget = TARGET * I
    def tot(l): return sum(BITS[k]*G for k in l)
    while True:
        cur = tot(lvl); bi, bg = -1, -1.0
        for gi in range(ngrp):
            k = lvl[gi]
            if k+1 >= len(BITS): continue
            if cur - BITS[k]*G + BITS[k+1]*G > budget: continue
            g = (Dm[gi, k] - Dm[gi, k+1]) / ((BITS[k+1]-BITS[k])*G)
            if g > bg: bg, bi = g, gi
        if bi < 0: break
        lvl[bi] += 1
    bpw = tot(lvl)/I
    Qfull = np.empty_like(WrB)
    for gi in range(ngrp):
        Q = store[(gi, lvl[gi])]; Qfull[:, list(grp_cols[gi])] = Q
    return bpw, osqnr(Ro.T @ Qfull @ Rb.T), [BITS[k] for k in lvl]

bpw_s, sq_bs, alloc_s = mixed_alloc(grp_scalar_recon)
bpw_v, sq_bv, alloc_v = mixed_alloc(grp_vq_recon)

print(f"\ntensor {key}  ({O}x{I})  target {TARGET} bpw  group={G} ({ngrp} groups)")
print(f"{'method':<44}{'bpw':>7}{'SQNR dB':>10}")
print(f"{'GLOBAL-rot + flat scalar':<44}{float(bflat):7.3f}{sq_gflat_s:10.2f}")
print(f"{'GLOBAL-rot + flat VQ  (CHAMPION to beat)':<44}{float(round(TARGET)):7.3f}{sq_gflat_vq:10.2f}")
print(f"{'BLOCK-rot + MIXED scalar (new)':<44}{bpw_s:7.3f}{sq_bs:10.2f}")
print(f"{'BLOCK-rot + MIXED VQ (new)':<44}{bpw_v:7.3f}{sq_bv:10.2f}")
print(f"\nblock-mixed-VQ  vs  global-flat-VQ : {sq_bv - sq_gflat_vq:+.2f} dB  <-- does combining WIN?")
print(f"group bit allocation (VQ): {alloc_v}")
