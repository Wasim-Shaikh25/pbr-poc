# GO/NO-GO for the agent's crack: activation-weighted low-rank correction of the RVQ
# residual.  Wq = rot+RVQ(W) + U V^T, with UV^T (rank r, 8-bit) fit to MINIMIZE the
# activation-metric residual ||A(E - UV^T)^T||^2, E = W - Wq_rvq.  Since the activation
# Gram is ~rank-9, a white VQ error cannot cancel the top activation directions but a
# low-rank term aligned to them can.  Honest test: does the correction beat spending the
# SAME extra bpw on more RVQ stages (interpolated)?  If not, it's not a better use of bits.
#
# Usage: python lowrank_actcorr.py <weight_key> <acts.npy>
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
D = 8
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n))); return q*np.sign(np.diag(r))
def assign(V, C, ch=40000):
    return np.concatenate([((V[i:i+ch]**2).sum(1)[:,None]-2*V[i:i+ch]@C.T+(C**2).sum(1)[None]).argmin(1) for i in range(0,len(V),ch)])
def kmeans(V, k, it=6):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = V[m].mean(0)
    return C
Ro, Ri = ortho(O, 1), ortho(I, 2)
Wr = Ro @ W @ Ri; s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
Vn = (Wr / s).reshape(-1, D)
def rvq(nst):
    q = np.zeros_like(Vn); r = Vn.copy()
    for _ in range(nst):
        C = kmeans(r, 256); a = assign(r, C); q += C[a]; r -= C[a]
    return Ro.T @ (q.reshape(O, I) * s) @ Ri.T
def q8(X):                                              # symmetric 8-bit per-matrix
    m = np.abs(X).max();  L = 127
    return np.round(X / m * L) * (m / L) if m > 0 else X

# activation-metric eigenbasis (rank of A)
G = A.T @ A / len(A)
ev, U_g = np.linalg.eigh(G)                              # ascending
ev = ev[::-1]; U_g = U_g[:, ::-1]
pr = (ev.sum()**2) / (ev**2).sum()
print(f"tensor {key} ({O}x{I})  activation participation ratio = {pr:.2f}")

def lowrank_corr(E, r):
    """best rank-r C minimizing ||A(E-C)^T||^2, via whitening in the G-metric."""
    sq = np.sqrt(np.clip(ev, 0, None))
    B = (E @ U_g) * sq[None, :]                          # E in whitened activation coords (O x I)
    P, S, Qt = np.linalg.svd(B, full_matrices=False)
    r = min(r, len(S))
    Br = (P[:, :r] * S[:r]) @ Qt[:r]                     # rank-r approx of B
    inv = np.where(sq > 1e-9, 1.0 / sq, 0.0)
    C = (Br * inv[None, :]) @ U_g.T                      # back to weight space
    U = P[:, :r] * S[:r]                                 # not used for storage; factorize C directly
    return C

# --- RVQ reference curve ---
sq3 = osqnr(rvq(3)); sq4 = osqnr(rvq(4))
print(f"\nRVQ 3-stage @3.000 bpw = {sq3:.2f} dB   |   RVQ 4-stage @4.000 bpw = {sq4:.2f} dB")
print(f"(RVQ slope ~{sq4-sq3:.2f} dB/bpw)\n")

Wq3 = rvq(3); E = W - Wq3
print(f"{'method':<40}{'bpw':>8}{'SQNR dB':>10}{'vs RVQ@bpw':>12}")
for r in (8, 16, 32, 64):
    C = lowrank_corr(E, r)
    # factorize C to rank-r 8-bit factors U(Oxr), Vt(rxI)
    Pc, Sc, Qc = np.linalg.svd(C, full_matrices=False)
    Uf = q8(Pc[:, :r] * np.sqrt(Sc[:r]))
    Vf = q8((np.sqrt(Sc[:r])[:, None]) * Qc[:r])
    Cq = Uf @ Vf
    Wq_new = Wq3 + Cq
    dbpw = r * (O + I) * 8.0 / (O * I)
    tot = 3.0 + dbpw
    sq_new = osqnr(Wq_new)
    rvq_equiv = sq3 + (sq4 - sq3) * (tot - 3.0)          # linear-interp RVQ at same bpw
    print(f"{'RVQ3 + rank-'+str(r)+' act-corr (8bit)':<40}{tot:>8.3f}{sq_new:>10.2f}{sq_new-rvq_equiv:>+12.2f}")
print("\n+ve last column => low-rank correction is a BETTER use of the same bits than more RVQ.")
