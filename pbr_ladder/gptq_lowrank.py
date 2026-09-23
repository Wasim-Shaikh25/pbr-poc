# DECISIVE test: does the activation low-rank correction ADD on top of GPTQ (which is
# already output-aware via H=A^T A)?  Build real GPTQ (int3) on the rotated tensor, then
# fit a rank-16 activation-weighted correction to ITS residual.  Kill if increment <0.5 dB.
#
# Usage: python gptq_lowrank.py <weight_key> <acts.npy>
import sys, numpy as np
from safetensors import safe_open
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n))); return q*np.sign(np.diag(r))
Ro, Ri = ortho(O, 1), ortho(I, 2)
Wr = Ro @ W @ Ri; Ar = A @ Ri

def gptq(Wr, Ar, bits):
    Wc = Wr.copy(); O, I = Wc.shape
    H = Ar.T @ Ar / len(Ar)
    damp = 0.01 * np.mean(np.diag(H))
    H[np.diag_indices_from(H)] += damp
    Hinv = np.linalg.inv(H)
    R = np.linalg.cholesky(Hinv).T                 # upper, R^T R = Hinv
    Q = np.zeros_like(Wc); L = (1 << bits) - 1
    for i in range(I):
        w = Wc[:, i].copy(); di = R[i, i]
        lo, hi = w.min(), w.max()
        sc = (hi - lo) / L if hi > lo else 1.0
        q = lo + np.round((w - lo) / sc) * sc
        Q[:, i] = q
        err = (w - q) / di
        if i + 1 < I:
            Wc[:, i+1:] -= np.outer(err, R[i, i+1:])
    return Q

# activation eigenbasis for low-rank correction
G = A.T @ A / len(A)
ev, U_g = np.linalg.eigh(G); ev = ev[::-1]; U_g = U_g[:, ::-1]
def q8(X):
    m = np.abs(X).max(); L = 127
    return np.round(X / m * L) * (m / L) if m > 0 else X
def lowrank_corr_quant(E, r):
    sq = np.sqrt(np.clip(ev, 0, None))
    B = (E @ U_g) * sq[None, :]
    P, S, Qt = np.linalg.svd(B, full_matrices=False)
    r = min(r, len(S)); inv = np.where(sq > 1e-9, 1.0 / sq, 0.0)
    C = ((P[:, :r] * S[:r]) @ Qt[:r] * inv[None, :]) @ U_g.T
    Pc, Sc, Qc = np.linalg.svd(C, full_matrices=False)
    Uf = q8(Pc[:, :r] * np.sqrt(Sc[:r])); Vf = q8(np.sqrt(Sc[:r])[:, None] * Qc[:r])
    return Uf @ Vf

print(f"tensor {key} ({O}x{I})")
for bits in (3, 4):
    Qg = gptq(Wr, Ar, bits)
    Wq_g = Ro.T @ Qg @ Ri.T
    sq_g = osqnr(Wq_g)
    E = W - Wq_g
    r = 16; Cq = lowrank_corr_quant(E, r)
    sq_gl = osqnr(Wq_g + Cq)
    dbpw = r * (O + I) * 8.0 / (O * I)
    print(f"\nint{bits}+GPTQ            @ {float(bits):.3f} bpw : {sq_g:.2f} dB")
    print(f"int{bits}+GPTQ + rank{r} corr @ {bits+dbpw:.3f} bpw : {sq_gl:.2f} dB   "
          f"(increment {sq_gl-sq_g:+.2f} dB for +{dbpw:.3f} bpw)")
print("\nincrement >0.5 dB on top of GPTQ => low-rank correction is genuinely ADDITIVE (new tool).")
print("increment ~0 => GPTQ already harvests the activation directions; not new.")
