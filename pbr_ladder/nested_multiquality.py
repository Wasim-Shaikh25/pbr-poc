# Nested multi-quality single file, on a REAL Qwen tensor (the market differentiator).
# Encode ONCE with successive-refinement RVQ; decode at 1/2/3-bit tiers by truncating
# stages. Compare each tier's output SQNR to a DEDICATED file fit for that exact bpw.
# Small penalty => one file serves all tiers (vs storing 3 separate files at 1+2+3=6 bpw).
#
# Usage: python nested_multiquality.py <weight_key> <acts.npy>
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
D, K, CH = 8, 256, 40000
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
V = (Wr / s).reshape(-1, D)
def assign(Vv, C): return np.concatenate([((Vv[i:i+CH]**2).sum(1)[:,None]-2*Vv[i:i+CH]@C.T+(C**2).sum(1)[None]).argmin(1) for i in range(0,len(Vv),CH)])
def kmeans(Vv, k, it=12):
    C = Vv[rng.choice(len(Vv), k, replace=False)].copy()
    for _ in range(it):
        a = assign(Vv, C)
        for j in range(k):
            m=a==j
            if m.any(): C[j]=Vv[m].mean(0)
    return C
def recon_from_books(books):
    q = np.zeros_like(V); r = V.copy()
    outs = []
    for C in books:
        a = assign(r, C); q = q + C[a]; r = V - q; outs.append(q.copy())
    return outs                                   # outs[t] = reconstruction using first t+1 stages

def to_W(recon): return Ro.T @ (recon.reshape(O, I)*s) @ Ri.T

# --- NESTED: fit 3-stage successive-refinement codebook ONCE, decode at each tier ---
books, r = [], V.copy()
for _ in range(3):
    C = kmeans(r, K); books.append(C); r = r - C[assign(r, C)]
nested = recon_from_books(books)                  # [tier1(1bpw), tier2(2bpw), tier3(3bpw)]
nested_sqnr = [osqnr(to_W(nested[t])) for t in range(3)]

# --- DEDICATED: fit a fresh RVQ specifically for each target bpw ---
def dedicated(nstages):
    bb, rr = [], V.copy()
    for _ in range(nstages):
        C = kmeans(rr, K); bb.append(C); rr = rr - C[assign(rr, C)]
    q = np.zeros_like(V); rr = V.copy()
    for C in bb:
        a = assign(rr, C); q += C[a]; rr -= C[a]
    return osqnr(to_W(q))
ded_sqnr = [dedicated(t+1) for t in range(3)]

print(f"tensor {key}  ({O}x{I})")
print(f"\n{'tier':<8}{'bpw':>6}{'nested(1 file) dB':>20}{'dedicated dB':>15}{'penalty':>10}")
for t in range(3):
    print(f"tier{t+1:<4}{t+1:6d}{nested_sqnr[t]:20.2f}{ded_sqnr[t]:15.2f}{ded_sqnr[t]-nested_sqnr[t]:+10.2f}")
print(f"\nStorage: one nested file = 3 bpw serves all 3 tiers.")
print(f"         three dedicated files = 1+2+3 = 6 bpw. Nested saves 50% IF penalty is small.")
avg_pen = np.mean([ded_sqnr[t]-nested_sqnr[t] for t in range(3)])
print(f"mean penalty across tiers: {avg_pen:+.2f} dB  "
      f"({'small -> product works' if avg_pen < 1.0 else 'large -> nesting costs too much'})")
