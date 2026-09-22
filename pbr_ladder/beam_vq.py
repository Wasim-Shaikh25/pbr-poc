# Beam-search RVQ assignment vs greedy, on a real rotated tensor (output SQNR).
# Same stored bits (same codebooks, same stage sizes) -- pure search improvement,
# the way AQLM keeps top-B partial code combinations across stages instead of the
# greedy nearest-per-stage the current pipeline uses.
#
# Usage: python beam_vq.py <weight_key> <acts.npy> [stages] [beam]
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
STAGES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "256,256,64").split(",")]
BEAM = int(sys.argv[4]) if len(sys.argv) > 4 else 4
D = 8; CH = 40000
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
H = A.T @ A / len(A)
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q*np.sign(np.diag(r)))
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri
s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
V = (Wr/s).reshape(-1, D)
def assign(Vc, C): return ((Vc**2).sum(1)[:,None]-2*Vc@C.T+(C**2).sum(1)[None]).argmin(1)
def kmeans(Vv, k, it=10):
    C = Vv[rng.choice(len(Vv), k, replace=False)].copy()
    for _ in range(it):
        a = np.concatenate([assign(Vv[i:i+CH], C) for i in range(0, len(Vv), CH)])
        for j in range(k):
            m = a==j
            if m.any(): C[j] = Vv[m].mean(0)
    return C
# fit codebooks (greedy successive refinement) -- SAME books for both decoders
books, R = [], V.copy()
for k in STAGES:
    C = kmeans(R, k); a = np.concatenate([assign(R[i:i+CH], C) for i in range(0, len(R), CH)])
    books.append(C); R = R - C[a]

def greedy(V):
    q = np.zeros_like(V); r = V.copy()
    for C in books:
        a = np.concatenate([assign(r[i:i+CH], C) for i in range(0, len(r), CH)])
        q += C[a]; r -= C[a]
    return q

def beam(V, B):
    """keep B best cumulative reconstructions across stages, per vector (chunked)."""
    out = np.zeros_like(V)
    for i in range(0, len(V), CH):
        Vc = V[i:i+CH]; n = len(Vc)
        # beam state: (n, b, D) partial reconstructions and their residual sq-norms
        recon = np.zeros((n, 1, D));
        for si, C in enumerate(books):
            b = recon.shape[1]; k = len(C)
            # candidate reconstructions: recon[:,bi,:] + C[k]  -> (n, b, k, D)
            cand = recon[:, :, None, :] + C[None, None, :, :]
            res = Vc[:, None, None, :] - cand
            d = (res**2).sum(-1)                      # (n, b, k)
            d = d.reshape(n, b*k)
            keep = min(B, b*k)
            idx = np.argpartition(d, keep-1, axis=1)[:, :keep]   # (n, keep)
            cand = cand.reshape(n, b*k, D)
            recon = np.take_along_axis(cand, idx[:, :, None], axis=1)  # (n, keep, D)
        # final: best of the B
        fd = ((Vc[:, None, :] - recon)**2).sum(-1)   # (n, keep)
        best = recon[np.arange(n), fd.argmin(1)]
        out[i:i+CH] = best
    return out

Wq_g = (greedy(V).reshape(O, I))*s
Wq_b = (beam(V, BEAM).reshape(O, I))*s
sg, sb = osqnr(Ro.T@Wq_g@Ri.T), osqnr(Ro.T@Wq_b@Ri.T)
bpw = sum(np.log2(k) for k in STAGES)/D
print(f"tensor {key}  stages {STAGES}  ({bpw:.3f} bpw, identical for both)")
print(f"  greedy assignment : {sg:.2f} dB")
print(f"  beam={BEAM} assignment: {sb:.2f} dB   ({sb-sg:+.2f} dB, FREE -- same bits)")
