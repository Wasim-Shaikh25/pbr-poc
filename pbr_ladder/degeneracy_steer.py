# Degeneracy-manifold cross-layer steering (physics agent's untested idea).
# Beam search yields B near-equal-distortion reconstructions per D-block. Given an
# accumulated drift e entering this layer, pick among those candidates the ones whose
# output error CANCELS e -- free (distortion-neutral) drift reduction. If the beam
# manifold is too narrow (beam only gave +0.65 dB), there's little freedom and no gain.
#
# Measures: ||E_base + e|| vs ||E_steered + e|| for drift magnitudes matching 1/4/9
# accumulated layers. Reduction = free cancellation available.
#
# Usage: python degeneracy_steer.py <weight_key> <acts.npy> [beam]
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
D, K, CH = 8, 256, 40000
STAGES = [256, 256, 256]
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n))); return q*np.sign(np.diag(r))
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri
s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
Ar = A @ Ri                                     # rotated activations: E = Ar @ (Wr - Q).T
def assign(V, C): return np.concatenate([((V[i:i+CH]**2).sum(1)[:,None]-2*V[i:i+CH]@C.T+(C**2).sum(1)[None]).argmin(1) for i in range(0,len(V),CH)])
def kmeans(V, k, it=12):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m=a==j
            if m.any(): C[j]=V[m].mean(0)
    return C
Vpool = (Wr/s).reshape(-1, D)
books, r = [], Vpool.copy()
for k in STAGES:
    C = kmeans(r, k); books.append(C); r = r - C[assign(r, C)]

def beam_candidates(Rb, Bn):
    """Return (recon (n,Bn,D), dist (n,Bn)) of the Bn best cumulative combos per row."""
    n = Rb.shape[0]; recon = np.zeros((n, 1, D))
    for C in books:
        bb = recon.shape[1]; kk = len(C)
        cand = recon[:, :, None, :] + C[None, None, :, :]
        d = ((Rb[:, None, None, :] - cand)**2).sum(-1).reshape(n, bb*kk)
        keep = min(Bn, bb*kk); idx = np.argpartition(d, keep-1, axis=1)[:, :keep]
        recon = np.take_along_axis(cand.reshape(n, bb*kk, D), idx[:, :, None], axis=1)
    d = ((Rb[:, None, :] - recon)**2).sum(-1)
    return recon, d

# assemble baseline (min-dist) and keep candidate errors per block for steering
Q_base = np.zeros_like(Wr)
blocks = list(range(0, I, D))
cand_store = {}                                  # block -> (recon (O,B,D), dist (O,B))
for j in blocks:
    Rb = Wr[:, j:j+D] / s
    recon, d = beam_candidates(Rb, B)
    cand_store[j] = (recon, d)
    Q_base[:, j:j+D] = recon[np.arange(O), d.argmin(1)] * s
E_base = Ar @ (Wr - Q_base).T                    # (tokens, O)

rms = np.sqrt((E_base**2).mean())
print(f"tensor {key}  beam={B}  baseline ||E|| rms={rms:.4e}")
print(f"{'drift (layers)':>16}{'||E+e|| base':>16}{'||E+e|| steer':>16}{'free reduction':>16}")
print(f"{'drift(layers)':>14}{'||E+e|| base':>15}{'||E+e|| steer':>15}{'reduction':>11}{'wdist x':>9}")
for nlayers in (1, 4, 9):
    e = rng.normal(size=E_base.shape); e *= (np.sqrt(nlayers)*rms) / np.sqrt((e**2).mean())  # incoherent drift ~ sqrt(N)
    Q_st = np.zeros_like(Wr); wd_base = 0.0; wd_st = 0.0
    T = e.copy()                                 # running residual drift to cancel (SEQUENTIAL)
    for j in blocks:
        recon, d = cand_store[j]                 # recon (O,B,D)
        Wb = Wr[:, j:j+D]; Arb = Ar[:, j:j+D]
        G = Arb.T @ Arb                          # (D,D)
        diff = Wb[None, :, :] - recon.transpose(1, 0, 2) * s   # (B,O,D)
        out_dist = np.einsum('bod,de,boe->bo', diff, G, diff)  # (B,O) ||contribution||^2
        MT = Arb.T @ T                           # (D,O) for <contribution, remaining drift>
        cross = np.einsum('bod,do->bo', diff, MT)              # (B,O)
        base_pick = out_dist.argmin(0)
        st_pick = (out_dist + 2.0*cross).argmin(0)             # cancel the REMAINING residual
        chosen = recon[np.arange(O), st_pick]
        Q_st[:, j:j+D] = chosen * s
        T = T + Arb @ (Wb - chosen*s).T          # subtract this block's contribution from residual
        wd_base += out_dist[base_pick, np.arange(O)].sum()
        wd_st  += out_dist[st_pick,  np.arange(O)].sum()
    E_st = Ar @ (Wr - Q_st).T
    base_n = np.linalg.norm(E_base + e); st_n = np.linalg.norm(E_st + e)
    print(f"{nlayers:>14}{base_n:>15.4e}{st_n:>15.4e}{100*(1-st_n/base_n):>10.2f}%{wd_st/wd_base:>9.3f}")
print("\nreduction >5% with wdist x ~1 = FREE win; ~0% = manifold too narrow; wdist x >>1 = not free.")
