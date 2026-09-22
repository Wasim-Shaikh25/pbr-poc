# Coherent error accumulation probe.
#
# Hypothesis (cross-domain, from wave physics / random-walk statistics):
#   The whole-model PBR failure (+36% PPL from 24 layers, each individually <1%)
#   is driven not by per-layer QUALITY but by how per-layer output ERRORS add in
#   the shared residual stream. quantize_full_model.py caches ONE rotation per
#   tensor width and reuses it for every layer of that width. If that makes the
#   per-layer error vectors point in correlated directions, they sum COHERENTLY
#   (||sum||^2 ~ N^2 of the mean) instead of INCOHERENTLY (~ N). That is exactly
#   the kind of interaction the consolidated findings measured as "~28% of the
#   combined damage is interaction between chunks".
#
# down_proj is the clean probe: its OUTPUT dim == hidden size (896), so the error
# each layer's down_proj injects is a vector in the SAME residual space, and the
# saved activations are token-aligned across layers (same calib text, same rng
# seed 0 subsample in capture_activations.py). So we can literally add the error
# vectors and compare ||sum||^2 to sum(||.||^2).
#
# We compare the driver's SHARED rotation against a PER-LAYER rotation (seed keyed
# on layer index too) -- free to store either way (both regenerate from a seed).
#
# Usage:  python coherence_probe.py            # down_proj, layers 2/12/21
import os, sys, time, numpy as np
from safetensors import safe_open

CKPT = "./qwen05b/model.safetensors"
LAYERS = [2, 12, 21]
STAGES = [int(x) for x in os.environ.get("PBR_STAGES", "512,256,64").split(",")]
D = 8
PROJ = os.environ.get("PBR_PROJ", "mlp.down_proj")   # down_proj: output == residual space
ACT = {"mlp.down_proj": "down", "self_attn.q_proj": "q", "mlp.gate_proj": "gate"}[PROJ]

def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)

def kmeans(V, k, it=8, chunk=50000):
    rng = np.random.default_rng(0)
    C = V[rng.choice(len(V), k, replace=False)].copy()
    def assign(C):
        return np.concatenate([((V[i:i+chunk]**2).sum(1)[:, None] - 2*V[i:i+chunk]@C.T
                                 + (C**2).sum(1)[None]).argmin(1) for i in range(0, len(V), chunk)])
    for _ in range(it):
        a = assign(C)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = V[m].mean(0)
    return C

def quantize_linear(W, H, stages, Ro, Ri):
    """Exact copy of quantize_full_model.py's method, rotations passed in."""
    O, I = W.shape
    Wr = Ro @ W @ Ri
    Hr = Ri.T @ H @ Ri
    Hr = Hr + 0.01 * np.mean(np.diag(Hr)) * np.eye(I)
    U = np.linalg.cholesky(np.linalg.inv(Hr)).T
    s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
    pool = (Wr / s).reshape(-1, D)
    books, Rres = [], pool.copy()
    for k in stages:
        C = kmeans(Rres, k)
        a = np.concatenate([((Rres[i:i+50000]**2).sum(1)[:, None] - 2*Rres[i:i+50000]@C.T
                             + (C**2).sum(1)[None]).argmin(1) for i in range(0, len(Rres), 50000)])
        books.append(C); Rres = Rres - C[a]
    T = Wr.copy(); Q = np.zeros_like(T)
    for j in range(0, I, D):
        b = slice(j, j + D); R = T[:, b] / s; q = np.zeros_like(R)
        for C in books:
            a = ((R**2).sum(1)[:, None] - 2*R@C.T + (C**2).sum(1)[None]).argmin(1)
            q += C[a]; R -= C[a]
        Q[:, b] = q * s
        if j + D < I:
            T[:, j+D:] -= (T[:, b] - Q[:, b]) @ np.linalg.solve(U[b, b], U[b, j+D:])
    return Ro.T @ Q @ Ri.T

def run(mode):
    """mode: 'shared' (seed = width, driver default) or 'perlayer' (seed = width*1000+layer)."""
    errs, sqnrs = {}, {}
    for l in LAYERS:
        with safe_open(CKPT, "pt") as f:
            W = f.get_tensor(f"model.layers.{l}.{PROJ}.weight").float().numpy().astype(np.float64)
        A = np.load(f"acts_L{l}_{ACT}.npy").astype(np.float64)
        O, I = W.shape
        assert A.shape[1] == I, f"{A.shape} vs in={I}"
        H = A.T @ A / len(A)
        if mode == "shared":
            Ro, Ri = ortho(O, O), ortho(I, I)
        else:
            Ro, Ri = ortho(O, O*1000 + l), ortho(I, I*1000 + l)
        Wq = quantize_linear(W, H, STAGES, Ro, Ri)
        d = A @ (W - Wq).T                       # (tokens, O) token-aligned error into output
        errs[l] = d
        sig = (A @ W.T)**2
        sqnrs[l] = 10*np.log10(sig.sum() / (d**2).sum())
        print(f"  [{mode}] L{l:2d} {PROJ}  out_SQNR={sqnrs[l]:6.2f} dB  ||err||^2={ (d**2).sum():.4e}", flush=True)
    return errs, sqnrs

def report(mode, errs):
    ls = LAYERS
    p = {l: (errs[l]**2).sum() for l in ls}
    # pairwise coherence (Frobenius cosine of token-aligned error matrices)
    print(f"\n  [{mode}] pairwise error correlation (Frobenius cosine):")
    for i in range(len(ls)):
        for j in range(i+1, len(ls)):
            a, b = errs[ls[i]], errs[ls[j]]
            rho = (a*b).sum() / (np.sqrt((a**2).sum())*np.sqrt((b**2).sum()))
            print(f"    L{ls[i]} vs L{ls[j]}:  rho = {rho:+.4f}")
    # only down_proj outputs share the residual space -> coherent-sum test is meaningful
    if PROJ == "mlp.down_proj":
        S = sum(errs[l] for l in ls)
        coherent = (S**2).sum()
        incoherent = sum(p[l] for l in ls)
        amp = coherent / incoherent
        print(f"\n  [{mode}] residual-stream accumulation over {len(ls)} layers:")
        print(f"    sum of individual error powers  = {incoherent:.4e}   (incoherent / random-walk expectation)")
        print(f"    power of the SUMMED error       = {coherent:.4e}   (what the residual stream feels)")
        print(f"    amplification factor            = {amp:.3f}x   (1.0 = perfectly incoherent; >1 = coherent build-up)")
    return p

if __name__ == "__main__":
    print(f"PROJ={PROJ}  layers={LAYERS}  stages={STAGES}  index_bpw={sum(np.log2(k) for k in STAGES)/D:.3f}")
    for mode in ("shared", "perlayer"):
        t = time.time()
        print(f"\n=== rotation mode: {mode} ===")
        errs, sq = run(mode)
        report(mode, errs)
        print(f"  ({time.time()-t:.0f}s)")
