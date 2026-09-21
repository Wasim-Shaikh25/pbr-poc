# Push the node idea: 64 constants, combinations ("connections") as structures.
import sys, numpy as np
rng = np.random.default_rng(0)
if len(sys.argv) > 2:
    from safetensors import safe_open
    with safe_open(sys.argv[1], "pt") as f: W = f.get_tensor(sys.argv[2]).float().numpy()
else:
    W = (0.02 * rng.standard_t(6, size=(512, 896)) / np.sqrt(1.5)).astype(np.float32)
G = 64; X = W.reshape(-1, G); s = np.abs(X).max(1, keepdims=True); Z = X / s; N = X.size
def sqnr(Zh): return 10*np.log10((X**2).sum() / ((X - Zh*s)**2).sum())
def q1(z, b): return b[np.abs(z[..., None] - b).argmin(-1)]
def lloyd(z, k, it=30):
    c = np.quantile(z, (np.arange(k)+.5)/k)
    for _ in range(it):
        a = np.abs(z[:, None]-c).argmin(1)
        c = np.array([z[a==j].mean() if (a==j).any() else c[j] for j in range(k)])
    return np.sort(c)
R = []
R.append(("INT4 uniform (baseline)", 4+16/G, sqnr(q1(Z, np.linspace(-1,1,16)))))
# FP4 E2M1 levels: sign * {0,.5,1,1.5,2,3,4,6}/6  -> exponent/mantissa-shaped nodes
fp = np.array([0,.5,1,1.5,2,3,4,6])/6; fp4 = np.unique(np.concatenate([-fp, fp]))
R.append(("FP4 E2M1 nodes (exp/mant)", 4+16/G, sqnr(q1(Z, fp4))))
b16 = lloyd(Z.ravel()[::5], 16)
R.append(("16 learned nodes", 4+16/G, sqnr(q1(Z, b16))))
# Method C extended: 64 nodes... tile picks one of K books (menus)
def menus(K, nodes_per=16):
    bk = np.stack([b16*f for f in np.linspace(.75, 1.15, K)])
    for _ in range(6):
        e = np.stack([((Z-q1(Z,b))**2).sum(1) for b in bk]); p = e.argmin(0)
        bk = np.stack([lloyd(Z[p==k].ravel()[::2], nodes_per) if (p==k).sum()>30 else bk[k] for k in range(K)])
    e = np.stack([((Z-q1(Z,b))**2).sum(1) for b in bk]); p = e.argmin(0)
    Zh = np.empty_like(Z)
    for k in range(K): Zh[p==k] = q1(Z[p==k], bk[k])
    return Zh
R.append(("4 menus x16 (=64 nodes)", 4+(16+2)/G, sqnr(menus(4))))
# PAIRS: adjacent weights share one 8-bit code; 64 numbers = 2 books x 16 2-D vectors
P = Z.reshape(-1, 2)
def additive_pairs(k=16, it=12):
    A = np.stack([lloyd(P[:,0][::5], k), lloyd(P[:,1][::5], k)], 1) * 0.7
    B = rng.normal(0, .05, (k, 2))
    combos = (A[:, None, :] + B[None, :, :]).reshape(-1, 2)          # 256 structures
    for _ in range(it):
        combos = (A[:, None, :] + B[None, :, :]).reshape(-1, 2)
        idx = np.concatenate([((P[i:i+20000, None, :]-combos)**2).sum(-1).argmin(1)
                              for i in range(0, len(P), 20000)])
        a, b = idx // k, idx % k
        for j in range(k):
            if (a==j).any(): A[j] = (P[a==j] - B[b[a==j]]).mean(0)
        for j in range(k):
            if (b==j).any(): B[j] = (P[b==j] - A[a[b==j]]).mean(0)
    return combos[idx].reshape(Z.shape)
R.append(("pairs: 2x16 vecs (64 nums)", 4+16/G+64*16/N, sqnr(additive_pairs())))
# Full pair codebook (256 free 2-D vectors = 512 numbers) as upper reference
def vq(d, k, it=15):
    V = Z.reshape(-1, d); C = V[rng.choice(len(V), k, replace=False)]
    for _ in range(it):
        idx = np.concatenate([((V[i:i+20000,None]-C)**2).sum(-1).argmin(1) for i in range(0,len(V),20000)])
        C = np.stack([V[idx==j].mean(0) if (idx==j).any() else C[j] for j in range(k)])
    return C[idx].reshape(Z.shape), k*d
Zh, n = vq(2, 256); R.append((f"pairs: 256 free vecs ({n} nums)", 4+16/G+n*16/N, sqnr(Zh)))
# Push lower: fewer bits per weight
Zh, n = vq(2, 64);  R.append((f"pairs 6-bit code ({n} nums)", 3+16/G+n*16/N, sqnr(Zh)))
R.append(("INT3 uniform", 3+16/G, sqnr(q1(Z, np.linspace(-1,1,8)))))
Zh, n = vq(4, 256); R.append((f"quads 8-bit code ({n} nums)", 2+16/G+n*16/N, sqnr(Zh)))
R.append(("INT2 uniform", 2+16/G, sqnr(q1(Z, np.linspace(-1,1,4)))))
print(f"{'method':<32}{'bpw':>7}{'SQNR dB':>9}")
for n_, b, q in R: print(f"{n_:<32}{b:7.3f}{q:9.2f}")
