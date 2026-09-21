# Lossy "formula evolves": w_hat[i,j] = nodes[book(t)*16 + c[i,j]] * scale(t)
# c in 0..15 (4-bit), <=64 shared nodes, per-tile scale (+ optional book id).
# Compared against plain INT4 at matched bpw. Score = SQNR (dB) and bpw.
import sys, numpy as np
rng = np.random.default_rng(0)
if len(sys.argv) > 2:
    from safetensors import safe_open
    with safe_open(sys.argv[1], "pt") as f: W = f.get_tensor(sys.argv[2]).float().numpy()
else:  # surrogate: Qwen-like heavy-tailed gate_proj slice
    W = (0.02 * rng.standard_t(6, size=(512, 896)) / np.sqrt(1.5)).astype(np.float32)
G = 64                                   # weights per tile/group (8x8 nodes)
X = W.reshape(-1, G)                     # flat groups (row-major 1x64 strips)
s = np.abs(X).max(1, keepdims=True)      # per-tile scale (stored bf16 = 16 bits)
Z = X / s                                # normalized to [-1, 1]
def sqnr(Wh): return 10*np.log10((X**2).sum() / ((X - Wh)**2).sum())
def lloyd(z, k=16, it=30):
    c = np.quantile(z, (np.arange(k)+.5)/k)
    for _ in range(it):
        a = np.abs(z[:, None] - c[None]).argmin(1)
        c = np.array([z[a == j].mean() if (a == j).any() else c[j] for j in range(k)])
    return np.sort(c)
def quant(z, book):                      # nearest node per weight
    return book[np.abs(z[..., None] - book).argmin(-1)]
res = []
# A) plain INT4 symmetric, 16 levels uniform in [-1,1]
uni = np.linspace(-1, 1, 16)
res.append(("A INT4 uniform", 4 + 16/G, sqnr(quant(Z, uni) * s)))
# B) one shared 16-node codebook (Lloyd), F_c = node[c]*scale
b1 = lloyd(Z.ravel()[::7])
res.append(("B 16 shared nodes", 4 + 16/G + 16*16/X.size, sqnr(quant(Z, b1) * s)))
# C) node combinations: 64 nodes = 4 books x 16; each tile picks a book (2 bits)
books = np.stack([lloyd(Z.ravel()[::7]) * f for f in (0.8, 0.9, 1.0, 1.1)])
for _ in range(8):
    err = np.stack([((Z - quant(Z, bk))**2).sum(1) for bk in books])
    pick = err.argmin(0)
    books = np.stack([lloyd(Z[pick == k].ravel()[::3]) if (pick == k).sum() > 20 else books[k] for k in range(4)])
err = np.stack([((Z - quant(Z, bk))**2).sum(1) for bk in books]); pick = err.argmin(0)
Wc = np.stack([quant(Z[t], books[pick[t]]) for t in range(len(Z))]) * s
res.append(("C 64 nodes, 4 books", 4 + (16+2)/G + 64*16/X.size, sqnr(Wc)))
# D) C + position term: F = (node + b*pos)*scale  (does position add anything?)
pos = (np.arange(G) - G/2) / G
bfit = ((Z - Wc/s) * pos).sum(1, keepdims=True) / (pos**2).sum()
bq = np.round(bfit * 16) / 16            # 4-bit-ish slope per tile
res.append(("D C + pos slope", 4 + (16+2+5)/G + 64*16/X.size, sqnr((Wc/s + bq*pos) * s)))
# reference: INT5 / INT6 uniform
for b in (5, 6):
    res.append((f"ref INT{b}", b + 16/G, sqnr(quant(Z, np.linspace(-1, 1, 2**b)) * s)))
codes = np.abs(Z[..., None] - b1).argmin(-1); p = np.bincount(codes.ravel(), minlength=16)/codes.size
print(f"entropy of 4-bit codes (method B) = {-(p[p>0]*np.log2(p[p>0])).sum():.3f} bits")
print(f"{'method':<22}{'bpw':>7}{'SQNR dB':>9}")
for n, b, q in res: print(f"{n:<22}{b:7.3f}{q:9.2f}")
