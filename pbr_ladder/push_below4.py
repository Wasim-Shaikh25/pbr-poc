# Push below 4 bpw: rotation + GPTQ error feedback + 8-D residual VQ (node structures).
# Scores: weight SQNR and OUTPUT SQNR  ||XW'||^2 / ||X(W-Wq)'||^2  (what the model feels).
import sys, os, time, numpy as np
rng = np.random.default_rng(0)
if len(sys.argv) > 2:
    from safetensors import safe_open
    with safe_open(sys.argv[1], "pt") as f: W = f.get_tensor(sys.argv[2]).float().numpy()
    W = W[:int(os.environ.get("PBR_ROWS", W.shape[0]))]
else:
    W = (0.02*rng.standard_t(6, size=(512, 896))/np.sqrt(1.5)).astype(np.float64)
W = W.astype(np.float64); O, I = W.shape
# synthetic activations: correlated, with a few outlier channels (as in real LLMs)
if len(sys.argv) > 3:      # real calibration activations saved as .npy, shape (tokens, in_features)
    A = np.load(sys.argv[3]).astype(np.float64)
    assert A.shape[1] == I, f"activation width {A.shape[1]} != weight in_features {I}"
else:
    A = rng.normal(size=(1024, I)) @ (np.eye(I) + 0.1*rng.normal(size=(I, I))/np.sqrt(I))
    A[:, rng.choice(I, 8, replace=False)] *= 12
H = A.T @ A / len(A)
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())
def wsqnr(Wq): return 10*np.log10((W**2).sum()/((W-Wq)**2).sum())
def ortho(n): q, r = np.linalg.qr(rng.normal(size=(n, n))); return q*np.sign(np.diag(r))
Ro, Ri = ortho(O), ortho(I)          # rotation: W_r = Ro W Ri ; inputs A_r = A Ri

def rtn_group(Wm, bits, g=64):        # round-to-nearest, per-group absmax scale
    L = 2**bits; Z = Wm.reshape(O, -1, g); s = np.abs(Z).max(-1, keepdims=True)/((L-1)/2)
    return (np.clip(np.round(Z/s-.5), -L/2, L/2-1)+.5).__mul__(s).reshape(O, I)

def gptq(Wm, Hm, bits, g=64, damp=.01):   # column-by-column with error feedback
    Wm = Wm.copy(); L = 2**bits; Q = np.zeros_like(Wm)
    Hm = Hm + damp*np.mean(np.diag(Hm))*np.eye(I)
    Hi = np.linalg.cholesky(np.linalg.inv(Hm)).T
    for j in range(I):
        if j % g == 0:
            s = np.abs(Wm[:, j:j+g]).max(1)/((L-1)/2)
        q = (np.clip(np.round(Wm[:, j]/s-.5), -L/2, L/2-1)+.5)*s
        Q[:, j] = q; e = (Wm[:, j]-q)/Hi[j, j]
        Wm[:, j+1:] -= np.outer(e, Hi[j, j+1:])
    return Q

# CHUNK caps the (rows, k) distance matrix built per batch, so a wide real
# tensor (e.g. L21 down_proj: 544,768 rows) doesn't try to allocate a single
# ~1GB+ float64 array and OOM (numpy.core._exceptions._ArrayMemoryError, see
# docs/pbr_ladder/README.md). Same fix, same chunk size, as quantize_full_model.py.
CHUNK = int(os.environ.get("PBR_CHUNK", 50000))
def assign(V, C):    # nearest-codeword index per row of V, chunked to bound peak memory
    return np.concatenate([((V[i:i+CHUNK]**2).sum(1)[:, None] - 2*V[i:i+CHUNK]@C.T
                             + (C**2).sum(1)[None]).argmin(1) for i in range(0, len(V), CHUNK)])
def kmeans(V, k, it=12):
    C = V[rng.choice(len(V), k, replace=False)]
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = V[m].mean(0)
    return C
def rvq(Wm, stages, d=8, k=256, passes=2):   # residual VQ: w = sum of one node-vector per stage
    s = np.sqrt((Wm**2).mean(1, keepdims=True)); V = (Wm/s).reshape(-1, d)
    books, idx = [], []; R = V.copy()
    for _ in range(stages):
        C = kmeans(R, k); a = assign(R, C)
        books.append(C); idx.append(a); R = R - C[a]
    for _ in range(passes):                      # refine each stage given the others
        for t in range(stages):
            R = R + books[t][idx[t]]
            idx[t] = assign(R, books[t])
            for j in range(k):
                m = idx[t] == j
                if m.any(): books[t][j] = R[m].mean(0)
            R = R - books[t][idx[t]]
    Vq = sum(b[i] for b, i in zip(books, idx))
    side = (stages*k*d*16 + O*16) / W.size      # codebooks + per-row scale
    return (Vq.reshape(O, I))*s, stages*np.log2(k)/d + side

rows = []
def add(name, bpw, Wq): rows.append((name, bpw, wsqnr(Wq), osqnr(Wq)))
unrot = lambda Wr: Ro.T @ Wr @ Ri.T
Wr, Hr = Ro @ W @ Ri, Ri.T @ H @ Ri
for bits in (3, 2):
    g = 64; ov = 16/g
    add(f"INT{bits} RTN", bits+ov, rtn_group(W, bits))
    add(f"INT{bits} GPTQ", bits+ov, gptq(W, H, bits))
    add(f"INT{bits} rot+GPTQ", bits+ov, unrot(gptq(Wr, Hr, bits)))
    Wq, b = rvq(Wr, stages=bits); add(f"{bits}x256 8-D nodes, rot", b, unrot(Wq))
print(f"{'method':<26}{'bpw':>7}{'weight dB':>11}{'output dB':>11}")
for n, b, w, o in rows: print(f"{n:<26}{b:7.3f}{w:11.2f}{o:11.2f}")
