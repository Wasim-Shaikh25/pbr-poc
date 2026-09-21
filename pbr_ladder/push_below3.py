# Fill the 2-3 bpw range: 8-D feedback VQ with mixed stage sizes, + entropy of codes.
import numpy as np
exec(open("push_below4.py").read().split("rows = []")[0])
d = 8
Wr, Hr = Ro @ W @ Ri, Ri.T @ H @ Ri
unrot = lambda M: Ro.T @ M @ Ri.T
s = np.sqrt((Wr**2).mean(1, keepdims=True))
U = np.linalg.cholesky(np.linalg.inv(Hr + .01*np.mean(np.diag(Hr))*np.eye(I))).T
def train(ks):
    R = (Wr/s).reshape(-1, d); out = []
    for k in ks:
        C = kmeans(R, k); a = assign(R, C)   # chunked; same indices as a full distance matrix
        out.append(C); R = R - C[a]
    return out
def enc(books):
    T = Wr.copy(); Q = np.zeros_like(T); codes = [[] for _ in books]
    for j in range(0, I, d):
        b = slice(j, j+d); R = T[:, b]/s; q = np.zeros_like(R)
        for t, C in enumerate(books):
            a = ((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1); q += C[a]; R -= C[a]; codes[t].append(a)
        Q[:, b] = q*s
        if j+d < I: T[:, j+d:] -= (T[:, b]-Q[:, b]) @ np.linalg.solve(U[b, b], U[b, j+d:])
    return Q, [np.concatenate(c) for c in codes]
def H(x): p = np.bincount(x)/len(x); p = p[p>0]; return -(p*np.log2(p)).sum()
print(f"{'stages (codebook sizes)':<24}{'bpw':>7}{'bpw+entropy':>13}{'output dB':>11}")
for ks in ([256,256], [256,256,4], [256,256,16], [256,256,64], [256,256,256]):
    books = train(ks); Q, codes = enc(books)
    side = (sum(ks)*d*16 + O*16)/W.size
    raw = sum(np.log2(k) for k in ks)/d + side
    ent = sum(H(c) for c in codes)/d + side
    print(f"{str(ks):<24}{raw:7.3f}{ent:13.3f}{osqnr(unrot(Q)):11.2f}")
print(f"{'INT4 RTN (ref)':<24}{4.25:7.3f}{'':>13}{osqnr(rtn_group(W,4)):11.2f}")
# Optional: save the dequantized weight for eval_ppl.py
#   PBR_SAVE=out.npz PBR_NAME=model.layers.12.mlp.gate_proj.weight PBR_SAVE_KS=256,256,64 python push_below3.py ...
import os
if os.environ.get("PBR_SAVE"):
    ks = [int(k) for k in os.environ.get("PBR_SAVE_KS", "256,256,64").split(",")]
    Q, _ = enc(train(ks))
    np.savez(os.environ["PBR_SAVE"], **{os.environ["PBR_NAME"]: unrot(Q).astype(np.float32)})
    print("saved", os.environ["PBR_SAVE"], "config", ks, "shape", W.shape)
