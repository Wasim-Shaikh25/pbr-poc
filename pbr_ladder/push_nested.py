# Combined: rotation + 8-D residual node codebooks + GPTQ-style block error feedback,
# and NESTED tiers: one encoding, truncate to 1/2/3 stages -> ~1/2/3 bpw.
import sys, numpy as np
exec(open("push_below4.py").read().split("rows = []")[0])   # reuse data, H, rotation, helpers
d, k, S = 8, 256, 3
Wr, Hr = Ro @ W @ Ri, Ri.T @ H @ Ri
s = np.sqrt((Wr**2).mean(1, keepdims=True))
def train_books(Wm):
    V = (Wm/s).reshape(-1, d); R = V.copy(); books = []
    for _ in range(S):
        C = kmeans(R, k); a = ((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1)
        books.append(C); R = R - C[a]
    return books
def assign(V, books, n):
    R = V.copy(); out = np.zeros_like(V)
    for t in range(n):
        C = books[t]; a = ((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1)
        out += C[a]; R -= C[a]
    return out
def encode(Wm, books, n, feedback):
    Wm = Wm.copy(); Q = np.zeros_like(Wm)
    U = np.linalg.cholesky(np.linalg.inv(Hr + .01*np.mean(np.diag(Hr))*np.eye(I))).T
    for j in range(0, I, d):
        b = slice(j, j+d)
        Q[:, b] = assign(Wm[:, b]/s, books, n)*s
        if feedback and j+d < I:
            E = Wm[:, b]-Q[:, b]
            Wm[:, j+d:] -= E @ np.linalg.solve(U[b, b], U[b, j+d:])
    return Q
unrot = lambda M: Ro.T @ M @ Ri.T
books = train_books(Wr)
side = lambda n: (n*k*d*16 + O*16)/W.size
print(f"{'method':<34}{'bpw':>7}{'output dB':>11}")
for n in (3, 2, 1):
    for fb in (False, True):
        Wq = unrot(encode(Wr, books, n, fb))
        print(f"{f'{n}x256 nodes rot'+(' +feedback' if fb else ''):<34}{n+side(n):7.3f}{osqnr(Wq):11.2f}")
# nesting test: dedicated 2-stage books vs truncated from the 3-stage set (same here by
# construction: greedy RVQ is nested). Report the feedback-optimized per-tier result above.
for bits in (3, 2):
    print(f"{f'INT{bits} rot+GPTQ (ref)':<34}{bits+.25:7.3f}{osqnr(unrot(gptq(Wr,Hr,bits))):11.2f}")
print(f"{'INT4 RTN (ref)':<34}{4.25:7.3f}{osqnr(rtn_group(W,4)):11.2f}")
# TRUE nesting: encode once at 3 stages with feedback, then drop stages (truncate codes).
def encode_parts(Wm):
    Wm = Wm.copy(); parts = np.zeros((S,)+Wm.shape)
    U = np.linalg.cholesky(np.linalg.inv(Hr + .01*np.mean(np.diag(Hr))*np.eye(I))).T
    for j in range(0, I, d):
        b = slice(j, j+d); R = Wm[:, b]/s
        for t in range(S):
            C = books[t]; a = ((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1)
            parts[t][:, b] = C[a]*s; R = R - C[a]
        if j+d < I:
            E = Wm[:, b]-parts[:, :, b].sum(0)
            Wm[:, j+d:] -= E @ np.linalg.solve(U[b, b], U[b, j+d:])
    return parts
P = encode_parts(Wr)
print("\nONE file, truncated tiers (codes from single 3-stage feedback encode):")
for n in (3, 2, 1):
    print(f"  tier {n} stages  bpw {n+side(n):.3f}  output dB {osqnr(unrot(P[:n].sum(0))):.2f}")
# Tier-aware feedback: propagate a blend of tier-2 and tier-3 errors so every prefix stays good.
def encode_tiered(Wm, alpha):
    Wm = Wm.copy(); parts = np.zeros((S,)+Wm.shape)
    U = np.linalg.cholesky(np.linalg.inv(Hr + .01*np.mean(np.diag(Hr))*np.eye(I))).T
    for j in range(0, I, d):
        b = slice(j, j+d); R = Wm[:, b]/s
        for t in range(S):
            C = books[t]; a = ((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1)
            parts[t][:, b] = C[a]*s; R = R - C[a]
        if j+d < I:
            E2 = Wm[:, b]-parts[:2, :, b].sum(0); E3 = Wm[:, b]-parts[:, :, b].sum(0)
            Wm[:, j+d:] -= (alpha*E2+(1-alpha)*E3) @ np.linalg.solve(U[b, b], U[b, j+d:])
    return parts
print("\nTier-aware feedback (alpha = weight on tier-2 error):")
for alpha in (1.0, 0.7, 0.5):
    P = encode_tiered(Wr, alpha)
    print(f"  alpha {alpha}: " + "  ".join(f"T{n} {n+side(n):.2f}bpw {osqnr(unrot(P[:n].sum(0))):5.2f}dB" for n in (3,2,1)))
# Successive refinement: each tier gets its OWN feedback pass on what the previous tiers left.
def fb_pass(T, bk):             # encode target T with codebook list bk + feedback
    T = T.copy(); Q = np.zeros_like(T)
    U = np.linalg.cholesky(np.linalg.inv(Hr + .01*np.mean(np.diag(Hr))*np.eye(I))).T
    sc = np.sqrt((T**2).mean(1, keepdims=True))
    for j in range(0, I, d):
        b = slice(j, j+d); R = T[:, b]/sc; q = np.zeros_like(R)
        for C in bk:
            a = ((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1); q += C[a]; R -= C[a]
        Q[:, b] = q*sc
        if j+d < I: T[:, j+d:] -= (T[:, b]-Q[:, b]) @ np.linalg.solve(U[b, b], U[b, j+d:])
    return Q
def books_for(T, n):
    sc = np.sqrt((T**2).mean(1, keepdims=True)); R = (T/sc).reshape(-1, d); out = []
    for _ in range(n):
        C = kmeans(R, k); a = ((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1); out.append(C); R = R - C[a]
    return out
print("\nSuccessive refinement (one file, every tier feedback-optimized):")
Q1 = fb_pass(Wr, books[:1]); Q2 = Q1 + fb_pass(Wr - Q1, books_for(Wr - Q1, 1))
Q3 = Q2 + fb_pass(Wr - Q2, books_for(Wr - Q2, 1))
side2 = lambda n: side(n) + (n-1)*16/W.size*O   # extra per-row scale per refinement tier
for n, Q in ((1, Q1), (2, Q2), (3, Q3)):
    print(f"  tier {n}: {n+side2(n):.3f} bpw  output {osqnr(unrot(Q)):.2f} dB")
