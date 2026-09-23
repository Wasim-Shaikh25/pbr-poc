# PATH B -- the user's actual idea, tested fairly for the first time.
# NO rotation (rotation whitens importance and kills this idea). Give each weight
# COLUMN the number of bits it actually needs {1,2,3,4,6,8} by its real output-error
# sensitivity, pack into one model, push the AVERAGE under 3 bpw, keep quality.
#
# Sensitivity uses the true input Hessian diag H_ii (GPTQ/OBD proxy): the output
# error of quantizing input-column i is H_ii * sum_o (W[o,i]-q)^2 (diagonal approx).
# Allocation = greedy marginal-gain water-filling over discrete bit levels to a budget.
# Compare: mixed(no-rot) vs flat scalar(no-rot) at matched bpw, and vs rotate+flat VQ.
#
# Usage: python mixed_precision_noR.py <weight_key> <acts.npy> [target_bpw]
import sys, numpy as np
from safetensors import safe_open
rng = np.random.default_rng(0)
CK = "./qwen05b/model.safetensors"
key, actf = sys.argv[1], sys.argv[2]
TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
BITS = [1, 2, 3, 4, 6, 8]
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64); O, I = W.shape
H = A.T @ A / len(A)
hd = np.clip(np.diag(H), 1e-12, None)                      # per-input-column Hessian weight
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())

def qcol(x, b):
    """uniform min-max b-bit quantize a column vector x (length O)."""
    lo, hi = x.min(), x.max()
    if hi <= lo: return x.copy()
    L = (1 << b) - 1
    s = (hi - lo) / L
    return lo + np.round((x - lo) / s) * s

# --- per-column distortion at each candidate bit-width (weighted output error) ---
# d[i, k] = H_ii * ||W[:,i] - q_b(W[:,i])||^2   for BITS[k]
Dmat = np.zeros((I, len(BITS)))
Qcache = {}                                                # (i,k) -> quantized column
for i in range(I):
    col = W[:, i]
    for k, b in enumerate(BITS):
        q = qcol(col, b); Qcache[(i, k)] = q
        Dmat[i, k] = hd[i] * ((col - q) ** 2).sum()

# --- greedy marginal-gain allocation to hit average TARGET bits/weight ---
# start everyone at min bits, then repeatedly spend +1 level where error drops most per bit.
lvl = np.zeros(I, dtype=int)                               # index into BITS, start at BITS[0]=1 bit
budget_bits = TARGET * I
def total_bits(l): return sum(BITS[k] for k in l)
# precompute marginal gain of stepping column i from level k to k+1: (err drop)/(extra bits)
while True:
    cur = total_bits(lvl)
    # candidate steps
    best_i, best_gain = -1, -1.0
    for i in range(I):
        k = lvl[i]
        if k + 1 >= len(BITS): continue
        extra = BITS[k+1] - BITS[k]
        if cur - BITS[k] + BITS[k+1] > budget_bits: continue   # would blow budget
        gain = (Dmat[i, k] - Dmat[i, k+1]) / extra
        if gain > best_gain: best_gain, best_i = gain, i
    if best_i < 0: break
    lvl[best_i] += 1

achieved_bpw = total_bits(lvl) / I
# build mixed-precision Wq
Wq_mix = np.empty_like(W)
for i in range(I):
    Wq_mix[:, i] = Qcache[(i, lvl[i])]
sq_mix = osqnr(Wq_mix)

# --- baseline 1: FLAT scalar (no rotation) at the same achieved bpw (nearest int bits) ---
bflat = int(round(achieved_bpw))
Wq_flat = np.empty_like(W)
for i in range(I):
    Wq_flat[:, i] = qcol(W[:, i], bflat)
sq_flat = osqnr(Wq_flat)

# --- baseline 2: rotate + flat VQ (our pipeline) at ~3 bpw, for reference ---
def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n))); return q*np.sign(np.diag(r))
def assign(V, C, ch=40000):
    return np.concatenate([((V[i:i+ch]**2).sum(1)[:,None]-2*V[i:i+ch]@C.T+(C**2).sum(1)[None]).argmin(1) for i in range(0,len(V),ch)])
def kmeans(V, k, it=12):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m=a==j
            if m.any(): C[j]=V[m].mean(0)
    return C
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri
s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
V = (Wr/s).reshape(-1, 8); q = np.zeros_like(V); r = V.copy()
for _ in range(3):                                          # 3 stages = 3 bpw
    C = kmeans(r, 256); a = assign(r, C); q += C[a]; r -= C[a]
Wq_vq = Ro.T @ (q.reshape(O, I)*s) @ Ri.T
sq_vq = osqnr(Wq_vq)

hist = {b: int((np.array([BITS[k] for k in lvl]) == b).sum()) for b in BITS}
print(f"tensor {key}  ({O}x{I})   target {TARGET} bpw")
print(f"bit histogram over {I} columns: " + "  ".join(f"{b}b:{hist[b]}" for b in BITS))
print(f"\n{'method':<34}{'bpw':>7}{'output SQNR dB':>16}")
print(f"{'MIXED precision, NO rotation (idea)':<34}{achieved_bpw:7.3f}{sq_mix:16.2f}")
print(f"{'FLAT scalar, NO rotation':<34}{float(bflat):7.3f}{sq_flat:16.2f}")
print(f"{'rotate + flat VQ (our pipeline)':<34}{3.0:7.3f}{sq_vq:16.2f}")
print(f"\nmixed - flat(no-rot) : {sq_mix - sq_flat:+.2f} dB   (does per-need allocation beat flat, un-rotated?)")
print(f"mixed - rotate+VQ    : {sq_mix - sq_vq:+.2f} dB   (does the idea beat OUR pipeline at ~3 bpw?)")
