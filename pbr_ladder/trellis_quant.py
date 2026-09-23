# trellis_quant.py -- QTIP-style bitshift-trellis (Viterbi) weight quantizer, CPU numpy.
#
# WHY: after the incoherence rotation the rotated weights are ~i.i.d. Gaussian, so the ONLY
# gain a quantizer can have over good scalar is the space-filling gain (<=1.53 dB, ~0.65 dB
# at D=8). A memoryless D=8 codebook (our RVQ) leaves most of that on the table. A trellis
# raises the EFFECTIVE quantization dimension to the whole row (Viterbi-decoded jointly)
# while keeping a trivial decoder -- this is the mechanism that should recover the rest.
#
# Structure (QTIP bitshift trellis): 2^L states. From state i, appending k bits b gives
# next state j = ((i<<k)|b) mod 2^L. The emitted value depends only on the NEW state via a
# fixed hashed Gaussian codebook (zero stored codebook). Because emit depends only on j, the
# Viterbi step reduces to:  new_cost[j] = emit_cost[j] + min_{a} old_cost[a*2^(L-k)+ (j>>k)].
# We fix the initial register to 0, so total bits = sum_t k_t exactly (bpw = mean(k) + row-scale).
#
# Baselines reused from push_below4.py / beam_vq.py: INT3+GPTQ scalar, 3-stage D8/K256 RVQ.
#
# Usage:
#   python trellis_quant.py <weight_key> <acts.npy> [--L 10] [--hess] [--rowchunk N]
# e.g.
#   python trellis_quant.py model.layers.12.mlp.gate_proj.weight acts_L12_gate.npy --L 10
#
# HARD MEMORY RULE: every (N x K) distance/cost matrix is chunked to <= CHUNK rows (RVQ/kmeans),
# and the trellis backpointer store is bounded by chunking the OUTPUT ROWS (--rowchunk / auto).
import sys, os, time, numpy as np

CK = "./qwen05b/model.safetensors"
CHUNK = 40000                      # cap on (rows x K) matrices, per the OOM rule (544768x256 f64 OOMs)
rng = np.random.default_rng(0)

# ---------------- args ----------------
key  = sys.argv[1]
actf = sys.argv[2]
def flag(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        return sys.argv[i+1] if (i+1 < len(sys.argv) and not sys.argv[i+1].startswith("--")) else True
    return default
L         = int(flag("--L", 10))            # log2(#states); try 8,10,12
USE_HESS  = bool(flag("--hess", False))     # fold diag of rotated Hessian into per-column cost (default OFF)
FEEDBACK  = bool(flag("--feedback", False)) # trellis + GPTQ/LDLQ column error feedback (the QTIP+LDLQ combo)
TILE      = int(flag("--tile", 16))         # Viterbi tile width for feedback mode (small=more feedback)
ROWCHUNK  = flag("--rowchunk", None)
S = 1 << L

# ---------------- load + rotate (same conventions as beam_vq.py / push_below4.py) --------------
from safetensors import safe_open
with safe_open(CK, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64)
O, I = W.shape
assert A.shape[1] == I, f"activation width {A.shape[1]} != weight in_features {I}"

def ortho(n, seed):                          # exact pattern from beam_vq.py (seeded, sign-fixed QR)
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q * np.sign(np.diag(r))
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri                              # rotated weights (quantize here)
H  = A.T @ A / len(A)                          # input Hessian (original space)
Hr = Ri.T @ H @ Ri                             # rotated Hessian (for GPTQ + optional trellis weighting)

def osqnr(Wq):  return 10*np.log10(((A@W.T)**2).sum() / ((A@(W-Wq).T)**2).sum())
def wsqnr(Wq):  return 10*np.log10((W**2).sum() / ((W-Wq)**2).sum())
unrot = lambda Wrq: Ro.T @ Wrq @ Ri.T          # rotated-space reconstruction -> original space

# ---------------- inverse normal CDF (Acklam), dependency-free Phi^-1 for the Gaussian codebook ----
def norm_ppf(p):
    p = np.asarray(p, dtype=np.float64)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.zeros_like(p)
    lo = p < plow; hi = p > phigh; mid = ~(lo | hi)
    q = np.sqrt(-2*np.log(p[lo]))
    x[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p[mid]-0.5; r = q*q
    x[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2*np.log(1-p[hi]))
    x[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return x

# Gaussian codebook over the 2^L states: the MARGINAL set of values is exactly Phi^-1 quantile
# bins; a bijective multiplicative hash assigns them to states so trellis-adjacency != value-
# adjacency (QTIP's decorrelation trick, "compute-based Gaussian code", nothing stored).
base = norm_ppf((np.arange(S) + 0.5) / S)             # sorted standard-normal quantile bins
perm = (np.arange(S) * 2654435761) & (S - 1)          # odd multiplier -> bijection mod 2^L
gvals = base[perm].astype(np.float64)                 # gvals[state] -> reproduction value

# ---------------- the trellis / Viterbi quantizer ----------------
def kschedule(bpw_target, T):
    """integer bits-per-step sequence of length T averaging bpw_target (supports x.0 and x.5)."""
    k0 = int(np.floor(bpw_target)); frac = bpw_target - k0
    ks = np.full(T, k0, dtype=np.int64)
    nhi = int(round(frac * T))
    ks[np.linspace(0, T-1, nhi).round().astype(int)] += 1     # spread the +1 steps evenly
    return ks

def trellis_row_block(X, w, ks, init_state=None):
    """Viterbi-quantize a block of rows. X:(m,T) normalized rows, w:(T,) col weights, ks:(T,) bits/step.
       init_state: (m,) per-row starting register (delta cost); None -> fixed register 0.
       Returns (decoded states (m,T) int, final register (m,) int)."""
    m, T = X.shape
    INF = 1e30
    cost = np.full((m, S), INF)
    if init_state is None: cost[:, 0] = 0.0                  # fixed initial state 0 (first tile)
    else: cost[np.arange(m), init_state] = 0.0              # continue trellis from previous tile
    bptr = []                                                 # per-step a-choice (m, Lk_t), int8
    sk_list, Lk_list = [], []
    for t in range(T):
        k = int(ks[t]); sk = 1 << k; Lk = S >> k
        emit = w[t] * (X[:, t:t+1] - gvals[None, :])**2       # (m, S)  cost of landing in each state
        pr = cost.reshape(m, sk, Lk)                          # predecessors grouped by (a, p)
        mp = pr.min(axis=1); ac = pr.argmin(axis=1).astype(np.int8)   # (m, Lk)
        minpred = np.repeat(mp, sk, axis=1)                   # (m, S): minpred[:,j] = mp[:, j>>k]
        cost = minpred + emit
        bptr.append(ac); sk_list.append(sk); Lk_list.append(Lk)
    state = cost.argmin(axis=1)                               # best final state per row
    final = state.copy()
    dec = np.empty((m, T), dtype=np.int64)
    ar = np.arange(m)
    for t in range(T-1, -1, -1):
        dec[:, t] = state
        sk, Lk = sk_list[t], Lk_list[t]
        p = state // sk                                       # low (L-k) bits carried to predecessor
        a = bptr[t][ar, p].astype(np.int64)
        state = a * Lk + p                                    # predecessor state
    return dec, final

def trellis_quant(Wr_, bpw_target, use_hess, rowchunk):
    s = np.sqrt((Wr_**2).mean(1, keepdims=True)) + 1e-12      # per-row scale (matches beam_vq.py)
    Xn = Wr_ / s
    T = I
    w = np.ones(T)
    if use_hess:                                              # diag(Hr) importance, mean-normalized
        w = np.clip(np.diag(Hr), 1e-12, None); w = w / w.mean()
    ks = kschedule(bpw_target, T)
    if rowchunk is None:
        Lkmax = S >> int(ks.min())                            # widest backpointer row (int8)
        rowchunk = max(1, min(O, CHUNK, int(6e8 / (T * Lkmax))))
    else:
        rowchunk = int(rowchunk)
    dec = np.empty((O, T), dtype=np.int64)
    for i in range(0, O, rowchunk):
        dec[i:i+rowchunk], _ = trellis_row_block(Xn[i:i+rowchunk], w, ks)
    Wrq = gvals[dec] * s
    bpw = ks.mean() + 16.0 / I                                # code bits + per-row scale (codebook: 0, computed)
    return Wrq, bpw

# ---------------- trellis + GPTQ/LDLQ error feedback (the QTIP+LDLQ combination) ----------
# Standalone Viterbi minimizes WEIGHT MSE (wrong objective: its error is isotropic and lands partly
# in the high-activation directions the model feels). GPTQ's win over it is purely the Hessian-aware
# error feedback that shapes error OUT of those directions. But PURE greedy per-column feedback
# destroys the trellis: with only 2^k reachable (hashed, random) values per step and no lookahead,
# it degenerates to a ~2^k-level random-codebook scalar quantizer (measured: weight SQNR ~7-9 dB).
# The trellis needs the Viterbi lookahead. So we TILE: run full Viterbi inside short column-tiles
# (keeps the trellis geometry), and apply the exact GPTQ update across tiles (adds Hessian shaping).
# tile -> I is plain trellis (no feedback); tile=1 is the failed greedy; a small tile (8-32) gets both.
# Rows are independent given the shared Hi, so all O rows march the tiles in parallel. Peak transient
# is the O x (I-c1) inter-tile outer-product update -- no giant N x K matrix, no chunking needed.
def trellis_feedback(Wr_, bpw_target, tile=16, damp=.01):
    Om, Im = Wr_.shape
    s = np.sqrt((Wr_**2).mean(1, keepdims=True)) + 1e-12
    Xn = (Wr_ / s).copy()                                  # normalized; per-row scale is a per-row
    Hm = Hr + damp*np.mean(np.diag(Hr))*np.eye(Im)         # constant -> doesn't change per-row argmin
    Hi = np.linalg.cholesky(np.linalg.inv(Hm)).T           # upper-triangular (same machinery as gptq)
    ks = kschedule(bpw_target, Im); w1 = np.ones(Im)
    Qn = np.empty_like(Xn); state = np.zeros(Om, dtype=np.int64)
    for c0 in range(0, Im, tile):
        c1 = min(Im, c0 + tile)
        dec, state = trellis_row_block(Xn[:, c0:c1], w1[c0:c1], ks[c0:c1], init_state=state)
        Qn[:, c0:c1] = gvals[dec]                           # Viterbi within the tile
        for j in range(c0, c1):                             # exact GPTQ feedback to columns AFTER the tile
            e = (Xn[:, j] - Qn[:, j]) / Hi[j, j]
            if c1 < Im: Xn[:, c1:] -= np.outer(e, Hi[j, c1:])
    return Qn * s, ks.mean() + 16.0/Im

# ---------------- baselines: INT3+GPTQ scalar, D8/K256 RVQ (reused from push_below4.py) ----------
def gptq(Wm, Hm, bits, g=64, damp=.01):
    Wm = Wm.copy(); Lv = 2**bits; Q = np.zeros_like(Wm); n = Wm.shape[1]
    Hm = Hm + damp*np.mean(np.diag(Hm))*np.eye(n)
    Hi = np.linalg.cholesky(np.linalg.inv(Hm)).T
    s = None
    for j in range(n):
        if j % g == 0: s = np.abs(Wm[:, j:j+g]).max(1)/((Lv-1)/2)
        q = (np.clip(np.round(Wm[:, j]/s-.5), -Lv/2, Lv/2-1)+.5)*s
        Q[:, j] = q; e = (Wm[:, j]-q)/Hi[j, j]
        Wm[:, j+1:] -= np.outer(e, Hi[j, j+1:])
    return Q

def _assign(V, C):
    return np.concatenate([((V[i:i+CHUNK]**2).sum(1)[:, None] - 2*V[i:i+CHUNK]@C.T
                            + (C**2).sum(1)[None]).argmin(1) for i in range(0, len(V), CHUNK)])
def _kmeans(V, k, it=12):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = _assign(V, C)
        for j in range(k):
            msk = a == j
            if msk.any(): C[j] = V[msk].mean(0)
    return C
def rvq(Wm, stages, k=256, d=8, passes=2):
    s = np.sqrt((Wm**2).mean(1, keepdims=True)) + 1e-12; V = (Wm/s).reshape(-1, d)
    books, idx = [], []; R = V.copy()
    for _ in range(stages):
        C = _kmeans(R, k); a = _assign(R, C); books.append(C); idx.append(a); R = R - C[a]
    for _ in range(passes):
        for t in range(stages):
            R = R + books[t][idx[t]]; idx[t] = _assign(R, books[t])
            for j in range(k):
                msk = idx[t] == j
                if msk.any(): books[t][j] = R[msk].mean(0)
            R = R - books[t][idx[t]]
    Vq = sum(b[i] for b, i in zip(books, idx))
    side = (stages*k*d*16 + O*16) / W.size
    return (Vq.reshape(O, I))*s, stages*np.log2(k)/d + side

# ---------------- run ----------------
print(f"tensor {key}  W{W.shape}  acts {A.shape}   L={L} (states={S})  hess-weight={USE_HESS}  feedback={FEEDBACK}")
rows = []
def add(name, bpw, Wq): rows.append((name, bpw, wsqnr(Wq), osqnr(Wq)))

t0 = time.time()
# scalar reference (INT3 ~3.0-bpw floor; INT4 brackets the 3.5 target from above)
add("INT3 rot+GPTQ",  3 + 16/64, unrot(gptq(Wr, Hr, 3, g=64)))
add("INT4 rot+GPTQ",  4 + 16/64, unrot(gptq(Wr, Hr, 4, g=64)))
# --- 3.0 bpw bucket ---
Wq, b = rvq(Wr, stages=3, k=256);        add("RVQ 3x256 (D8)",   b, unrot(Wq))
Wq, b = trellis_quant(Wr, 3.0, USE_HESS, ROWCHUNK); add(f"TRELLIS L{L} @3.0", b, unrot(Wq))
if FEEDBACK:
    Wq, b = trellis_feedback(Wr, 3.0, tile=TILE); add(f"TRELLIS-FB L{L} t{TILE} @3.0", b, unrot(Wq))
# --- 3.5 bpw bucket ---
Wq, b = rvq(Wr, stages=4, k=128);        add("RVQ 4x128 (D8)",   b, unrot(Wq))
Wq, b = trellis_quant(Wr, 3.5, USE_HESS, ROWCHUNK); add(f"TRELLIS L{L} @3.5", b, unrot(Wq))
if FEEDBACK:
    Wq, b = trellis_feedback(Wr, 3.5, tile=TILE); add(f"TRELLIS-FB L{L} t{TILE} @3.5", b, unrot(Wq))

print(f"{'method':<22}{'bpw':>8}{'weight dB':>11}{'output dB':>11}")
for n, b, wq, oq in sorted(rows, key=lambda r: r[1]):
    print(f"{n:<22}{b:8.3f}{wq:11.2f}{oq:11.2f}")
print(f"[{time.time()-t0:.1f}s]  output SQNR = 10log10(||A W^T||^2 / ||A (W-Wq)^T||^2)")
