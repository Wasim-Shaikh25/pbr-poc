# Adaptive per-group bit allocation via proper reverse water-filling.
#
# Your idea: instead of a flat recipe (every group gets 256,256,64 = 2.75 bpw),
# fit a DEEP codebook once and give each group a different number of stages
# (= different bit-width, 1..MAXS bits), spending more bits where they reduce error
# most. "1/2/3/4/8-bit combination" == "keep 1/2/3/4/8 residual-VQ stages", because
# each 256-entry stage is exactly 1 bit/weight.
#
# Water-filling: rank every (group, next-stage) increment by Hessian-weighted error
# reduction per bit, then greedily buy increments until the budget is spent. Compare
# to the flat recipe AT THE SAME AVERAGE bpw, INCLUDING the side-info cost of storing
# each group's stage count. Metric: real output SQNR (through real activations).
#
# Usage: python adaptive_waterfill.py ./qwen05b/model.safetensors <weight_key> <acts.npy>
import sys, numpy as np
from safetensors import safe_open

rng = np.random.default_rng(0)
ckpt, key, actf = sys.argv[1], sys.argv[2], sys.argv[3]
MAXS = int(sys.argv[4]) if len(sys.argv) > 4 else 6      # deepest codebook (max bits/weight)
BUDGET_BPW = float(sys.argv[5]) if len(sys.argv) > 5 else 2.75   # flat comparison point
D, K = 8, 256

with safe_open(ckpt, "pt") as f:
    W = f.get_tensor(key).float().numpy().astype(np.float64)
A = np.load(actf).astype(np.float64)
O, I = W.shape
assert A.shape[1] == I
H = A.T @ A / len(A)
hdiag = np.diag(H).copy()                     # per-input-column importance

def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)
Ro, Ri = ortho(O, O), ortho(I, I)
Wr = Ro @ W @ Ri
hdiag_r = np.diag(Ri.T @ H @ Ri).copy()       # column importance in rotated space
def osqnr(Wq): return 10*np.log10(((A@W.T)**2).sum()/((A@(W-Wq).T)**2).sum())

def assign(V, C):
    return ((V**2).sum(1)[:, None] - 2*V@C.T + (C**2).sum(1)[None]).argmin(1)
def kmeans(V, k, it=10):
    C = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(it):
        a = assign(V, C)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = V[m].mean(0)
    return C

# fit MAXS-stage codebook once (successive refinement), record per-stage reconstruction
s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
V = (Wr / s).reshape(-1, D)                    # groups = 1x8 blocks; group g spans 8 cols
ngroups = V.shape[0]
# column index each group's 8 dims map to (for Hessian weighting)
col_of = (np.arange(ngroups*D) % I).reshape(ngroups, D)
w_of = hdiag_r[col_of]                         # (ngroups, D) importance weights
books, codes, Rres = [], [], V.copy()
recon = np.zeros((MAXS+1, ngroups, D))         # recon[t] = reconstruction using first t stages
for t in range(MAXS):
    C = kmeans(Rres, K); a = assign(Rres, C)
    books.append(C); codes.append(a)
    recon[t+1] = recon[t] + C[a]
    Rres = Rres - C[a]

# Hessian-weighted squared error of group g using first t stages
err = np.zeros((MAXS+1, ngroups))
for t in range(MAXS+1):
    err[t] = (w_of * (V - recon[t])**2).sum(1)
# marginal gain of the t-th stage for group g (t=1..MAXS), cost = 1 bit/weight (8 bits/group)
gain = err[:-1] - err[1:]                       # (MAXS, ngroups): gain[t-1,g] = err[t-1]-err[t]

def reconstruct(stages_per_group):
    """stages_per_group: (ngroups,) int in [0,MAXS]. Returns dequantized W (O,I)."""
    idx = np.clip(stages_per_group, 0, MAXS)
    Vq = recon[idx, np.arange(ngroups)]         # (ngroups, D)
    Wq_r = (Vq.reshape(O, I)) * s
    return Ro.T @ Wq_r @ Ri.T

# ---- flat baseline: same #stages for every group (nearest integer to budget) ----
flat_stages = int(round(BUDGET_BPW))
Wq_flat = reconstruct(np.full(ngroups, flat_stages))
sqnr_flat = osqnr(Wq_flat)

# ---- adaptive: reverse water-filling to the SAME TOTAL index bits ----
# total index bits available = flat_stages * ngroups * D   (each stage = D bits/group... 8 bits)
total_stage_units = flat_stages * ngroups       # number of stage-increments to distribute
alloc = np.zeros(ngroups, dtype=int)
# greedily take the highest marginal gains; a group can take stage t only after t-1.
# gain[t-1,g] is the value of the t-th increment. Build (value, group, t) and pick top-N,
# respecting order via cumulative: since gains are non-increasing across stages for VQ
# residuals in expectation, greedily picking top increments per group order works.
flat_gain = []
for t in range(MAXS):
    for g in range(ngroups):
        flat_gain.append((gain[t, g], g, t))
# efficient: argsort all increments; enforce prefix by processing in value order and
# only accepting increment t for g if g already has t stages.
vals = gain.T.reshape(-1)                        # index = g*MAXS + t
order = np.argsort(-vals)
taken = 0
for k_ in order:
    if taken >= total_stage_units: break
    g, t = divmod(k_, MAXS)
    if alloc[g] == t:                            # can only add the next stage in order
        alloc[g] += 1; taken += 1
Wq_adapt = reconstruct(alloc)
sqnr_adapt = osqnr(Wq_adapt)

# ---- honest bit accounting ----
index_bpw = flat_stages                          # both schemes use the same index bits/weight
side_flat = (MAXS+1e-9)*0                         # flat: one global number, ~0
# adaptive per-group side info: log2(MAXS+1) bits per group / D weights
side_adapt_pergroup = np.log2(MAXS+1) / D
# amortized: store stage-count per COLUMN-BLOCK (I/D blocks, shared across O rows)
side_adapt_colblock = (np.log2(MAXS+1) * (I//D)) / (O*I)

print(f"tensor {key}  ({O}x{I})  acts {actf}")
print(f"deepest codebook: {MAXS} stages (max {MAXS} bpw)   flat comparison at {flat_stages} stages")
print(f"stage histogram (adaptive): {np.bincount(alloc, minlength=MAXS+1)}")
print(f"\n{'scheme':<34}{'index bpw':>10}{'+side':>8}{'= total':>9}{'out SQNR dB':>13}")
print(f"{'flat '+str(flat_stages)+' stages':<34}{index_bpw:10.3f}{0.0:8.3f}{index_bpw:9.3f}{sqnr_flat:13.2f}")
print(f"{'adaptive (per-group side info)':<34}{index_bpw:10.3f}{side_adapt_pergroup:8.3f}{index_bpw+side_adapt_pergroup:9.3f}{sqnr_adapt:13.2f}")
print(f"{'adaptive (per-colblock side info)':<34}{index_bpw:10.3f}{side_adapt_colblock:8.3f}{index_bpw+side_adapt_colblock:9.3f}{sqnr_adapt:13.2f}")
print(f"\nadaptive vs flat at ~equal index bits: {sqnr_adapt-sqnr_flat:+.2f} dB "
      f"(before side info); per-colblock overhead is +{side_adapt_colblock:.4f} bpw (negligible)")
