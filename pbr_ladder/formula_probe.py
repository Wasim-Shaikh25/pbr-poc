# Can ŵ[i,j] = F_c(i,j,nodes) with c in 0..15, <=64 nodes, rebuild BF16 exactly?
import sys, numpy as np
def bf16_bits(x):  # float32 array -> uint16 bf16 (round-to-nearest-even)
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def bits_to_f(b): return (b.astype(np.uint32) << 16).view(np.float32)

if len(sys.argv) > 1:   # real tensor: python formula_probe.py model.safetensors model.layers.12.mlp.gate_proj.weight
    from safetensors import safe_open
    with safe_open(sys.argv[1], "pt") as f: W = f.get_tensor(sys.argv[2]).view(__import__("torch").int16).numpy().view(np.uint16)
else:                   # surrogate with Qwen-like stats
    rng = np.random.default_rng(0)
    W = bf16_bits(0.02 * rng.standard_t(6, size=(512, 896)) / np.sqrt(1.5))
H, Wd = W.shape; N = W.size; w = bits_to_f(W)

# 1) information bound: empirical entropy of the BF16 symbols
_, cnt = np.unique(W, return_counts=True); p = cnt / N
ent = -(p * np.log2(p)).sum()
print(f"n={N}  distinct BF16 values={len(cnt)}  entropy={ent:.2f} bits/weight")
top = np.sort(cnt)[::-1]
print(f"mass of 16 most common values={top[:16].sum()/N:.2%}, 64 most common={top[:64].sum()/N:.2%}")

# 2) formula search. Nodes = 64 floats. Family per code:
#    a, a+b, a*i, a*j, a*i+b*j, round-to-bf16 of each (i,j are in-tile coords).
#    Greedy: each step add the (formula, node assignment) that covers most uncovered weights.
T = 16
I = (np.arange(H) % T)[:, None].astype(np.float32) * np.ones((1, Wd), np.float32)
J = (np.arange(Wd) % T)[None, :].astype(np.float32) * np.ones((H, 1), np.float32)
nodes = bits_to_f(np.array(sorted(zip(cnt, np.unique(W)))[::-1][:64])[:, 1].astype(np.uint16))
covered = np.zeros(W.shape, bool); codes_used = []
cands = []
for a in range(64):
    cands.append((f"n{a}", lambda a=a: np.full(W.shape, nodes[a], np.float32)))
    cands.append((f"n{a}*i", lambda a=a: nodes[a] * I)); cands.append((f"n{a}*j", lambda a=a: nodes[a] * J))
    for b in range(0, 64, 8):
        cands.append((f"n{a}+n{b}", lambda a=a, b=b: np.full(W.shape, nodes[a] + nodes[b], np.float32)))
        cands.append((f"n{a}*i+n{b}*j", lambda a=a, b=b: nodes[a] * I + nodes[b] * J))
for c in range(16):
    best = (0, None, None)
    for name, fn in cands:
        hit = (bf16_bits(fn()) == W) & ~covered
        s = hit.sum()
        if s > best[0]: best = (s, name, hit)
    if best[1] is None: break
    covered |= best[2]; codes_used.append(best[1])
    print(f"code {c:2d}: {best[1]:<14} cumulative exact_cover={covered.mean():.3%}")

side = (64 * 16 + 4 * N) / N
print(f"\nexact_cover={covered.mean():.3%}  side_bpw={side:.3f}  (4-bit code map dominates)")
print(f"PASS" if covered.mean() >= 0.99 and side <= 6 else "FAIL")
