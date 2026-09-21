# PBR-Ladder test: lossy node tiers + a BIT-EXACT BF16 top tier coded CONDITIONED on the lossy tier.
# Question: does "lossy tiers + exact tier" cost about the same as exact alone (PBR-E ~10.6)?
import numpy as np
src = open("push_nested.py").read().split("print(\"\\nSuccessive refinement")[0]
src = src.replace('print(f"{\'method\'', '#').split("for n in (3, 2, 1):")[0]
exec(src)
full = open("push_nested.py").read()
exec(full[full.index("def fb_pass"):full.index("print(\"\\nSuccessive")])
def to_bf16(x):
    u = x.astype(np.float32).view(np.uint32); return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.int64)
def ordinal(b): return np.where(b & 0x8000, -(b & 0x7FFF), b)       # monotone integer order of BF16
def H(x):
    _, c = np.unique(x, return_counts=True); p = c/c.sum(); return -(p*np.log2(p)).sum()
def Hcond(x, ctx):
    tot = 0.0
    for c in np.unique(ctx):
        m = ctx == c; tot += m.mean()*H(x[m])
    return tot
Wb = to_bf16(W); Wexact = ordinal(Wb)
sgn, ex, man = (Wb >> 15) & 1, (Wb >> 7) & 0xFF, Wb & 0x7F
pbrE = 1 + H(ex) + 7            # sign raw + exponent entropy + mantissa raw
print(f"exact alone (PBR-E style estimate): {pbrE:.2f} bpw   [word entropy {H(Wb):.2f}]")
Q1 = fb_pass(Wr, books[:1]); Q2 = Q1 + fb_pass(Wr - Q1, books_for(Wr - Q1, 1))
Q3 = Q2 + fb_pass(Wr - Q2, books_for(Wr - Q2, 1))
for n, Q, b in ((1, Q1, 1.089), (2, Q2, 2.179), (3, Q3, 3.268)):
    P = to_bf16(unrot(Q)); diff = Wexact - ordinal(P)
    ctx = (P >> 7) & 0xFF                                           # context: predicted exponent
    ex_tier = Hcond(diff, ctx)
    print(f"ladder up to tier {n}: lossy {b:.2f} + exact-given-lossy {ex_tier:.2f} = {b+ex_tier:.2f} bpw"
          f"  (overhead vs exact alone {b+ex_tier-pbrE:+.2f})")
