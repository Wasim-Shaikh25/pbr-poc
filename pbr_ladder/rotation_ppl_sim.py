#!/usr/bin/env python
"""
Phase 1 v2 — corrected fake-quant PERPLEXITY sim (no kernel, no training).

v1 was inconclusive and I say why in the open: (a) eval text was one paragraph
repeated 40x, so fp PPL collapsed to ~1.07 and every perturbation exploded;
(b) it ran on 0.5B, which is embedding-fragile and cannot hold sub-3-bit at all;
(c) it quantized ALL layers incl. sensitive ones. Fixed here:
  - REAL WikiText eval slice (pbr_ladder/pipeline_data/wiki_eval.txt) -> realistic PPL
  - Qwen2.5-1.5B (largest that fits ~6.4 GB free RAM in bf16; a valid low-bit testbed,
    unlike 0.5B). The true 3B is confirmed later via the llama.cpp path.
  - keep embeddings / lm_head / norms at full precision (realistic mixed precision);
    quantize only the transformer Linear weights.
  - proper NF (normal-float) codebook, the kind rotation is supposed to unlock.

Isolates ONE thing via drop-in weight replacement (orthogonal rotation, reconstructed
back to original basis, so no activation rotation / no kernel needed):
    W_hat = derotate( Q_codebook( rotate(W) ) )   vs   Q_codebook(W)
Lower PPL = better. If rot 2-bit clearly beats no-rot 2-bit and approaches 3-bit,
sub-3-bit is rescuable and Phase 2 (the online-Hadamard kernel) is justified.
"""
import numpy as np, scipy.linalg, torch, time, math, os
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GS = 64; HB = 128; BITS = [2, 3]; NTOK = 1536
EVAL = "C:/python_project/pbr-poc/pbr_ladder/pipeline_data/wiki_eval.txt"

def had(n): return scipy.linalg.hadamard(n).astype(np.float32)/np.sqrt(n)
def _blk(n):
    if n % HB == 0: return HB
    b=1
    while b*2<=n and n%(b*2)==0: b*=2
    return max(2,b)
def block_rht(M, axis, seed):
    rng=np.random.default_rng(seed); X=M if axis==1 else M.T; r,n=X.shape; B=_blk(n)
    s=rng.choice([-1.,1.],size=n).astype(np.float32); H=had(B)
    Y=((X*s).reshape(r,n//B,B) @ H.T).reshape(r,n)
    return Y if axis==1 else Y.T
def derotate(M, axis, seed):
    rng=np.random.default_rng(seed); X=M if axis==1 else M.T; r,n=X.shape; B=_blk(n)
    H=had(B); Y=(X.reshape(r,n//B,B) @ H).reshape(r,n)
    s=rng.choice([-1.,1.],size=n).astype(np.float32); Y=Y*s
    return Y if axis==1 else Y.T

def nf_levels(bits):
    k=1<<bits; lv=norm.ppf((np.arange(k)+0.5)/k).astype(np.float32); return lv/np.abs(lv).max()
def quant_nf(W, bits, gs=GS):
    o,i=W.shape; gs=gs if i%gs==0 else i; lv=nf_levels(bits)
    Wr=W.reshape(o,i//gs,gs); amax=np.abs(Wr).max(2,keepdims=True); amax[amax==0]=1
    idx=np.abs((Wr/amax)[...,None]-lv[None,None,None,:]).argmin(-1)
    return (lv[idx]*amax).reshape(o,i)

def recon(W, bits, rot):
    if rot:
        Wr=block_rht(block_rht(W,1,1),0,2)
        return derotate(derotate(quant_nf(Wr,bits),0,2),1,1)
    return quant_nf(W,bits)

print("loading", MODEL, "(downloads ~3GB first run)"); t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16); model.eval(); model.requires_grad_(False)
print(f"loaded {time.time()-t0:.1f}s")

txt=open(EVAL, encoding="utf-8", errors="ignore").read()
ids=tok(txt, return_tensors="pt").input_ids[:, :NTOK]
print(f"eval tokens: {ids.shape[1]}")

orig={n:p.detach().clone() for n,p in model.named_parameters()
      if n.endswith(".weight") and p.dim()==2 and "layers." in n and "norm" not in n}
names=list(orig)
print(f"quantizing {len(names)} transformer Linear weights (embed/lm_head/norms kept full)\n")

@torch.no_grad()
def ppl():
    return math.exp(model(ids, labels=ids).loss.item())
@torch.no_grad()
def swap(bits, rot):
    sd=dict(model.named_parameters())
    for n in names:
        W=orig[n].float().numpy()
        sd[n].copy_(torch.from_numpy(recon(W,bits,rot)).to(sd[n].dtype))
@torch.no_grad()
def restore():
    sd=dict(model.named_parameters())
    for n in names: sd[n].copy_(orig[n])

restore(); base=ppl(); print(f"fp(bf16) baseline PPL = {base:.3f}\n")
print(f"{'bits':>4} {'rotation':>9} {'PPL':>10} {'vs fp':>8}"); print("-"*36)
R={}
for b in BITS:
    for rot in (False, True):
        swap(b,rot); p=ppl(); restore(); R[(b,rot)]=p
        print(f"{b:>4} {str(rot):>9} {p:>10.3f} {p/base:>7.2f}x")
print("\n=== rotation effect on model quality (PPL, lower=better) ===")
for b in BITS:
    pf,pt=R[(b,False)],R[(b,True)]; d=100*(pf-pt)/pf
    verdict="HELPS" if d>3 else "neutral" if d>-3 else "HURTS"
    print(f"  {b}-bit: no-rot {pf:8.3f} -> rot {pt:8.3f}  ({d:+.1f}% PPL)  [{verdict}]")
print(f"\nGate: if 2-bit rot PPL << 2-bit no-rot AND approaches 3-bit/fp, sub-3-bit is")
print(f"rescuable -> Phase 2 (online-Hadamard kernel on real 3B) justified. Baseline fp={base:.2f}")
