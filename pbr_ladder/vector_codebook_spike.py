#!/usr/bin/env python
"""
Phase 2 prototype — VECTOR codebook (VPTQ-style) vs scalar, +/- rotation, at ~3 and ~2.25 bpw.

The literature is explicit: at low bits the big win is a VECTOR/lattice codebook, not a
scalar one. Our earlier sims only used scalar NF -> that's why rotation looked modest.
This tests the real lever: group weights into short vectors, learn a k-means codebook,
quantize to codebook indices. With a randomized-Hadamard rotation first, the vectors are
Gaussian-incoherent, so a small codebook covers them tightly.

Metric = weight reconstruction rel-error ||W - W_hat||_F / ||W||_F at MATCHED bpw.
Methods per tensor, per bpw tier:
    scalar-NF            (per-group normal-float, our current baseline family)
    scalar-NF + rot
    VQ                   (vector codebook)
    VQ + rot             <- the beat-IQ3 candidate
Lower error at ~3 bpw for VQ+rot vs scalar = green light to build it into llama.cpp.

Usage: python vector_codebook_spike.py [hf:<model> | gguf:<path>]
Default: hf:Qwen/Qwen2.5-0.5B-Instruct (local, instant). For 3B: gguf:/d/pbr_work/models/...f16.gguf
"""
import sys, numpy as np, scipy.linalg
from sklearn.cluster import MiniBatchKMeans

GS = 64                      # scalar group size / scale group
HB = 128                     # Hadamard block
np.random.seed(0)

# ---------- rotation ----------
def had(n): return scipy.linalg.hadamard(n).astype(np.float32)/np.sqrt(n)
def _blk(n):
    if n % HB == 0: return HB
    b=1
    while b*2<=n and n%(b*2)==0: b*=2
    return max(2,b)
def rot2(W, seed=1):  # two-sided block randomized Hadamard (orthogonal)
    def one(M, axis, sd):
        rng=np.random.default_rng(sd); X=M if axis==1 else M.T; r,n=X.shape; B=_blk(n)
        s=rng.choice([-1.,1.],size=n).astype(np.float32); H=had(B)
        Y=((X*s).reshape(r,n//B,B) @ H.T).reshape(r,n)
        return Y if axis==1 else Y.T
    return one(one(W,1,seed),0,seed+1)

# ---------- scalar NF ----------
def nf_levels(bits):
    from scipy.stats import norm
    k=1<<bits; lv=norm.ppf((np.arange(k)+0.5)/k).astype(np.float32); return lv/np.abs(lv).max()
def q_scalar_nf(W, bits, gs=GS):
    o,i=W.shape; gs=gs if i%gs==0 else i; lv=nf_levels(bits)
    Wr=W.reshape(o,i//gs,gs); amax=np.abs(Wr).max(2,keepdims=True); amax[amax==0]=1
    idx=np.abs((Wr/amax)[...,None]-lv[None,None,None,:]).argmin(-1)
    Wh=(lv[idx]*amax).reshape(o,i)
    bpw=bits + 16.0/gs           # + fp16 scale per group
    return Wh, bpw

# ---------- vector codebook (VPTQ-style) ----------
def q_vector(W, vd, K, gs=GS):
    o,i=W.shape
    # per-group scale normalize (absmax over gs along input), then VQ the normalized weights
    gs2 = gs if i%gs==0 else i
    Wr=W.reshape(o,i//gs2,gs2); amax=np.abs(Wr).max(2,keepdims=True); amax[amax==0]=1
    Wn=(Wr/amax).reshape(o,i)
    n_vec = W.size//vd
    V=Wn.reshape(n_vec, vd)
    # train codebook on subsample, assign all
    ntr=min(200000, n_vec)
    sub=V[np.random.choice(n_vec, ntr, replace=False)] if n_vec>ntr else V
    km=MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, max_iter=60, random_state=0)
    km.fit(sub)
    idx=km.predict(V)
    Wn_hat=km.cluster_centers_[idx].reshape(o,i)
    Wh=(Wn_hat.reshape(o,i//gs2,gs2)*amax).reshape(o,i)
    bpw=(np.log2(K)/vd) + 16.0/gs2
    return Wh.astype(np.float32), bpw

def relerr(A,B): return float(np.linalg.norm(A-B)/(np.linalg.norm(A)+1e-12))

# ---------- load tensors ----------
def load_tensors(spec):
    if spec.startswith("gguf:"):
        from gguf import GGUFReader
        r=GGUFReader(spec[5:]); out={}
        for t in r.tensors:
            a=np.array(t.data)
            if a.ndim==2 and min(a.shape)>=256 and (("attn" in t.name) or ("ffn" in t.name)):
                out[t.name]=a.astype(np.float32)
        # pick a representative spread
        names=list(out); pick=names[:2]+names[len(names)//2:len(names)//2+2]+names[-2:]
        return {n:out[n] for n in pick}
    else:
        from transformers import AutoModelForCausalLM
        m=AutoModelForCausalLM.from_pretrained(spec[3:], torch_dtype="float32")
        sd=m.state_dict(); out={}
        want=["layers.0.self_attn.q_proj","layers.0.self_attn.o_proj","layers.0.mlp.gate_proj",
              "layers.0.mlp.down_proj","layers.12.self_attn.v_proj","layers.12.mlp.down_proj"]
        for w in want:
            k=[n for n in sd if w in n and n.endswith(".weight")]
            if k: out[k[0]]=sd[k[0]].numpy().astype(np.float32)
        return out

spec = sys.argv[1] if len(sys.argv)>1 else "hf:Qwen/Qwen2.5-0.5B-Instruct"
print("loading", spec); T=load_tensors(spec); print(f"{len(T)} tensors\n")

# tiers: (label, fn) grouped by ~bpw
tiers = {
  "~3.0bpw": [("scalarNF3",   lambda W: q_scalar_nf(W,3)),
              ("scalarNF3+rot",lambda W: q_scalar_nf(rot2(W),3)),
              ("VQ d2 K64",    lambda W: q_vector(W,2,64)),
              ("VQ d2 K64+rot",lambda W: q_vector(rot2(W),2,64))],
  "~2.25bpw":[("scalarNF2",   lambda W: q_scalar_nf(W,2)),
              ("scalarNF2+rot",lambda W: q_scalar_nf(rot2(W),2)),
              ("VQ d4 K256",   lambda W: q_vector(W,4,256)),
              ("VQ d4 K256+rot",lambda W: q_vector(rot2(W),4,256))],
}
# NOTE: for +rot, error is measured in the rotated domain (orthogonal => equals original-domain error)

for tier,methods in tiers.items():
    print(f"=== {tier} ===")
    print(f"{'method':<16} {'mean rel-err':>12} {'bpw':>6}")
    agg={}
    for label,fn in methods:
        errs=[]; bpw=0
        for name,W in T.items():
            Wt = rot2(W) if "+rot" in label else W    # rotate the reference too, so error is in the domain we quantized
            # for +rot methods fn already rotates internally; recompute consistently:
            if "+rot" in label:
                Wh,bpw=fn(W)                            # fn rotates W, quantizes; returns reconstruction in rotated domain
                errs.append(relerr(rot2(W), Wh))
            else:
                Wh,bpw=fn(W); errs.append(relerr(W,Wh))
        print(f"{label:<16} {np.mean(errs):>12.4f} {bpw:>6.2f}")
        agg[label]=np.mean(errs)
    # verdict for this tier
    base=[v for k,v in agg.items() if k.startswith("scalarNF") and "+rot" not in k][0]
    best=min(agg,key=agg.get)
    print(f"  best={best} ({agg[best]:.4f})  vs scalar baseline {base:.4f}  -> {100*(base-agg[best])/base:+.1f}%\n")
print("Read: if 'VQ + rot' beats scalar baseline substantially at ~3bpw, the vector")
print("codebook is the lever to make 3-bit lossless -> justifies the llama.cpp weight vec_dot.")
