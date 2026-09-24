#!/usr/bin/env python
"""
Phase 2 decisive proxy — can rotation + vector codebook BEAT real IQ3 on Qwen-3B?

Apples-to-apples weight-reconstruction error against the ACTUAL llama.cpp IQ3_M model:
  - reference W  = f16 GGUF tensor (dequantized)
  - IQ3 error    = ||W - dequant(IQ3_M tensor)|| / ||W||     (real llama.cpp, ~3.86 bpw, WITH imatrix)
  - our error    = ||rot(W) - VQ(rot(W))|| / ||rot(W)||       (== original-domain error; orthogonal)
Compare our error at ~3.0-3.25 bpw vs IQ3's at 3.86 bpw. If ours <= IQ3 at equal-or-fewer
bits, that's a genuine beat-IQ3 (plain L2; IQ3 additionally has an imatrix advantage we do
NOT yet use, so a win here is conservative). Honest caveat printed with the result.
"""
import numpy as np, scipy.linalg
from gguf import GGUFReader, quants as gq
from sklearn.cluster import MiniBatchKMeans
np.random.seed(0)
F16="D:/pbr_work/models/Qwen2.5-3B-Instruct-f16.gguf"
IQ3="D:/pbr_work/out/qwen3b-iq3-imat.gguf"
GS=64; HB=128

def had(n): return scipy.linalg.hadamard(n).astype(np.float32)/np.sqrt(n)
def _blk(n):
    if n%HB==0: return HB
    b=1
    while b*2<=n and n%(b*2)==0: b*=2
    return max(2,b)
def rot2(W,seed=1):
    def one(M,axis,sd):
        rng=np.random.default_rng(sd); X=M if axis==1 else M.T; r,n=X.shape; B=_blk(n)
        s=rng.choice([-1.,1.],size=n).astype(np.float32); H=had(B)
        Y=((X*s).reshape(r,n//B,B)@H.T).reshape(r,n); return Y if axis==1 else Y.T
    return one(one(W,1,seed),0,seed+1)
def q_vector(W,vd,K,gs=GS):
    o,i=W.shape; gs2=gs if i%gs==0 else i
    Wr=W.reshape(o,i//gs2,gs2); amax=np.abs(Wr).max(2,keepdims=True); amax[amax==0]=1
    Wn=(Wr/amax).reshape(o,i); nv=W.size//vd; V=Wn.reshape(nv,vd)
    ntr=min(200000,nv); sub=V[np.random.choice(nv,ntr,replace=False)] if nv>ntr else V
    km=MiniBatchKMeans(n_clusters=K,batch_size=8192,n_init=3,max_iter=60,random_state=0); km.fit(sub)
    Wh=(km.cluster_centers_[km.predict(V)].reshape(o,i//gs2,gs2)*amax).reshape(o,i)
    return Wh.astype(np.float32), (np.log2(K)/vd)+16.0/gs2
def relerr(A,B): return float(np.linalg.norm(A-B)/(np.linalg.norm(A)+1e-12))

print("loading f16 + IQ3 GGUF ...")
rf=GGUFReader(F16); ri=GGUFReader(IQ3)
f16={t.name:t for t in rf.tensors}; iq3={t.name:t for t in ri.tensors}
names=[n for n in f16 if n in iq3 and f16[n].data.ndim==2 and min(f16[n].shape)>=256
       and ("attn" in n or "ffn" in n)]
# representative spread across depth
names=names[:3]+names[len(names)//2-1:len(names)//2+2]+names[-3:]
print(f"comparing {len(names)} tensors\n")

configs=[("VQ d2 K64",2,64),("VQ d4 K4096",4,4096)]
print(f"{'tensor':<26} {'IQ3 err':>8} "+" ".join(f"{c[0]:>13}" for c in configs))
sums={c[0]:[] for c in configs}; iq3errs=[]; bpws={}
for n in names:
    W=gq.dequantize(f16[n].data, f16[n].tensor_type).astype(np.float32)
    Wi=gq.dequantize(iq3[n].data, iq3[n].tensor_type).astype(np.float32)
    ei=relerr(W,Wi); iq3errs.append(ei)
    Wr=rot2(W)
    row=[]
    for label,vd,K in configs:
        Wh,bpw=q_vector(Wr,vd,K); e=relerr(Wr,Wh); sums[label].append(e); bpws[label]=bpw; row.append(e)
    print(f"{n:<26} {ei:>8.4f} "+" ".join(f"{e:>13.4f}" for e in row))

print(f"\n{'mean IQ3 err':<20}{np.mean(iq3errs):.4f}   (IQ3_M ~3.86 bpw, WITH imatrix)")
for label,vd,K in configs:
    m=np.mean(sums[label]); d=100*(np.mean(iq3errs)-m)/np.mean(iq3errs)
    verdict="BEATS IQ3" if d>0 else "loses to IQ3"
    print(f"{label:<20}{m:.4f}   ~{bpws[label]:.2f} bpw  -> {d:+.1f}% vs IQ3  [{verdict}]")
print("\nCaveat: plain L2; IQ3 uses imatrix (importance weighting) we do NOT yet apply,")
print("so any win here is conservative. If we win, adding imatrix widens the gap.")
print("If we lose, next step = imatrix-weighted VQ + larger codebook before any kernel work.")
