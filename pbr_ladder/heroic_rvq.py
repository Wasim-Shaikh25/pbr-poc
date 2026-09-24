#!/usr/bin/env python
"""
Phase 2c — the heroic attempt: RVQ + imatrix + MATCHED per-tensor bit budget vs IQ3.

Controls for everything IQ3 has except the codebook itself:
  - imatrix weighting (same importance)               -> scale cols by sqrt(imp)
  - matched per-tensor bpw (IQ3's own bit allocation) -> give RVQ the SAME bits IQ3 spent
  - residual multi-stage VQ (RVQ), good k-means        -> strongest training-free codebook
Then compare weighted rel-err at EQUAL bits. If ours < IQ3 -> our codebook genuinely beats
IQ3's (real win). If >= -> IQ3's codebook is at/above the ceiling and we stop. Honest either way.
"""
import numpy as np
from gguf import GGUFReader, quants as gq
from sklearn.cluster import KMeans
np.random.seed(0)
F16="D:/pbr_work/models/Qwen2.5-3B-Instruct-f16.gguf"
IQ3="D:/pbr_work/out/qwen3b-iq3-imat.gguf"
IMAT="D:/pbr_work/out/qwen3b-ship3.imatrix.dat"
GS=64; VD=2

def load_imat():
    r=GGUFReader(IMAT); out={}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"): out[t.name[:-8]]=np.array(t.data,dtype=np.float32)
    return out
def align(W,L):
    if W.shape[1]==L: return W
    if W.shape[0]==L: return W.T.copy()
    return None
def km_fit(X,K):
    ntr=min(150000,len(X)); sub=X[np.random.choice(len(X),ntr,replace=False)] if len(X)>ntr else X
    return KMeans(n_clusters=K,n_init=3,max_iter=100,random_state=0).fit(sub)
def rvq_weighted(W, imp, bpw_target, vd=VD, gs=GS):
    o,i=W.shape; w=np.sqrt(np.maximum(imp,1e-12)).astype(np.float32)
    Wsc=W*w[None,:]; gs2=gs if i%gs==0 else i
    Wr=Wsc.reshape(o,i//gs2,gs2); am=np.abs(Wr).max(2,keepdims=True); am[am==0]=1
    V=(Wr/am).reshape(o,i).reshape(-1,vd)
    ibits=max(1.0,(bpw_target-16.0/gs2)*vd)          # total index bits per vector
    stages=[]; rem=ibits
    while rem>0.5 and len(stages)<3:
        b=int(min(8,round(rem))); b=max(1,b); stages.append(b); rem-=b
    recon=np.zeros_like(V); res=V.copy()
    for b in stages:
        km=km_fit(res,2**b); recon=recon+km.cluster_centers_[km.predict(res)]; res=res-km.cluster_centers_[km.predict(res)]
    Wh=((recon.reshape(o,i)).reshape(o,i//gs2,gs2)*am).reshape(o,i)/w[None,:]
    return Wh.astype(np.float32), sum(stages)/vd + 16.0/gs2
def werr(W,Wh,imp):
    d2=((W-Wh)**2).sum(0); n2=(W**2).sum(0)
    return float(np.sqrt((d2*imp).sum()/((n2*imp).sum()+1e-12)))

print("loading ...")
rf=GGUFReader(F16); ri=GGUFReader(IQ3); IM=load_imat()
f16={t.name:t for t in rf.tensors}; iq3={t.name:t for t in ri.tensors}
names=[n for n in f16 if n in iq3 and n in IM and f16[n].data.ndim==2
       and min(f16[n].shape)>=256 and ("attn" in n or "ffn" in n)]
names=names[:3]+names[len(names)//2-1:len(names)//2+2]+names[-3:]
print(f"{len(names)} tensors, RVQ matched to IQ3 per-tensor bpw\n")
print(f"{'tensor':<26} {'IQ3 bpw':>7} {'IQ3(w)':>8} {'ourbpw':>7} {'RVQ(w)':>8} {'win?':>5}")
ie=[]; oe=[]
for n in names:
    imp=IM[n]
    W=align(gq.dequantize(f16[n].data,f16[n].tensor_type).astype(np.float32),len(imp))
    Wi=align(gq.dequantize(iq3[n].data,iq3[n].tensor_type).astype(np.float32),len(imp))
    if W is None or Wi is None: print(f"{n:<26} skip"); continue
    nel=W.size; iq3_bpw=iq3[n].data.nbytes*8.0/nel
    ei=werr(W,Wi,imp)
    Wh,obpw=rvq_weighted(W,imp,iq3_bpw)
    eo=werr(W,Wh,imp)
    ie.append(ei); oe.append(eo)
    print(f"{n:<26} {iq3_bpw:>7.2f} {ei:>8.4f} {obpw:>7.2f} {eo:>8.4f} {'YES' if eo<ei else 'no':>5}")
mi,mo=np.mean(ie),np.mean(oe); d=100*(mi-mo)/mi
print(f"\nmean: IQ3(w)={mi:.4f}  ourRVQ(w)={mo:.4f}  -> {d:+.1f}% vs IQ3  [{'BEATS IQ3' if d>0 else 'loses'}]")
print("Equal per-tensor bits + same imatrix => this isolates codebook quality alone.")
print("If we still lose here, IQ3's codebook is the ceiling and the quantizer chase ends honestly.")
