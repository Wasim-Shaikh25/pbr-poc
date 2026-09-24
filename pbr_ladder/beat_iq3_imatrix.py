#!/usr/bin/env python
"""
Phase 2b — imatrix-weighted VQ vs real IQ3, in the WEIGHTED error space that matters.

IQ3's edge was importance weighting (imatrix) + mixed precision. Here we give our VQ the
same imatrix and compare fairly: minimize and measure importance-weighted reconstruction
error  ||W - W_hat||_imp  (imp_j = per-input-column sum of squared activations), which is
what actually affects model output. Rotation dropped (confirmed neutral for VQ).

  weighted VQ: scale columns by sqrt(imp), VQ, unscale  ->  plain-L2 in scaled space == weighted-L2 in W.
Compare weighted rel-err of real IQ3_M (3.86 bpw, imatrix) vs our imatrix-VQ (~3.0-3.25 bpw).
If ours <= IQ3 at equal-or-fewer bits, that's a fair beat-IQ3 in the space that counts.
"""
import numpy as np, scipy.linalg
from gguf import GGUFReader, quants as gq
from sklearn.cluster import MiniBatchKMeans
np.random.seed(0)
F16="D:/pbr_work/models/Qwen2.5-3B-Instruct-f16.gguf"
IQ3="D:/pbr_work/out/qwen3b-iq3-imat.gguf"
IMAT="D:/pbr_work/out/qwen3b-ship3.imatrix.dat"
GS=64

def load_imat():
    r=GGUFReader(IMAT); out={}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            base=t.name[:-len(".in_sum2")]
            out[base]=np.array(t.data,dtype=np.float32)
    return out

def align(W, L):  # return W as (out, in=L)
    if W.shape[1]==L: return W
    if W.shape[0]==L: return W.T.copy()
    return None

def vq_weighted(W, imp, vd, K, gs=GS):
    o,i=W.shape
    w=np.sqrt(np.maximum(imp,1e-12)).astype(np.float32)      # per-column weight
    Wsc=W*w[None,:]                                           # emphasize important columns
    gs2=gs if i%gs==0 else i
    Wr=Wsc.reshape(o,i//gs2,gs2); am=np.abs(Wr).max(2,keepdims=True); am[am==0]=1
    Wn=(Wr/am).reshape(o,i); nv=W.size//vd; V=Wn.reshape(nv,vd)
    ntr=min(200000,nv); sub=V[np.random.choice(nv,ntr,replace=False)] if nv>ntr else V
    km=MiniBatchKMeans(n_clusters=K,batch_size=8192,n_init=3,max_iter=60,random_state=0).fit(sub)
    Wsc_hat=(km.cluster_centers_[km.predict(V)].reshape(o,i//gs2,gs2)*am).reshape(o,i)
    Wh=Wsc_hat/w[None,:]                                      # unscale back
    return Wh.astype(np.float32), (np.log2(K)/vd)+16.0/gs2

def werr(W,Wh,imp):  # importance-weighted rel-error
    d2=((W-Wh)**2).sum(0); n2=(W**2).sum(0)
    return float(np.sqrt((d2*imp).sum()/((n2*imp).sum()+1e-12)))

print("loading f16 + IQ3 + imatrix ...")
rf=GGUFReader(F16); ri=GGUFReader(IQ3); IM=load_imat()
f16={t.name:t for t in rf.tensors}; iq3={t.name:t for t in ri.tensors}
names=[n for n in f16 if n in iq3 and n in IM and f16[n].data.ndim==2
       and min(f16[n].shape)>=256 and ("attn" in n or "ffn" in n)]
names=names[:3]+names[len(names)//2-1:len(names)//2+2]+names[-3:]
print(f"comparing {len(names)} tensors (weighted error)\n")

configs=[("iVQ d2 K64",2,64),("iVQ d4 K4096",4,4096)]
print(f"{'tensor':<26} {'IQ3(w)':>8} "+" ".join(f"{c[0]:>13}" for c in configs))
iq3e=[]; agg={c[0]:[] for c in configs}; bpw={}
for n in names:
    imp=IM[n]
    W=align(gq.dequantize(f16[n].data,f16[n].tensor_type).astype(np.float32), len(imp))
    Wi=align(gq.dequantize(iq3[n].data,iq3[n].tensor_type).astype(np.float32), len(imp))
    if W is None or Wi is None:
        print(f"{n:<26} (axis mismatch, skip)"); continue
    ei=werr(W,Wi,imp); iq3e.append(ei); row=[]
    for label,vd,K in configs:
        Wh,b=vq_weighted(W,imp,vd,K); e=werr(W,Wh,imp); agg[label].append(e); bpw[label]=b; row.append(e)
    print(f"{n:<26} {ei:>8.4f} "+" ".join(f"{e:>13.4f}" for e in row))

print(f"\n{'mean IQ3 (weighted)':<22}{np.mean(iq3e):.4f}   (3.86 bpw)")
for label,vd,K in configs:
    m=np.mean(agg[label]); d=100*(np.mean(iq3e)-m)/np.mean(iq3e)
    print(f"{label:<22}{m:.4f}   ~{bpw[label]:.2f} bpw  -> {d:+.1f}% vs IQ3  [{'BEATS' if d>0 else 'loses'}]")
print("\nFair test: same imatrix, weighted error (the space that affects model output).")
print("Beat at lower bpw => genuine win. Close/lose => IQ3 is at the practical ceiling.")
