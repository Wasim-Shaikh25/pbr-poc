# Cheap validation of drift-correction ("teacher targets") on REAL drifted layers.
#
# Uses a already-quantized checkpoint (drift baked in) as the source of realistic
# drifted inputs, and the original as the clean reference. For each target layer's
# tensors, re-quantizes two ways and asks which better reproduces the CLEAN output
# x_clean . W^T on HELD-OUT tokens:
#   plain   : quantize(W)      fed x_q          (current method)
#   teacher : quantize(W_eff)  fed x_q          (W_eff = W C^T H^-1, drift-cancelling)
# Also reports err_drift = how far uncorrected W on drifted input already is.
#
# Expectation from eval_accum: teacher should help at the deep knee (L22,L23) and
# do ~nothing at a stable mid layer (L12 control).
#
# Usage: PBR_LAYERS=12,22,23 python validate_teacher.py ./qwen05b <quant_ckpt>
import os, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

orig_dir, quant_dir = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in os.environ.get("PBR_LAYERS", "12,22,23").split(",")]
STAGES = [int(x) for x in os.environ.get("PBR_STAGES", "256,256").split(",")]
SAMPLES = int(os.environ.get("PBR_SAMPLES", "24"))
SEQ = int(os.environ.get("PBR_SEQ", "512"))
TEACHER_DAMP = float(os.environ.get("PBR_TEACHER_DAMP", "0.05"))
DAMP = 0.01
D = 8
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)
rc = {}
def rotation(n):
    if n not in rc: rc[n] = ortho(n, n)
    return rc[n]
def assign(V, C, ch=50000):
    return np.concatenate([((V[i:i+ch]**2).sum(1)[:,None]-2*V[i:i+ch]@C.T+(C**2).sum(1)[None]).argmin(1)
                           for i in range(0,len(V),ch)])
def kmeans(V,k,it=8):
    rng=np.random.default_rng(0); C=V[rng.choice(len(V),k,replace=False)]
    for _ in range(it):
        a=assign(V,C)
        for j in range(k):
            m=a==j
            if m.any(): C[j]=V[m].mean(0)
    return C
def quantize(W,H,stages):
    O,I=W.shape; Ro,Ri=rotation(O),rotation(I)
    Wr=Ro@W@Ri; Hr=Ri.T@H@Ri; Hr=Hr+DAMP*np.mean(np.diag(Hr))*np.eye(I)
    U=np.linalg.cholesky(np.linalg.inv(Hr)).T
    s=np.sqrt((Wr**2).mean(1,keepdims=True))+1e-12
    pool=(Wr/s).reshape(-1,D); books=[]; Rres=pool.copy()
    for k in stages:
        C=kmeans(Rres,k); a=assign(Rres,C); books.append(C); Rres=Rres-C[a]
    T=Wr.copy(); Q=np.zeros_like(T)
    for j in range(0,I,D):
        b=slice(j,j+D); R=T[:,b]/s; q=np.zeros_like(R)
        for C in books:
            a=((R**2).sum(1)[:,None]-2*R@C.T+(C**2).sum(1)[None]).argmin(1); q+=C[a]; R-=C[a]
        Q[:,b]=q*s
        if j+D<I: T[:,j+D:]-=(T[:,b]-Q[:,b])@np.linalg.solve(U[b,b],U[b,j+D:])
    return Ro.T@Q@Ri.T

tok=AutoTokenizer.from_pretrained(orig_dir)
from datasets import load_dataset
try: _wt=load_dataset("wikitext","wikitext-2-raw-v1",split="train")
except Exception: _wt=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
text="\n\n".join(_wt["text"][:4000])
ids=tok(text,return_tensors="pt").input_ids
starts=np.random.default_rng(0).integers(0,ids.shape[1]-SEQ,SAMPLES)
batch=torch.cat([ids[:,s:s+SEQ] for s in starts],dim=0)

print("loading clean + quantized models ...",flush=True)
m_clean=AutoModelForCausalLM.from_pretrained(orig_dir,dtype=torch.float32).eval()
m_quant=AutoModelForCausalLM.from_pretrained(quant_dir,dtype=torch.float32).eval()

class Stop(Exception): pass
def capture(m,li,targets):
    blk=m.model.layers[li]; buf={n:[] for n in targets}
    hs=[blk.get_submodule(n).register_forward_hook(
         lambda mo,inp,o,n=n: buf[n].append(inp[0].detach().double())) for n in targets]
    st=blk.register_forward_hook(lambda mo,i,o:(_ for _ in()).throw(Stop()))
    with torch.no_grad():
        for i in range(0,SAMPLES,8):
            try: m(batch[i:i+8])
            except Stop: pass
    st.remove(); [h.remove() for h in hs]
    return {n:torch.cat(buf[n]).reshape(-1,buf[n][0].shape[-1]).numpy() for n in targets}

print(f"\n{'layer.tensor':<28}{'drift%':>9}{'plain%':>9}{'teach%':>9}{'winner':>9}")
summ={}
for li in LAYERS:
    blk=m_quant.model.layers[li]
    targets=[]
    for n in TARGETS:
        try: blk.get_submodule(n); targets.append(n)
        except AttributeError: pass
    Xq=capture(m_quant,li,targets); Xc=capture(m_clean,li,targets)
    for n in targets:
        W=m_clean.model.layers[li].get_submodule(n).weight.detach().double().numpy()
        xq=Xq[n]; xc=Xc[n]; ntok=len(xq); half=ntok//2
        xqf,xqe=xq[:half],xq[half:]; xcf,xce=xc[:half],xc[half:]
        Hf=xqf.T@xqf/len(xqf)
        Teval=xce@W.T
        # plain
        Wq_p=quantize(W,Hf,STAGES)
        # teacher
        Cf=xqf.T@xcf/len(xqf); Hd=Hf+TEACHER_DAMP*np.mean(np.diag(Hf))*np.eye(Hf.shape[0])
        Weff=W@Cf.T@np.linalg.inv(Hd); Wq_t=quantize(Weff,Hf,STAGES)
        def rel(out): return np.linalg.norm(out-Teval)/np.linalg.norm(Teval)*100
        e_drift=rel(xqe@W.T); e_plain=rel(xqe@Wq_p.T); e_teach=rel(xqe@Wq_t.T)
        win="teacher" if e_teach<e_plain-1e-6 else "plain"
        summ.setdefault(li,[]).append((e_plain,e_teach))
        print(f"L{li}.{n.split('.')[-1]:<24}{e_drift:8.2f} {e_plain:8.2f} {e_teach:8.2f} {win:>9}",flush=True)

print("\n=== per-layer mean (clean-target rel err) ===")
print(f"{'layer':<8}{'plain%':>9}{'teach%':>9}{'delta':>9}")
for li in LAYERS:
    ps=np.mean([p for p,t in summ[li]]); ts=np.mean([t for p,t in summ[li]])
    print(f"L{li:<7}{ps:8.2f} {ts:8.2f} {ts-ps:+8.2f}")
