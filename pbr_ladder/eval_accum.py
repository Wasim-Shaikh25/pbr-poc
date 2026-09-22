# Layer-by-layer error accumulation probe for a quantized checkpoint.
#
# Runs the ORIGINAL model and a QUANTIZED checkpoint in lockstep on the same text
# and reports how the residual-stream error grows with depth:
#   rel_err(l) = || h_q(l) - h(l) || / || h(l) ||   at the OUTPUT of decoder layer l
#
# Shape of that curve is the whole game:
#   * linear in l            -> benign incoherent accumulation (a bits problem)
#   * super-linear / blow-up -> the network AMPLIFIES quantization noise (structural)
#   * a late knee            -> a few deep layers / the final norm+head see
#                               out-of-distribution input (the §9.6 drift hypothesis)
#
# Also reports the final next-token cross-entropy (=> perplexity) for both, so the
# accumulation curve is anchored to the real PPL number.
#
# Usage: python eval_accum.py ./qwen05b ./qwen05b_flat3875
import sys, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

orig_dir, quant_dir = sys.argv[1], sys.argv[2]
NSEQ = int(os.environ.get("PBR_NSEQ", "8"))
SEQ = int(os.environ.get("PBR_SEQ", "1024"))

tok = AutoTokenizer.from_pretrained(orig_dir)
if os.environ.get("PBR_TEXT"):
    text = open(os.environ["PBR_TEXT"], encoding="utf-8").read()
else:
    from datasets import load_dataset
    try:
        _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(_wt["text"])
ids = tok(text, return_tensors="pt").input_ids
# take NSEQ non-overlapping windows
wins = [ids[:, i*SEQ:(i+1)*SEQ] for i in range(NSEQ) if (i+1)*SEQ <= ids.shape[1]]
print(f"{len(wins)} windows x {SEQ} tokens", flush=True)

def load(d):
    return AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()

def capture(model, x):
    """Return (list of per-layer output hidden states, final loss)."""
    hs = []
    hooks = [blk.register_forward_hook(lambda m, i, o: hs.append(o[0].detach()))
             for blk in model.model.layers]
    with torch.no_grad():
        out = model(x, labels=x)
    for h in hooks: h.remove()
    return hs, out.loss.float().item()

print("loading original ...", flush=True); m0 = load(orig_dir)
print("loading quantized ...", flush=True); mq = load(quant_dir)
L = len(m0.model.layers)
acc = np.zeros(L); sig = np.zeros(L); nll0 = nllq = ntok = 0.0
for wi, x in enumerate(wins):
    h0, l0 = capture(m0, x)
    hq, lq = capture(mq, x)
    ntok += x.shape[1] - 1
    nll0 += l0 * (x.shape[1] - 1); nllq += lq * (x.shape[1] - 1)
    for l in range(L):
        e = (hq[l] - h0[l]).double()
        acc[l] += (e**2).sum().item()
        sig[l] += (h0[l].double()**2).sum().item()
    print(f"  window {wi}: baseline nll {l0:.4f}  quant nll {lq:.4f}", flush=True)

rel = np.sqrt(acc / sig)              # RMS relative error of the residual stream after layer l
print("\nlayer   rel_err(out)   step-up vs prev")
prev = 0.0
for l in range(L):
    print(f"  {l:2d}     {rel[l]*100:8.3f}%     {(rel[l]-prev)*100:+7.3f}%")
    prev = rel[l]
ppl0, pplq = np.exp(nll0/ntok), np.exp(nllq/ntok)
print(f"\nbaseline PPL {ppl0:.3f}   quantized PPL {pplq:.3f}   (+{100*(pplq/ppl0-1):.2f}%)")
# fit growth exponent: rel ~ (l+1)^p   -> p>0.5 super-random-walk, p~=0.5 incoherent, p~=1 linear/coherent
ll = np.arange(1, L+1); msk = rel > 0
p = np.polyfit(np.log(ll[msk]), np.log(rel[msk]), 1)[0]
print(f"growth exponent p (rel_err ~ layer^p): {p:.2f}   "
      f"[~0.5 incoherent random walk, ~1.0 linear/amplifying]")
