# FAST path: take the already-VQ-quantized model (k-means done) and bolt the activation
# low-rank correction on top, layer-by-layer with SEQUENTIAL activations (each layer sees
# upstream layers already quantized AND corrected). Skips all k-means (~470s/layer). Then
# eval WikiText-2 test PPL inline. Compares to the base VQ model's 16.829 @ 3.125 bpw.
#
# Usage: python apply_lowrank.py <orig_dir> <quantized_dir> <rank>
import sys, os, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
orig_dir, q_dir, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
SAMPLES, SEQ = int(os.environ.get("PBR_SAMPLES", 64)), int(os.environ.get("PBR_SEQ", 512))
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
class StopForward(Exception): pass

def lowrank_correct(W, Wq, H, r):
    E = W - Wq
    ev, Ug = np.linalg.eigh(H); ev = ev[::-1]; Ug = Ug[:, ::-1]
    sq = np.sqrt(np.clip(ev, 0, None))
    B = (E @ Ug) * sq[None, :]
    P, S, Qt = np.linalg.svd(B, full_matrices=False)
    r = min(r, len(S)); Br = (P[:, :r] * S[:r]) @ Qt[:r]
    inv = np.where(sq > 1e-9, 1.0 / sq, 0.0)
    C = (Br * inv[None, :]) @ Ug.T
    Pc, Sc, Qc = np.linalg.svd(C, full_matrices=False)
    m1 = np.abs(Pc[:, :r] * np.sqrt(Sc[:r])).max(); m2 = np.abs(np.sqrt(Sc[:r])[:, None] * Qc[:r]).max()
    Uf = np.round((Pc[:, :r] * np.sqrt(Sc[:r])) / m1 * 127) * (m1 / 127)
    Vf = np.round((np.sqrt(Sc[:r])[:, None] * Qc[:r]) / m2 * 127) * (m2 / 127)
    return Wq + Uf @ Vf

print(f"loading orig weights from {orig_dir} ...", flush=True)
tok = AutoTokenizer.from_pretrained(orig_dir)
omodel = AutoModelForCausalLM.from_pretrained(orig_dir, dtype=torch.float32).eval()
orig_W = {}
for L, block in enumerate(omodel.model.layers):
    for name in TARGETS:
        try: orig_W[(L, name)] = block.get_submodule(name).weight.detach().double().numpy()
        except AttributeError: pass
del omodel

print(f"loading quantized model from {q_dir} ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(q_dir, dtype=torch.float32).eval()
layers = model.model.layers; n_layers = len(layers)

# calibration batch (WikiText-2 train), same recipe as the driver
from datasets import load_dataset
try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tok("\n\n".join(_wt["text"][:4000]), return_tensors="pt").input_ids
starts = np.random.default_rng(0).integers(0, ids.shape[1] - SEQ, SAMPLES)
batch = torch.cat([ids[:, s:s+SEQ] for s in starts], dim=0)

t0 = time.time()
for L in range(n_layers):
    block = layers[L]
    targets = {}
    for name in TARGETS:
        try: targets[name] = block.get_submodule(name)
        except AttributeError: pass
    inputs = {name: [] for name in targets}
    hooks = [mod.register_forward_hook(
        lambda m, inp, o, nm=name: inputs[nm].append(inp[0].detach().double())) for name, mod in targets.items()]
    stop = block.register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(StopForward()))
    with torch.no_grad():
        for i in range(0, SAMPLES, 8):
            try: model(batch[i:i+8])
            except StopForward: pass
    stop.remove()
    for h in hooks: h.remove()
    for name, mod in targets.items():
        X = torch.cat(inputs[name]).reshape(-1, mod.in_features).numpy()
        inputs[name].clear()
        H = X.T @ X / len(X); del X
        Wq = mod.weight.detach().double().numpy()
        Wnew = lowrank_correct(orig_W[(L, name)], Wq, H, R)
        with torch.no_grad(): mod.weight.copy_(torch.from_numpy(Wnew).to(mod.weight.dtype))
    print(f"layer {L:2d}/{n_layers-1} corrected  [{time.time()-t0:6.1f}s]", flush=True)

# inline WikiText-2 TEST perplexity
try: _te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
except Exception: _te = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
tids = tok("\n\n".join(_te["text"]), return_tensors="pt").input_ids
seq, nll, n = 2048, 0.0, 0
with torch.no_grad():
    for i in range(0, tids.shape[1] - 1, seq):
        x = tids[:, i:i+seq]
        if x.shape[1] < 2: break
        nll += model(x, labels=x).loss.float().item() * (x.shape[1] - 1); n += x.shape[1] - 1
dbpw = R * 8.0 * sum((o + i) for (o, i) in [orig_W[k].shape for k in orig_W]) / sum(orig_W[k].size for k in orig_W)
print(f"\n=== rank-{R} low-rank correction on VQ@3.125 ===")
print(f"added bpw ~{dbpw:.3f}  (total ~{3.125+dbpw:.3f} bpw)")
print(f"WikiText-2 test perplexity: {np.exp(nll / n):.3f}   (base VQ@3.125 = 16.829, baseline = 14.247)")
