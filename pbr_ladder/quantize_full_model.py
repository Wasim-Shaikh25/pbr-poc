# Whole-model quantizer: reads a real Qwen checkpoint, quantizes every attention
# and MLP linear layer (rotation + GPTQ-style error feedback + 8-D node codebooks,
# the method proven layer-by-layer earlier), and WRITES OUT A NEW, COMPLETE,
# LOADABLE CHECKPOINT FOLDER — a real new "Qwen file", not just an in-memory score.
#
# Sequential, like real GPTQ: layer 0 is quantized first and written back into the
# live model, THEN calibration text is run again so layer 1 sees layer 0's real
# (already-quantized) output — not the original BF16 output. This matches what
# production quantizers do and what Phase 2 of the roadmap calls for.
#
# UNTESTED in the author's sandbox (no Hugging Face access). Needs: torch,
# transformers, and either `datasets` or a local calibration text file.
#
# Usage:
#   python quantize_full_model.py ./qwen05b ./qwen05b_pbr3bit
#   PBR_TEXT=calib.txt python quantize_full_model.py ./qwen05b ./qwen05b_pbr3bit
#   PBR_STAGES=256,256,64 PBR_SAMPLES=64 PBR_SEQ=512 PBR_LAYERS=0,1,2 \
#     python quantize_full_model.py ./qwen05b ./qwen05b_test   # quick dry run
#   # Mixed bits, one sequential pass (unset layers keep PBR_STAGES):
#   PBR_LAYERS=0,1,2 PBR_STAGES=256,256,64 PBR_LAYER_STAGES='0:512,256,64' \
#     python quantize_full_model.py ./qwen05b ./qwen05b_L0hi
#
# PBR_LAYERS restricts to a few layers for a fast first check before committing
# to a full run, which can take a long time on CPU for a 0.5B model.
# PBR_LAYER_STAGES is optional: "layer:k1,k2,...;layer:k1,k2,..." overrides
# PBR_STAGES for those layers only. Omitted, every selected layer uses PBR_STAGES.
#
# Resume: after each finished layer the script writes
#   <out>/layer_npz/layer_<i>.npz
#   <out>/pbr_progress.json
# A later invocation with the same output path and the same recipe (stages,
# per-layer overrides, sample count, sequence length, layer list) reloads
# those weights and continues at the next layer. The quantization math is
# unchanged. Delete the output folder to force a full re-run. A progress
# file whose recipe does not match this invocation is refused.

import gc, os, sys, time, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
STAGES = [int(x) for x in os.environ.get("PBR_STAGES", "256,256,64").split(",")]
SAMPLES = int(os.environ.get("PBR_SAMPLES", "64"))
SEQ = int(os.environ.get("PBR_SEQ", "512"))
PERLAYER = os.environ.get("PBR_ROT_PERLAYER", "0") == "1"  # per-(proj,layer) rotation vs one shared per width

def parse_layer_stages(spec):
    """'0:512,256,64;1:256,256,256' -> {0: [512,256,64], 1: [256,256,256]}."""
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        layer_s, ks_s = part.split(":")
        out[int(layer_s)] = [int(x) for x in ks_s.split(",") if x.strip()]
    return out

LAYER_STAGES = parse_layer_stages(os.environ.get("PBR_LAYER_STAGES", ""))
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
D = 8            # node-vector width, same as the proven method
K = 256          # nodes per codebook

print(f"Loading model from {src} ...")
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
layers = model.model.layers
n_layers = len(layers)
which = [int(x) for x in os.environ["PBR_LAYERS"].split(",")] if os.environ.get("PBR_LAYERS") else list(range(n_layers))
print(f"Model has {n_layers} layers; quantizing: {which}", flush=True)

# PBR_INIT_CKPT: seed the NOT-selected layers from an already-quantized checkpoint,
# so a run that only re-quantizes a few layers (e.g. richer end layers) inherits the
# rest instead of leaving them at BF16. Used for curve-guided mixed precision.
if os.environ.get("PBR_INIT_CKPT"):
    from safetensors import safe_open as _sopen
    _ck = os.path.join(os.environ["PBR_INIT_CKPT"], "model.safetensors")
    _skip = set(which)
    with _sopen(_ck, "pt") as _f:
        _keys = set(_f.keys())
        with torch.no_grad():
            for _li in range(n_layers):
                if _li in _skip:
                    continue
                for _t in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                    _kn = f"model.layers.{_li}.{_t}.weight"
                    if _kn in _keys:
                        layers[_li].get_submodule(_t).weight.copy_(_f.get_tensor(_kn).float())
    print(f"Seeded non-selected layers from {os.environ['PBR_INIT_CKPT']}", flush=True)

# ---- per-layer resume (weights only; does not change the quantizer) ----
PROGRESS_VERSION = 1

def recipe_record():
    return {
        "version": PROGRESS_VERSION,
        "stages": STAGES,
        "layer_stages": {str(k): v for k, v in sorted(LAYER_STAGES.items())},
        "samples": SAMPLES,
        "seq": SEQ,
        "layers": which,
        "rot_perlayer": PERLAYER,
    }

def progress_path():
    return os.path.join(out, "pbr_progress.json")

def layer_npz_path(li):
    return os.path.join(out, "layer_npz", f"layer_{li}.npz")

def load_progress():
    path = progress_path()
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        prog = json.load(f)
    want = recipe_record()
    got = {k: prog.get(k) for k in want}
    if got != want:
        raise SystemExit(
            f"Partial checkpoint at {out} does not match this recipe.\n"
            f"  file: {json.dumps(got)}\n  this run: {json.dumps(want)}\n"
            f"Refusing to resume or overwrite. Remove {out} to start over.")
    done = [int(x) for x in prog.get("done", [])]
    if done != which[:len(done)]:
        raise SystemExit(
            f"Progress done={done} is not a prefix of layers={which}. Refusing to resume.")
    missing = [li for li in done if not os.path.isfile(layer_npz_path(li))]
    if missing:
        raise SystemExit(f"Progress lists layers {missing} but their npz files are missing.")
    return done

def restore_done_layers(done):
    for li in done:
        data = np.load(layer_npz_path(li))
        block = layers[li]
        for name in data.files:
            mod = block.get_submodule(name)
            arr = np.array(data[name], copy=True)
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(arr).to(dtype=mod.weight.dtype))
        data.close()
        print(f"restored layer {li:2d} from {layer_npz_path(li)}", flush=True)

def save_layer_progress(li):
    os.makedirs(os.path.join(out, "layer_npz"), exist_ok=True)
    block = layers[li]
    payload = {}
    for name in TARGETS:
        try:
            mod = block.get_submodule(name)
        except AttributeError:
            continue
        payload[name] = mod.weight.detach().float().cpu().numpy()
    partial = os.path.join(out, "layer_npz", f"layer_{li}.partial.npz")
    np.savez(partial, **payload)
    os.replace(partial, layer_npz_path(li))
    done_now = which[:which.index(li) + 1]
    rec = recipe_record()
    rec["done"] = done_now
    tmp = progress_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    os.replace(tmp, progress_path())
    print(f"progress saved through layer {li}", flush=True)

done_layers = load_progress()
if done_layers:
    print(f"Resuming from {out}: {len(done_layers)} layer(s) already quantized "
          f"({done_layers[0]}..{done_layers[-1]}). Restoring, then continuing.", flush=True)
    restore_done_layers(done_layers)
done_set = set(done_layers)

# ---- calibration text: local file (PBR_TEXT) or WikiText-2 train via `datasets` ----
if os.environ.get("PBR_TEXT"):
    text = open(os.environ["PBR_TEXT"], encoding="utf-8").read()
else:
    from datasets import load_dataset
    # Legacy id "wikitext" is the same corpus as Salesforce/wikitext. Current
    # huggingface_hub rejects the un-namespaced id.
    try:
        _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(_wt["text"][:4000])
ids = tok(text, return_tensors="pt").input_ids
n_tok = SAMPLES * SEQ
if ids.shape[1] < n_tok:
    raise SystemExit(f"Calibration text too short: need {n_tok} tokens, have {ids.shape[1]}. "
                      f"Provide more text via PBR_TEXT or lower PBR_SAMPLES/PBR_SEQ.")
starts = np.random.default_rng(0).integers(0, ids.shape[1] - SEQ, SAMPLES)
batch = torch.cat([ids[:, s:s+SEQ] for s in starts], dim=0)   # (SAMPLES, SEQ)

class StopForward(Exception): pass

def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)

# PBR_ROT_PERLAYER=1 gives every (projection, layer, side) its OWN random rotation
# instead of reusing one cached rotation per width across all 24 layers. Still free
# to store (both sides regenerate from the seed). Motivation: with a shared rotation,
# each layer's quantization error has a correlated structure, so the errors it injects
# into the shared residual stream add up COHERENTLY (~N) instead of like an incoherent
# random walk (~sqrt(N)) -- see coherence_probe.py. Distinct per-layer rotations
# scramble the error directions so they cancel rather than reinforce.
rot_cache = {}
def rotation(n, key=None):
    if PERLAYER and key is not None:
        # stable, deterministic seed per (projection-name, layer, width, side) --
        # zlib.crc32, NOT Python's process-randomized hash(), so a decoder can
        # regenerate the identical rotation from the same key (the "free" property).
        import zlib
        seed = zlib.crc32(f"{key}|{n}".encode()) & 0x7fffffff
        return ortho(n, seed)
    if n not in rot_cache: rot_cache[n] = ortho(n, n)   # cached: same rotation reused for every layer of this width
    return rot_cache[n]

# Same chunk bound as push_below4.py (PBR_CHUNK). The post-fit assign below
# used to build one (rows, k) distance matrix — 544,768 x 256 float64 is
# 1.04 GiB and OOMs next to the live fp32 model.
CHUNK = int(os.environ.get("PBR_CHUNK", 50000))
def assign_rows(V, C, chunk=CHUNK):
    return np.concatenate([((V[i:i+chunk]**2).sum(1)[:, None] - 2*V[i:i+chunk]@C.T
                             + (C**2).sum(1)[None]).argmin(1) for i in range(0, len(V), chunk)])
def kmeans(V, k, it=10, chunk=CHUNK):
    rng = np.random.default_rng(0)
    C = V[rng.choice(len(V), k, replace=False)]
    for _ in range(it):
        a = assign_rows(V, C, chunk)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = V[m].mean(0)
    return C

def beam_assign(Rb, books, B):
    """Beam-search RVQ assignment for a D-wide block. Rb: (rows, D). Keep the B best
    cumulative stage-combinations per row (AQLM-style) instead of greedy-nearest.
    Same stored bits; measured +0.25..0.65 dB free on real tensors (beam_vq.py)."""
    n = Rb.shape[0]; recon = np.zeros((n, 1, Rb.shape[1]))
    for C in books:
        bb = recon.shape[1]; k = len(C)
        cand = recon[:, :, None, :] + C[None, None, :, :]          # (n, bb, k, D)
        d = ((Rb[:, None, None, :] - cand)**2).sum(-1).reshape(n, bb*k)
        keep = min(B, bb*k)
        idx = np.argpartition(d, keep-1, axis=1)[:, :keep]
        recon = np.take_along_axis(cand.reshape(n, bb*k, -1), idx[:, :, None], axis=1)
    fd = ((Rb[:, None, :] - recon)**2).sum(-1)
    return recon[np.arange(n), fd.argmin(1)]

def quantize_linear(Wt, H, stages, name="", li=0):
    """Wt: torch weight (out, in). H: numpy (in, in) uncentered covariance of its real inputs.
       Returns a torch tensor of the same shape, quantized then dequantized.
       Two passes, matching the design already validated in push_nested.py:
       (1) fit codebooks once, pooling vectors across the WHOLE tensor — this is what
           makes it work on narrow tensors (e.g. k_proj/v_proj, shrunk by grouped-query
           attention) where any one column-block alone would have too few rows to fit
           256-entry codebooks against;
       (2) a column-block pass that applies GPTQ-style error feedback using those
           already-fixed codebooks."""
    W = Wt.detach().double().numpy(); O, I = W.shape
    Ro, Ri = rotation(O, key=f"{name}:{li}:out"), rotation(I, key=f"{name}:{li}:in")
    Wr = Ro @ W @ Ri
    Hr = Ri.T @ H @ Ri
    Hr = Hr + 0.01 * np.mean(np.diag(Hr)) * np.eye(I)
    U = np.linalg.cholesky(np.linalg.inv(Hr)).T

    # PBR_INT4: strong-baseline mode -- same rotation + GPTQ feedback, but uniform
    # per-group int4 instead of VQ codebooks. This is the fair "strong 4-bit" ref.
    if os.environ.get("PBR_INT4") == "1":
        bits = int(os.environ.get("PBR_INT4_BITS", "4")); g = int(os.environ.get("PBR_INT4_G", "128"))
        L = 2**bits; T = Wr.copy(); Q = np.zeros_like(T); sc = None
        for j in range(I):
            if j % g == 0:
                sc = np.abs(T[:, j:j+g]).max(1) / ((L-1)/2) + 1e-12
            q = (np.clip(np.round(T[:, j]/sc - .5), -L/2, L/2-1) + .5) * sc
            Q[:, j] = q; e = (T[:, j] - q) / U[j, j]
            T[:, j+1:] -= np.outer(e, U[j, j+1:])
        return torch.from_numpy(Ro.T @ Q @ Ri.T).to(Wt.dtype)

    s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12

    pool = (Wr / s).reshape(-1, D)               # pass 1: fit codebooks on the whole tensor
    books, Rres = [], pool.copy()
    for k in stages:
        C = kmeans(Rres, k)
        a = assign_rows(Rres, C)
        books.append(C); Rres = Rres - C[a]

    BEAM = int(os.environ.get("PBR_BEAM", "1"))    # >1 enables beam-search assignment (free quality)
    T = Wr.copy(); Q = np.zeros_like(T)            # pass 2: column-block feedback
    for j in range(0, I, D):
        b = slice(j, j + D); R = T[:, b] / s
        if BEAM > 1:
            q = beam_assign(R, books, BEAM)        # keep top-B cumulative combos per row
        else:
            q = np.zeros_like(R); r = R.copy()
            for C in books:
                a = ((r**2).sum(1)[:, None] - 2*r@C.T + (C**2).sum(1)[None]).argmin(1)
                q += C[a]; r -= C[a]
        Q[:, b] = q * s
        if j + D < I:
            T[:, j+D:] -= (T[:, b] - Q[:, b]) @ np.linalg.solve(U[b, b], U[b, j+D:])
    Wq = Ro.T @ Q @ Ri.T
    return torch.from_numpy(Wq).to(Wt.dtype)

def index_bpw(stages):
    return sum(np.log2(k) for k in stages) / D

def fmt_bpw(x):
    # 2.75 stays 2.75; 2.875 stays 2.875 (a .2f format would print 2.88).
    return f"{x:.4f}".rstrip("0").rstrip(".")

total_params, quantized_params = 0, 0
t0 = time.time()
used_stages = {}
for li in which:
    stages = LAYER_STAGES.get(li, STAGES)
    used_stages[li] = stages
    block = layers[li]
    if li in done_set:
        n_restored = 0
        for name in TARGETS:
            try:
                mod = block.get_submodule(name)
            except AttributeError:
                continue
            n_restored += 1
            total_params += mod.weight.numel()
            quantized_params += mod.weight.numel()
        stage_note = ""
        if LAYER_STAGES:
            stage_note = f"  stages={stages}  index {fmt_bpw(index_bpw(stages))} bpw"
        print(f"layer {li:2d}/{n_layers-1}  resumed {n_restored} tensors  "
              f"[{time.time()-t0:6.1f}s elapsed]{stage_note}", flush=True)
        gc.collect()
        continue
    targets = {}
    for name in TARGETS:
        try: targets[name] = block.get_submodule(name)
        except AttributeError: pass   # architecture doesn't have this projection; skip it
    inputs = {name: [] for name in targets}
    hooks = []
    for name, mod in targets.items():
        hooks.append(mod.register_forward_hook(
            lambda m, inp, o, name=name: inputs[name].append(inp[0].detach().double())))
    stop = block.register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(StopForward()))
    with torch.no_grad():
        for i in range(0, SAMPLES, 8):
            try: model(batch[i:i+8])
            except StopForward: pass
    stop.remove()
    for h in hooks: h.remove()

    for name, mod in targets.items():
        X = torch.cat(inputs[name]).reshape(-1, mod.in_features).numpy()   # (tokens, in_features)
        inputs[name].clear()
        H = X.T @ X / len(X)
        del X
        Wq = quantize_linear(mod.weight, H, stages, name=name, li=li)
        with torch.no_grad(): mod.weight.copy_(Wq)
        total_params += mod.weight.numel(); quantized_params += mod.weight.numel()
    stage_note = ""
    if LAYER_STAGES:
        stage_note = f"  stages={stages}  index {fmt_bpw(index_bpw(stages))} bpw"
    print(f"layer {li:2d}/{n_layers-1}  quantized {len(targets)} tensors  "
          f"[{time.time()-t0:6.1f}s elapsed]{stage_note}", flush=True)
    save_layer_progress(li)
    gc.collect()

# Index cost only (codebook storage is shared and <0.1 bpw on tensors this large,
# larger on the narrow k/v projections). One number when every layer shares a recipe.
unique = {tuple(s) for s in used_stages.values()} if used_stages else {tuple(STAGES)}
if os.environ.get("PBR_INT4") == "1":
    _bits = int(os.environ.get("PBR_INT4_BITS", "4")); _g = int(os.environ.get("PBR_INT4_G", "128"))
    print(f"\nQuantized {quantized_params:,} parameters across {len(which)} layer(s) as INT{_bits} "
          f"(group {_g}) = {_bits + 16.0/_g:.3f} bpw. (codebook 'stages' are ignored in INT4 mode)")
elif len(unique) == 1:
    stages = list(next(iter(unique))) if used_stages else STAGES
    codebook_bpw = index_bpw(stages)
    print(f"\nQuantized {quantized_params:,} parameters across {len(which)} layer(s), "
          f"~{fmt_bpw(codebook_bpw)} bpw for touched tensors (stages={stages}, codebook overhead <0.1 bpw).")
else:
    print(f"\nQuantized {quantized_params:,} parameters across {len(which)} layer(s), mixed stages:")
    for li in which:
        s = used_stages[li]
        print(f"  layer {li}: stages={s}  index {fmt_bpw(index_bpw(s))} bpw")
    print("Codebook overhead is <0.1 bpw on wide tensors and larger on narrow k/v.")
print(f"Embeddings, output head, and norms were left untouched (full precision).")

os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"\nSaved new checkpoint to: {out}")
print(f"Load and test it directly:  python eval_ppl.py {out}")
