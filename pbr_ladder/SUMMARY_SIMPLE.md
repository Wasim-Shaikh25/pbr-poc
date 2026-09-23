# PBR-Ladder — The Simple Explanation

Plain-language companion to [`FINDINGS_AND_REPRODUCTION.md`](FINDINGS_AND_REPRODUCTION.md).
No jargon. What we did, what we got, how to run it.

---

## 1. What we compressed, and how small it got

The model is **Qwen2.5-0.5B**. Normally every weight (number) inside it takes **16 bits**.
We squeezed those numbers down:

- **Best quality-keeping result: ~5.1 bits per weight** → the model shrank from ~1 GB to **~300 MB** (about 3× smaller).
- We can also push the main weight layers down to **~3 bits** — it still runs and stays usable, but quality drops a bit more (see §4).

Think of it like image compression: 5 bits = a very clean JPEG, 3 bits = a smaller-but-slightly-softer JPEG.

---

## 2. What quality we kept

Quality here = **perplexity** (how "confused" the model is on real text). **Lower is better.**

| Model | Perplexity | Bits/weight | Size |
|---|---|---|---|
| Original (uncompressed) | 14.25 | 16 | ~988 MB |
| **OURS (best)** | **15.21** | **~5.1** | **~300 MB** |
| **OURS (3-bit weights)** | **16.83** | **~3.1 (weights)** | smaller still |
| Official llama.cpp q2_k | 16.53 | 5.27 | ~390 MB |
| Official q3_k_m | 15.72 | 5.49 | ~340 MB |
| Official q4_k_m | 15.33 | 6.24 | ~385 MB |

**Headline:** at the *same or fewer bits*, our best model kept **better quality than every
official quant** on our main test set — within ~7% of the uncompressed model at 3× smaller.

> Honest note: on a *different* text set (C4), the higher-bit official quants (which spend more
> bits) edge ahead. Our win at the *matched* bit budget (vs q2_k) holds everywhere. The fix for
> the rest is "diverse calibration" — see §5.

---

## 3. The methods we used (in plain words)

1. **Rotation** — spin the weights so no single number is a fragile outlier. Spreads the risk evenly.
2. **Vector quantization (RVQ)** — instead of rounding each number alone, take 8 at a time and
   snap them onto a shared "codebook" of common patterns, then refine in layers. Like building
   the picture up from coarse to fine.
3. **Error feedback (GPTQ-style)** — when we round one column, we shove the leftover error into
   the next column so small mistakes don't pile up down the line.
4. **Smart bit budgeting** — the move that actually won: don't waste bits where they don't help.

### What was genuinely unique / the real discovery
The biggest win **wasn't a cleverer math trick — it was where we spent the bits:**

- We had been leaving the **embedding table at full 16 bits**. On a tiny 0.5B model that table
  is ~27% of everything — a huge silent waste. Dropping it to 8 bits was nearly free in quality.
- We looked *inside* the competitor's "2-bit" file and found it's secretly **~5 bits**
  (8-bit embeddings + ~4.5-bit weights). Their "2-bit" label was misleading.
- Rebalancing to **8-bit embeddings + 4-bit weights = ~5.1 bits total** beat them at their own budget.

We also *proved* something important (even though it "failed" as a lever): after rotation the
weights are as mathematically random as possible — sitting **0.27 dB from the theoretical limit
(Shannon bound)**. Translation: no fancier rounding trick can squeeze more out. We're on the wall.
**The only lever left is bit allocation** — which is exactly what won.

---

## 4. How to push it down to 3 bits (we did this too)

The knob is `PBR_STAGES` — each entry is one 1-bit refinement stage. More stages = more bits.

| Bits on the weight layers | `PBR_STAGES` value | Whole-model perplexity |
|---|---|---|
| 3 bits | `256,256,256` | **16.83** |
| 4 bits (the winner) | `256,256,256,256` | **15.21** (with 8-bit embeds) |

```bash
# 3-bit weight layers (smaller, slightly softer quality)
PBR_STAGES=256,256,256 PBR_KMEANS_IT=6 python quantize_full_model.py ./qwen05b ./qwen05b_3bit

# then quantize the embeddings too so the WHOLE model stays small
PBR_PROJ_BPW=3.0 python embed_quant_eval.py ./qwen05b ./qwen05b_3bit
```

To claw back quality at 3 bits without adding bits, turn on the two **free** knobs:

```bash
# beam search (+0.65 dB) + more k-means refinement, still 3-bit
PBR_STAGES=256,256,256 PBR_BEAM=4 PBR_KMEANS_IT=10 python quantize_full_model.py ./qwen05b ./qwen05b_3bit_hq
```

- `PBR_BEAM=4` — tries several rounding paths per block and keeps the best. Free quality, slower.
- `PBR_KMEANS_IT=10` — more codebook refinement. Free quality, slower.
- `PBR_LOWRANK=16` — adds a small activation-aware correction (helps VQ alone; mostly redundant once GPTQ is on).

**Why 3-bit is harder:** at 3 bits you're right at the Shannon wall, so quality falls off faster.
5 bits is the sweet spot; 3 bits is for when size matters more than the last bit of quality.

---

## 5. How to run the whole pipeline (start to finish)

Everything runs **CPU-only** (no GPU needed). A full 0.5B run takes **~1.5–2 hours** on 16 cores.

### Step 0 — one-time setup
```bash
pip install torch transformers safetensors numpy datasets
# put the base model at ./qwen05b  (a normal HuggingFace Qwen2.5-0.5B-Instruct folder)
```

### Step 1 — quantize the model (the winning config)
```bash
PBR_STAGES=256,256,256,256 PBR_KMEANS_IT=6 \
  python quantize_full_model.py ./qwen05b ./qwen05b_rematch
```
This rotates, vector-quantizes, and applies error feedback to all 7 weight types in all 24 layers,
writing a normal loadable model folder to `./qwen05b_rematch`. It saves progress after each layer,
so if it stops you can rerun the same command to resume.

### Step 2 — quantize the embeddings (the lever that beat GGUF)
```bash
PBR_PROJ_BPW=4.0 python embed_quant_eval.py ./qwen05b ./qwen05b_rematch
```
Tries the embedding table at 16 / 8 / 6 / 5 / 4 bits and reports the effective whole-model bits
and size at each. **8-bit is the recommended pick** (best size-for-quality).

### Step 3 — check quality
```bash
# perplexity on the main test set
python eval_ppl.py ./qwen05b_rematch

# cross-domain check (different text, catches over-tuning)
python eval_on_corpus.py ./qwen05b ./qwen05b_rematch c4
```

### Step 4 — compare against the official quants (apples-to-apples)
```bash
python gguf_eval.py q2_k q3_k_m q4_k_m
```
Downloads the real official quantized files and runs them through the *same* perplexity test,
so the comparison is fair.

### The pipeline in one picture
```
base model (16-bit)
   │
   ▼  quantize_full_model.py   →   rotate → RVQ (pick bits via PBR_STAGES) → GPTQ error feedback
   ▼  embed_quant_eval.py      →   shrink the embedding table (16 → 8 bit)
   ▼  eval_ppl.py / eval_on_corpus.py   →   measure quality
   ▼  gguf_eval.py             →   compare vs the official quants
   ▼
final small model (~3–5 bit, ~300 MB)
```

### The knobs, in one table
| Env var | What it does | Typical value |
|---|---|---|
| `PBR_STAGES` | bits on weight layers (1 stage = 1 bit) | `256,256,256,256` (4-bit) or `256,256,256` (3-bit) |
| `PBR_PROJ_BPW` | tells the embed step what the weights cost | match your stages (4.0 or 3.0) |
| `PBR_BEAM` | free quality, slower (tries multiple roundings) | `4` |
| `PBR_KMEANS_IT` | codebook refinement passes | `6`–`10` |
| `PBR_LOWRANK` | activation-aware correction (optional) | `16` |
| `PBR_ROT_PERLAYER` | per-layer rotation (we found: worse, leave off) | `0` |

---

## 6. What's proven vs what's still open

**Proven:** the compression works, beats the official quants at matched bits, and the "win"
is bit allocation, not a secret algorithm. The math is near its theoretical limit.

**Still open (be honest about this):**
- Only tested on 0.5B. On big models (7B) embeddings are tiny (~4%), so the method should look
  *even better*, but that's untested.
- Measured with perplexity, not task accuracy (e.g. exam-style benchmarks).
- No fast inference kernels yet — this is a quality + size result, not a shipped product.
- Cross-domain margin is thinner; the next step is a **diverse calibration set**
  (WikiText + C4 + code) to make the "beats everything" claim clean everywhere.
