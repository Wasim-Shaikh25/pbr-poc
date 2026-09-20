# PBR Stage 1A / 1B proof of concept

Position-Based Binary Reconstruction (PBR) encodes BF16 weights as a
**generated / predicted bit pattern plus an exact XOR residual**. Reconstruction
is `Decode(Encode(W)) == W` on the original uint16 words.

This repository implements **Stage 1A** (controlled tensors), **Stage 1B**
(real public-checkpoint tensors), **Stage 2** (a qualification *scanner*),
and **PBR-E / Stage 2.5** (exponent-Huffman so typical models can hit the
published ~11 BPW lossless band). **PBR-E is DF11/ZipNN-class, not a novel
rate.** The distinctive bet is a Hierarchical / position program on top of
that; it has been implemented and, on the Qwen and Llama samples below, did
not win vs PBR-E. This repo is **not** a production model compressor and
**not evidence that an 8 GB checkpoint becomes 1–2 GB**.

> **Research status:** concept and early proof of concept. Stage 1A uses
> synthetic tensors. Stage 1B checks that the same codecs stay bit-exact on
> real BF16/F16 weights and reports complete-container BPW. Stage 2 projects
> whole-model BPW from sampled encodings. PBR-E adds exponent entropy coding
> and measured **10.87 complete-container BPW** (DF11/ZipNN-class, not novel)
> on the same Qwen 42-tensor set that Stage 1B encoded at 13.61. Hierarchical
> leftovers (bit-planes, residual grammar, transformed refs, position+value
> dicts) compete on complete encoded bytes and **won 0 / 42** Qwen tensors
> vs PBR-E. The same story holds on a Llama-3.2-1B BF16 sample (~10.84 BPW).
> None of these is a production ratio claim. ≤4 BPW remains out of scope for
> dense LLMs.

## What Stage 1A proves (Gate 1)

From the PoC implementation guide, **Gate 1 — codec correctness** passes when
every controlled case round-trips **bit-exactly**. If Gate 1 fails, stop and fix
the codec. Do not move on to real checkpoints.

Stage 1A specifically checks:

1. BF16 weights are handled as **uint16 views**. There is no FP32 archival path.
2. Encode → decode restores the original words (SHA-256 equality).
3. Several PBR-Direct predictors / residual modes are available:
   previous-value, previous-row, constant / block prototype, residual default +
   position list or bitmap, residual dictionary, exact value dictionary, exact
   duplicate-tile references, BF16 component split, and **raw BF16 fallback**.
4. The encoder picks a mode by **complete encoded byte cost** (tile header +
   metadata + payload), not by MSE.
5. Structured synthetic tensors compress when the matching codec is cheaper;
   a near-random control stays on raw fallback (bounded container overhead).
6. A CLI prints original bytes, encoded bytes, BPW, ratio vs raw BF16,
   exactness PASS/FAIL, and which modes won.
7. Optional zlib / zstd numbers are labeled **baseline (not PBR)**.

**Gate 2** (honest fallback) is covered by tests: random uint16 data must use
raw fallback or stay within bounded container overhead.

## What this is not

- Not a full-model encode of every shard, tokenizer, or config file.
- Not a fused tile inference runtime.
- Not a claim of 1–2 GB storage for an 8 GB model. Do not scale Stage 1B BPW
  or Stage 2 projections into that story.
- Hierarchical leftovers (bit-planes, residual-sequence grammar, transformed
  refs, position+value dicts) **are implemented** and compete on complete
  encoded bytes. On Qwen and Llama dense samples they do **not** beat PBR-E.
  That is an honest miss, not a 1–2 GB / 8 GB or ≤4 BPW claim.

## Install

```bash
python3 -m pip install -e ".[dev,stage1b]"
```

Runtime: `numpy`. Tests: `pytest`. Stage 1B download: `huggingface_hub`.
Optional baseline: `zstandard`.

## Run Stage 1A (one command)

From the repo root:

```bash
python scripts/run_poc1.py --self-test
```

This generates the controlled tensors, encodes each case, decodes it, checks
SHA-256 / uint16 equality, and prints a table:

```
case                 orig_B  enc_B   BPW     ratio   exact  winning_modes
constant_block       ...     ...     ...     ...     PASS   constant:...
random_uint16        ...     ...     ~16     ~1.0    PASS   raw_bf16:...
```

`GATE 1 PASS` means every case reconstructed bit-exactly.

Equivalent installable entry point:

```bash
pbr-poc1 --self-test
```

### Generate data and encode separately

```bash
python scripts/generate_controlled_data.py --case all --output-dir outputs/controlled
python scripts/run_poc1.py --input-dir outputs/controlled --output-dir outputs/reports/poc1
```

Reports written under `outputs/reports/poc1/`:

| file | contents |
| --- | --- |
| `summary.json` | complete-byte totals, BPW, mode usage |
| `console_report.txt` | the same table printed to the terminal |
| `<case>/encoded.pbr` | complete container (this file's size **is** the reported size) |
| `<case>/verification.json` | SHA-256 exactness record |
| `<case>/block_modes.csv` | tiles and bytes by winning mode |

Config defaults live in `configs/poc_controlled.yaml`.

## Stage 1B / Gate 1B — real BF16 tensors

**Gate 1B** asks: do the Stage 1A codecs still reconstruct **real** 16-bit
checkpoint weights bit-exactly, and what complete-container BPW / mode mix
do they produce? A PASS is exactness. A BPW near 16 on real weights is an
honest result, not a failure.

Default checkpoint: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
(Safetensors, BF16, ~988 MB). If that download fails, the runner falls back to
[`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
and prints that it did so.

The whole Qwen file is larger than the 100–500 MB Stage 1B window, so the
script selects a stratified sample of 2-D linear / attention `*.weight` tensors
from early, middle, and late layers (default 100–160 MiB). Embeddings and
norm vectors are skipped unless you pass `--include-embeddings`. The console
and `summary.json` list every selected tensor and byte count.

```bash
python3 -m pip install -e ".[dev,stage1b]"
python scripts/run_poc1b.py
```

Equivalent: `pbr-poc1b`. Useful flags:

```bash
python scripts/run_poc1b.py --model-dir /path/to/local/checkpoint
python scripts/run_poc1b.py --repo Qwen/Qwen2.5-0.5B-Instruct --revision main
python scripts/run_poc1b.py --max-bytes 167772160 --block-sizes 256
```

Weights are loaded from the Safetensors byte offsets as **uint16 views**.
BF16 and F16 are accepted. F32 / wider dtypes are refused (no FP32 archival
path). Reported `enc_B` is the complete PBR container size on disk.

Reports land in `outputs/reports/poc1b/`. Config: `configs/poc_real.yaml`.

### Measured Gate 1B run (this repo)

Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, Apache 2.0. Selected **42**
linear/attention weight tensors from layers 0, 1, 2, 3, 4, 5, 12, 23
(**159.9 MiB** of BF16). Embeddings were not included. **Every tensor
PASS** (uint16 / SHA-256). Winning mode on all tiles: `bf16_components`.

| tensor | orig B | enc B | BPW | ratio | exact | zlib B (baseline) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| layers.0.mlp.down_proj.weight | 8716288 | 7408946 | 13.60 | 0.850 | PASS | 6933245 |
| layers.12.self_attn.q_proj.weight | 1605632 | 1367170 | 13.62 | 0.851 | PASS | 1278723 |
| layers.23.mlp.up_proj.weight | 8716288 | 7403422 | 13.59 | 0.849 | PASS | 6936780 |
| **TOTAL (42 tensors)** | **167673856** | **142577035** | **13.61** | **0.850** | **PASS** | **133488038** |

zlib is a **baseline, not PBR**. On this sample zlib is slightly smaller
(~12.7 BPW vs PBR 13.6). That is expected: Stage 1B proves exactness and
complete-byte accounting on real weights, not that PBR beats general
compressors. **Do not scale 13.6 BPW into a 1–2 GB / 8 GB claim.**

### Model attribution and license

Stage 1B may download **Qwen2.5-0.5B-Instruct**, © 2024 Alibaba Cloud,
[Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE).
The fallback **SmolLM2-360M-Instruct** is also Apache 2.0 (Hugging Face TB).
This repository does **not** redistribute those weights. Record the resolved
commit SHA from the run output before comparing results.

## Stage 2 — model qualification scanner

Stage 2 asks whether a **full encode is worth running**, using the same
Stage 1 codecs on **samples**. It does not produce a measured checkpoint
size.

```
python scripts/run_qualifier.py --model Qwen/Qwen2.5-0.5B-Instruct
# or
pbr-qualify --model-dir /path/to/local/checkpoint
```

Pipeline:

1. **Inventory** every Safetensors tensor (name, shape, dtype, nbytes).
2. **Entropy triage** on uint16 words, BF16 components, and previous-value /
   previous-row residuals (subsampled on large tensors).
3. **Predictor / mode screening** by running the existing cost-based encoder
   on a strided tile sample (or the whole tensor if it is tiny).
4. **Repetition** — exact-duplicate rate and mean XOR popcount vs the
   previous sampled tile.
5. **Actual sample encode** — complete container bytes (headers, mode ids,
   dictionaries, residuals). Each sample is decoded and must be bit-exact.
6. **Whole-model projection**
   `Total_BPW = 8 × total_projected_encoded_bytes / total_16bit_parameter_count`.
   Tile on-wire bytes are scaled to the full tile count; the container header
   is counted once per tensor. **Unscanned 16-bit remainder is raw BF16 at
   16 BPW.**

### Dtype policy

Only `BF16` / `F16` / `FP16` are scanned, as uint16 views. Other dtypes are
**not converted**. They are listed, their raw bytes are reported separately,
and they are excluded from the 16-bit BPW denominator.

### How to read the bands

| projected BPW | Band | What Stage 2 may say |
| --- | --- | --- |
| >8 | not exceptional | Not a PBR-Direct high-potential target |
| 4–8 | strong | Worth a closer look; not high-potential yet |
| 2–4 | exceptional | **High-potential** (projection ≤4 BPW) |
| ≤2 | one-GB *class* | Class label only |
| ≤1 | extreme | Class label only |

**High-potential** is allowed only when the projected complete BPW is **≤4**.
**One-GB qualified** is **never** set by Stage 2. That label requires a later
measured full-container encode at ≤2 BPW.

### Relation to the Stage 1B result

Stage 1B fully encoded 159.9 MiB of Qwen linear/attention weights at
**13.61 complete-container BPW**, with `bf16_components` winning and zlib
slightly smaller. That already says this checkpoint, under *these* codecs,
looks like ordinary lossless data. Stage 2 should land in the same
**not exceptional** band unless hierarchical modes (not implemented here)
change the sample encodes. A projection near 13–16 BPW is a consistency
check, not a new compression claim.

Config: `configs/qualifier_default.yaml`. Reports:
`outputs/reports/poc2/stage2_qwen05b.json` and `console_report.txt`.

## PBR-E / adapted method (Stage 2.5)

Stage 1B/2 on Qwen landed at **~12.7–13.6 BPW** with `bf16_components`.
That underperformed the published lossless bar because exponents were stored
as a raw 8-bit stream (or a tiny per-tile palette), not entropy-coded.

Consensus from recent lossless work:

- [ZipNN](https://arxiv.org/abs/2411.05239) — BF16 compressibility is mostly
  in the exponent field.
- [DFloat11 / DF11](https://arxiv.org/abs/2504.11651) (NeurIPS 2025) —
  ~11 BPW bit-exact on typical LLMs (~30% size cut).
- [NeuZip](https://arxiv.org/abs/2410.20650) — ANS/Huffman on exponents;
  mantissa often left raw for the lossless setting.

**PBR-E** keeps the exact XOR / uint16 contract and adds `bf16_exp_huffman`:

1. Split each BF16 word: `sign (1) | exponent (8) | mantissa (7)`.
2. Canonical-Huffman the exponent stream (optional previous-exponent XOR if
   that codebook+stream is smaller).
3. Pack `sign+mantissa` as 8 raw bits/weight.
4. Count the codebook in complete container bytes.
5. Compete with every existing PBR mode via `argmin(complete_encoded_bytes)`.
6. Also try **one codebook for the whole tensor** so the table amortizes
   (per-tile Huffman alone cannot reach ~11 BPW).

Realistic target for most dense LLMs: **~11 BPW / ~30%**, bit-exact.
**≤4 BPW is not the goal** on this class of models.

```bash
python scripts/run_pbre.py --model-dir /path/to/Qwen2.5-0.5B-Instruct
```

Uses the same pinned Qwen revision and the same ~160 MiB linear/attention
tensor set as Stage 1B so BPW is comparable to 13.61. Reports land in
`outputs/reports/pbre/`. The table includes zlib and an
`expHuff_bound_B` column (DF11-style bound, labeled baseline, not PBR).

### Measured PBR-E run (this repo)

Same checkpoint and **the same 42-tensor / 159.9 MiB** Stage 1B selection
(layers 0, 1, 2, 3, 4, 5, 12, 23 linear/attention weights; no embeddings).
Revision `7ae557604adf67be50417f59c2c2f167def9a775`. **Every tensor PASS**
(uint16 / SHA-256). Winning mode on **42 / 42** tensors (100% of tiles):
`bf16_exp_huffman` with one amortized codebook per tensor.

| tensor | orig B | enc B | BPW | ratio | exact | zlib B (baseline) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| layers.0.mlp.down_proj.weight | 8716288 | 5892011 | 10.82 | 0.676 | PASS | 6933245 |
| layers.12.self_attn.q_proj.weight | 1605632 | 1103238 | 10.99 | 0.687 | PASS | 1278723 |
| layers.23.mlp.up_proj.weight | 8716288 | 5926926 | 10.88 | 0.680 | PASS | 6936780 |
| **TOTAL (42 tensors)** | **167673856** | **113900400** | **10.87** | **0.679** | **PASS** | **133488038** |

| labeled baseline (not PBR) | bytes | BPW |
| --- | ---: | ---: |
| raw BF16 | 167673856 | 16.00 |
| zlib (level 9) | 133488038 | 12.74 |
| exponent-Huffman bound (DF11-style) | 111216676 | 10.61 |
| **PBR-E complete container** | **113900400** | **10.87** |
| Stage 1B PBR (`bf16_components`) | 142577035 | 13.61 |
| Stage 2 sample projection (pre-PBR-E) | 783604658 projected | 12.69 |

PBR-E **beats Stage 1B (13.61 → 10.87 BPW)** and **beats zlib** on this
typical Qwen sample. The complete-container rate sits **0.26 BPW above**
the information-theoretic exponent-Huffman bound (codebook + 8-bit
sign/mantissa + entropy-coded exponents). That is the DF11-class band
(~11 BPW / ~32% size cut). **≤4 BPW is still out of scope** for dense
LLMs; this run does not claim 1–2 GB / 8 GB.

## Blockers & mitigations

Invented tile / position / block methods lose on Qwen when they predict on
the **full uint16**. The mantissa is near-random (~7 bits), so XOR residuals
stay huge, exact duplicate tiles are rare, and per-tile dictionaries lose
to header + codebook overhead.

```bash
python scripts/run_blocker_diagnosis.py --model-dir /path/to/Qwen2.5-0.5B-Instruct
python scripts/run_blocker_ablation.py --model-dir /path/to/Qwen2.5-0.5B-Instruct
```

Reports: `artifacts/blocker_diagnosis_qwen.json` (plus `.md`) and
`artifacts/blocker_ablation_qwen.{json,md}`.

Mitigations (still lossless, still `Decode(Encode(W))==W`):

| mode | what it does |
| --- | --- |
| `exp_spatial_huffman` | prev / prev_row / block-prototype on the **exponent byte only**, then Huffman; sign+mantissa packed raw |
| `exp_hier_residual` | block-default exponent, then Huffman the local residual (optional residual-of-residual) |
| `cross_layer_tile_xor` | XOR vs a previous same-role / same-shaped tensor; Huffman the exponent residual or sparse uint16 patch. Ref bytes are not stored again; decode is causal |

Existing `bf16_exp_huffman` and the old uint16 spatial modes stay in the
menu. Ablation profiles: PBR-E only, new modes on, forced uint16-spatial
only, zlib baseline. Success for this run is **not** ≤4 BPW — it is a
clear diagnosis plus an honest win or miss for spatial-on-exponents vs
plain PBR-E (~10.87).

### Measured blocker diagnosis (same 42 tensors)

Weighted entropies on 83,836,928 BF16 words. Checkpoint has 290 16-bit
tensors; this table is the Stage 1B sample.

| field | bits |
| --- | ---: |
| H(uint16) | 10.54 |
| H(exponent) | **2.61** |
| H(mantissa) | **6.97** |
| H(sign) | 1.00 |
| H(uint16 prev_value residual) | 11.11 |
| H(uint16 prev_row residual) | 11.12 |
| H(exponent prev_value residual) | 3.12 |
| H(exponent prev_row residual) | 3.13 |

Spatial-on-uint16 residuals are **worse** than storing the words. Spatial-on-exponent residuals are **worse** than raw exponents. Exact duplicate tile rate is **0** at tile sizes 64 / 256 / 1024, including same-role tiles across layers. Ideal (no-codebook) uint16 residuals beat raw 16-bit storage, but **0%** of those tiles beat raw once tile headers + codebooks are counted.

### Measured blocker ablation (same 42 tensors, all exact PASS)

| profile | enc B | BPW | ratio | winning modes |
| --- | ---: | ---: | ---: | --- |
| PBR-E only | 113,900,442 | **10.87** | 0.679 | `bf16_exp_huffman` 42/42 |
| New modes (A/B/C enabled) | 113,900,652 | **10.87** | 0.679 | `bf16_exp_huffman` 42/42 |
| Forced uint16 spatial | 176,866,107 | **16.88** | 1.055 | `raw_bf16` 327,488/327,488 tiles |
| zlib (baseline, not PBR) | 133,488,038 | 12.74 | 0.796 | — |

**Honest negative:** spatial / hierarchical / cross-layer predictors on
exponents did **not** beat plain `bf16_exp_huffman` on this Qwen sample.
They stay in the menu and remain bit-exact; the winner is still PBR-E at
~10.87 BPW. Forced spatial-on-uint16 documents the original blocker
(raw fallback, worse than uncompressed because of tile headers). ≤4 BPW
is still out of scope. Not a 1–2 GB / 8 GB claim.

## Hierarchical leftovers (Option B)

PBR-E is the ZipNN / DFloat11 idea: Huffman (or ANS) the exponent byte,
store sign+mantissa packed. That is **not the novel claim**. The distinctive
bet is a **position / hierarchical program** that would have to beat PBR-E
on `complete_encoded_bytes` while staying bit-exact.

Four leftover modes now sit in the same menu as `bf16_exp_huffman` and the
spatial-on-exponent codecs (`--profiles hierarchical`):

| mode | what it does |
| --- | --- |
| `bit_planes` | 16 planes per tile; all-zero / all-one / sparse-patch / raw; rebuilds exact uint16 |
| `residual_grammar` | repeated prev-value residual ID phrases (len 4); admitted only if savings > dict cost |
| `transformed_ref` | exact or exact-after cheap transform (identity / transpose if square / sign-bit XOR / byteswap) plus XOR patch |
| `position_value_dict` | default word + small exception palette + positions; positive-savings admission |

```bash
python scripts/run_blocker_ablation.py --profiles hierarchical --tag qwen_hier
```

### Measured hierarchical ablation (same Qwen 42 tensors)

Same checkpoint and **the same 42-tensor / 159.9 MiB** Stage 1B selection.
Revision `7ae557604adf67be50417f59c2c2f167def9a775`. **Every tensor PASS.**
Winning mode on **42 / 42** tensors: `bf16_exp_huffman`.

| profile | orig B | enc B | BPW | ratio | exact | winning modes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| hierarchical (leftovers + PBR-E) | 167673856 | 113900778 | **10.87** | 0.679 | PASS | `bf16_exp_huffman` 42/42 |
| PBR-E only (same set) | 167673856 | 113900400 | **10.87** | 0.679 | PASS | `bf16_exp_huffman` 42/42 |

**Honest negative:** no leftover hierarchical mode won any tensor vs PBR-E
~10.87 BPW. The extra menu costs a few hundred bytes of search overhead in
the container (113900778 vs 113900400). Bit-exact still holds. Not ≤4 BPW.

Reports: `artifacts/blocker_ablation_qwen_hier.{json,md}`.

## Llama-family check (not MoE)

`meta-llama/Llama-3.2-1B-Instruct` is gated here (`GatedRepoError` 401 on
`config.json`). Used the public BF16 mirror
[`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)
revision **`5a8abab4a5d6f164389b1079fb721cfab8d7126c`**. Inventory: **146**
16-bit tensors. Sampled **11** linear/attention weights from layers 0 and 8
(**167,772,160 B** / 83,886,080 words) with the same 100–160 MiB window as
Stage 1B. Config: `configs/poc_llama.yaml`.

### Diagnosis snapshot

| field | bits |
| --- | ---: |
| H(uint16) | 10.51 |
| H(exponent) | **2.60** |
| H(mantissa) | **6.97** |
| H(sign) | 1.00 |
| H(uint16 prev_value residual) | 11.10 |
| exact dup tile rate (64 / 256 / 1024) | **0** |

Same blocker story as Qwen: mantissa ≈ 7 bits, spatial residuals raise
entropy, zero exact tile duplicates.

### PBR-E + hierarchical + zlib (all exact PASS)

| profile | orig B | enc B | BPW | ratio | exact | winning modes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| PBR-E (`bf16_exp_huffman`) | 167772160 | 113694055 | **10.84** | 0.678 | PASS | `bf16_exp_huffman` 11/11 |
| hierarchical leftovers on | 167772160 | 113694143 | **10.84** | 0.678 | PASS | `bf16_exp_huffman` 11/11 |
| zlib (baseline, not PBR) | 167772160 | 133302428 | 12.71 | 0.795 | n/a | zlib |

**Honest negative:** leftover hierarchical modes won **0 / 11** Llama
tensors vs PBR-E. PBR-E beats zlib (10.84 vs 12.71 BPW) in the DF11-class
band. Not ≤4 BPW. Not a 1–2 GB / 8 GB claim.

Reports: `artifacts/blocker_diagnosis_llama.{json,md}`,
`artifacts/blocker_ablation_llama.{json,md}`.

## New directions (PBR-E / PBR-M / Extreme)

Stop treating hierarchical leftovers / MoE as the main path. Three tracks:

1. **PBR-E Practical** — full bit-exact checkpoint codec in the DF11/ZipNN
   class (~11 BPW on typical dense LLMs). Not novel; it is the productized
   lossless bar.
2. **PBR-M Research** — reduce dense *mantissa* **conditional** entropy.
   Open question. Every predictor / table / transform byte counts.
3. **Extreme PBR** — ≤4 BPW only if structure actually appears; falsifiable.
   Do not claim it on these dense Qwen/Llama samples.

**Phase A gate (mantissa):** a compact context or reversible transform must
code held-out mantissas at **< 6.5 complete BPW after overhead**
(≈ <10.1 total with ~1 sign + ~2.6 exp). Anything that does not beat
unconditional H(M) including tables is rejected.

```bash
python scripts/run_phase_a_mantissa_audit.py --tag qwen
```

Reports: `artifacts/phase_a_mantissa_audit_qwen.{json,md}`.

### Measured Stage 2 run (this repo)

Same Qwen revision as Stage 1B
(`7ae557604adf67be50417f59c2c2f167def9a775`). **290 / 290** 16-bit tensors
scanned (494,032,768 parameters). Every sample encode/decode **PASS**.
Zero non-16-bit tensors. Zero unscanned 16-bit remainder.

| | value |
| --- | --- |
| Projected complete BPW | **12.69** |
| Band | **not exceptional** (>8) |
| High-potential (≤4 projected) | **NO** |
| One-GB qualified | **NO** (Stage 2 cannot award this) |
| Projected encoded bytes | 783,604,658 |
| Dominant sample modes | `bf16_components` on large weights; raw/header-heavy on tiny biases/norms |

Best projected large tensor: `model.embed_tokens.weight` at 11.70 BPW.
Layer linear/attention weights cluster around **13.0–13.2 BPW**, which
matches the Stage 1B full-encode measurement of **13.61 BPW** on a 160 MiB
linear subset (sample projection is slightly optimistic). Tiny 1-D norms and
biases show 35–47 BPW because complete container headers dominate a few
hundred words; they barely move the weighted total.

**This model is not high-potential under the pre-PBR-E Direct codecs**
(the ≤4 BPW gate). PBR-E later measured **10.87 BPW** on the Stage 1B
subset — DF11-class, still far from ≤4. A future hierarchical scanner
could revise the Stage 2 projection; this Stage 2 run did not include
exponent Huffman.

## Tests

```bash
pytest
```

The suite fails loudly (`ExactnessError: EXACTNESS FAIL ...`) if any uint16 word
differs. Cases include special BF16 bit patterns (signed zero, Inf, NaN
payloads, subnormals) that an FP32 detour would be likely to destroy.

Stage 1B / Stage 2 / PBR-E / blocker-mitigation / hierarchical / Phase A
unit tests write tiny local Safetensors fixtures.
They do **not** download the 988 MB checkpoint. Live download tests are
skipped unless `PBR_LIVE_HF=1`.

## How size is counted

```
BPW = 8 × complete_container_bytes / word_count
ratio = complete_container_bytes / (2 × word_count)
```

The complete container includes magic/version, JSON tensor metadata, per-tile
headers, dictionaries, residuals, references, and checksums. A raw uint16 dump
of the weights is **not** the reported compressed size. The number printed as
`enc_B` is `len(encoded.pbr)` on disk.

Raw BF16 is always a candidate, so encode cannot fail correctness by forcing a
broken codec. Random / incompressible tiles stay on `raw_bf16`.

## Reconstruction rule

For every predicted BF16 word `P` and original word `W`:

```
R = W XOR P
W = P XOR R
```

Stage 1A predictors:

| mode | prediction |
| --- | --- |
| `prev_value` | previous decoded word in raster order (`P[0] = 0`) |
| `prev_row` | same column, previous row (`P[0, :] = 0`) |
| `const_pred` | most frequent exact word in the tile |
| `constant` | one stored uint16, no residual stream |
| `raw_bf16` | stored words, no prediction |

Residuals are then packed as constant, default+list, default+bitmap, a small
exact dictionary, or raw 16-bit values — whichever complete encoding is
shortest.

## Layout

```
pbr_core/        uint16 views, tiles, container, hashing, Safetensors I/O
pbr_codecs/      raw, predictors, residuals, dictionaries, exp-Huffman,
                 bit-planes, grammar, transformed refs, position-value dict
pbr_encoder/     cost-based search, decoder, Stage 1A/1B/PBR-E/Phase A CLIs
pbr_qualifier/   Stage 2 inventory, entropy, sample encode, BPW projection
scripts/         run_poc1.py, run_poc1b.py, run_qualifier.py, run_pbre.py,
                 run_blocker_diagnosis.py, run_blocker_ablation.py,
                 run_family_eval.py, run_phase_a_mantissa_audit.py
tests/           exactness, codecs, Stage 1B fixtures, qualifier math,
                 hierarchical leftovers, mantissa audit
configs/         poc_controlled.yaml, poc_real.yaml, poc_llama.yaml,
                 qualifier_default.yaml
artifacts/       measured diagnosis / ablation / Phase A reports
```

Later stages (full selected-model encode vs projection, fused runtime) are
intentionally absent.

## Interpreting numbers

A low BPW on `constant_block` or `previous_row` only shows that the codec
recognizes the pattern it was given. The `random_uint16` row is the honesty
check: PBR must not invent compression on unstructured bits.

Stage 1B BPW (13.61, `bf16_components`) and PBR-E BPW (10.87,
`bf16_exp_huffman`) on real Qwen tensors are measurements of **this encoder
on those tensors**, including headers. Llama-3.2-1B (unsloth BF16 mirror)
measured **10.84 BPW** on an 11-tensor sample. Stage 2 BPW is a **sample
projection** for the whole 16-bit parameter set (pre-PBR-E codecs). None
of these is a measured 8 GB-model result. PBR-E ≈ DF11/ZipNN-class; the
hierarchical/position program is the distinctive bet and has not won on
these dense samples.

zlib / zstd columns, when present, are **general-purpose baselines**, not PBR
modes. Forced uint16-spatial rows in the blocker ablation exist to show
that predictor on mixed fields still loses; they are not a compression claim.

## License

This PoC is MIT. Research drafts that specify it remain separate documents.
Third-party checkpoints used in Stage 1B keep their own licenses (Qwen2.5
Instruct: Apache 2.0; Llama-3.2-1B-Instruct via the unsloth BF16 mirror:
see that model card / Llama 3.2 license).
