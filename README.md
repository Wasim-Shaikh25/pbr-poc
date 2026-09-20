# PBR Stage 1A / 1B proof of concept

Position-Based Binary Reconstruction (PBR) encodes BF16 weights as a
**generated / predicted bit pattern plus an exact XOR residual**. Reconstruction
is `Decode(Encode(W)) == W` on the original uint16 words.

This repository implements **Stage 1A** (controlled tensors), **Stage 1B**
(real public-checkpoint tensors), **Stage 2** (a qualification *scanner*),
and **PBR-E / Stage 2.5** (exponent entropy coding so typical models hit the
published ~11 BPW lossless band). Hierarchical / position programs on top of
it did not win vs PBR-E on the Qwen and Llama samples below. This is **not
evidence that an 8 GB checkpoint becomes 1–2 GB or 4 GB**.

> **Research status:** concept and early proof of concept. Stage 1A uses
> synthetic tensors. Stage 1B checks that the same codecs stay bit-exact on
> real BF16/F16 weights and reports complete-container BPW. Stage 2 projects
> whole-model BPW from sampled encodings. PBR-E adds exponent entropy coding
> and measured **10.62 complete-container BPW (rANS)** / **10.87 Huffman**
> (DF11/ZipNN-class, not novel) on the Stage 1B Qwen 42-tensor set.
> Hierarchical leftovers and **PBR-4 (experimental, negative, 20.58 BPW)**
> do not beat PBR-E. None of these is a production ratio claim. ≤4 BPW
> remains out of scope for dense LLMs.

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

Reports: `artifacts/phase_a_mantissa_audit_qwen.{json,md}` (and
`artifacts/phase_a_mantissa_audit_llama.{json,md}` for a Llama-3.2-1B lite
snapshot).

### Measured Phase A (Qwen Stage 1B 42 tensors)

Same 42-tensor / 167,673,856 B Qwen sample. Held-out last 20% of rows.
Tables counted. Gate: mantissa complete BPW **< 6.5**.

| method | ideal BPW | complete BPW | MI vs H(M) | beats 6.5? |
| --- | ---: | ---: | ---: | --- |
| H(M\|exp) | 6.933 | **6.955** | 0.039 | no |
| H(M\|sign,exp) | 6.938 | 6.962 | 0.034 | no |
| H(M) uncond | 6.972 | 6.977 | 0 | no |
| prev M / prev row / prev layer | ~6.98 | 6.977 | ≤0 | no |
| gray / mod-delta / bit-plane / Haar | ≥6.97 | ≥6.977 | ≤0 | no |

H(sign)=1.00, H(exp)=2.61, H(M)=6.97. Best implied total ≈ **10.57 BPW**.
**Gate MISS.** Spatial / combined contexts and cheap reversible transforms
are rejected (they do not beat unconditional H(M) after tables). Not ≤4 BPW.

Llama-3.2-1B lite (11 tensors, 167,772,160 B): best `H(M|exp)` complete
**6.932** mantissa BPW, implied total ≈ 10.53, same **MISS**.

Leftover 2-D tensors (embedding + layer-10 MLP, **298,418,176 B**):
best `H(M|exp)` **6.944**, same **MISS**. Embeddings are not a hidden
easy set.

### PBR-M mantissa model zoo

Pressure-test CTW/PPM bit contexts, histogram GBDT, tiny causal AR MLPs
(actual stored size 8.8 KB / 26 KB, under 64KB/256KB budgets), IDF-lite
integer coupling, and a per-tensor mixture against PBR-E. Same Qwen
Stage 1B 42-tensor set. Held-out last 20% of rows. Complete BPW counts
every model byte. Numpy only (sklearn/torch were not in the env).

```bash
python scripts/run_mantissa_zoo.py --tag qwen
```

#### Measured bakeoff (this repo)

83,836,928 words, holdout 16,767,104. Slice exactness **PASS** (uncond
rANS, bit-Markov, GBDT, AR, IDF). Wall 258 s. Python bit-rANS slice
~1.1k words/s (not a production speed claim).

| method | complete mant BPW | total BPW | vs PBR-E split | gate 6.5 |
| --- | ---: | ---: | ---: | --- |
| mixture (exp-cond + GBDT) | **6.938** | **10.585** | −0.062 | no |
| H(M\|exp) tables | 6.939 | 10.586 | −0.061 | no |
| GBDT 16×depth-2 | 6.961 | 10.608 | −0.039 | no |
| uncond rANS M | 6.972 | 10.619 | −0.028 | no |
| CTW / bit-Markov / PPM | ≥6.973 | ≥10.620 | ≥ −0.027 | no |
| IDF-lite | 6.988 | 10.635 | −0.012 | no |
| tiny AR h32 / h96 | ≥6.994 | ≥10.641 | ≥ −0.006 | no |
| raw M (PBR-E mantissa) | 7.000 | 10.647 | 0 | no |

**Phase A gate MISS. Stretch ≤4 BPW total: no.** The 0.06 BPW total trim
is entropy-coding the mantissa (DF11-class, same move as exponent rANS),
not a new principle. CTW, AR, and IDF did not beat H(M|exp). Bits-back /
sign-fold was skipped: signs are already ~1 bit of entropy.

Reports: `artifacts/mantissa_multimodel_bakeoff.{json,md}`,
`artifacts/mantissa_principle_candidate.md` (negative zoo; negative PBR-4).

### PBR-4 structured nibble + node formulas (experimental, negative)

User structural idea: do not jump to 1-bit; give each weight **4 bits** of
side info `c4` and let a shallow block tree supply the rest via a stored
generator `F`. Formal, bit-exact:

```
W[i] = F(node(i), c4[i]) XOR R[i]
complete = |S| + 4N/8 + |R| + metadata
```

Coordinates are already known (not a place to hide payload). Families
competed per block: constant uint16 prototype, K≤16 palette, affine
`(a·r+b·c+d·c4+e) mod 2^16`, nibble insert, shared sign/exp + 4-bit
mantissa nibble, 16 planar-XOR templates. Sparse/bitmap XOR residual
where F misses. Standalone `PBR4` container — **not** on `STAGE1A_CODECS`.

```bash
python scripts/run_pbr4.py --tag qwen
```

#### Measured PBR-4 (same 42 tensors / 83,836,928 words)

Slice exactness **PASS**. Best complete setting `16x16`.

| setting | \|S\| | c4 B | \|R\| | meta | BPW | % R=0 | vs 10.616 | vs 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 16x16 | 6,724,733 | 41,918,464 | 163,750,642 | 3,279,886 | **20.580** | 8.59% | +9.96 | no |
| 8x8 | 312,370 | 41,918,464 | 167,673,898 | 13,104,526 | 21.280 | 0.52% | +10.66 | no |
| 1x64 | 335,753 | 41,918,464 | 167,673,898 | 13,104,526 | 21.283 | 0.55% | +10.67 | no |

`choose_tile_hw(256)` = 16×16 here; 16→8 tree never split. Palette/inherit
won almost every node; affine and SE-nibble won none. H(c4)=3.14 bits, so
ANS-coding the nibble stream cannot remove the 4-bit floor enough to matter.
**Does not approach 4 BPW. Does not beat PBR-E 10.616 or the zoo 10.585.
Worse than raw 16** because F misses ~91% of weights, so \|R\| stays ~16 BPW
and c4 is extra. Named principle: **negative**.

Reports: `artifacts/pbr4_bakeoff.{json,md}`,
`artifacts/mantissa_principle_candidate.md` (PBR-4 section).

### Zoo vs PBR-E accounting

The zoo's **10.585 vs 10.616** is a **0.03–0.06 BPW DF11 mantissa-rANS
micro-gain**, not a new axis. Codec-view NLL + amortized tables vs a
synthetic raw-M split (−0.062) or vs the measured rANS container (−0.031).
Holdout-charged tables land at ~10.611. Realizing that mantissa rANS is
the only zoo lever that MDL-beats raw 7; it is tried as a bitstream in
the ≤8.0 experiments (`artifacts/path_to_50pct.md`).

### Job 3 — Huffman vs rANS on exponents

Same Qwen 42-tensor set. One tile header + freq/code table + exponent
stream + packed sign/mantissa. Exact PASS both coders.

| coder | enc B | BPW | vs bound |
| --- | ---: | ---: | --- |
| entropy bound | 111216676 | **10.613** | 0 |
| rANS | 111253029 | **10.616** | +0.003 |
| canonical Huffman | 113875515 | 10.866 | +0.254 |

rANS closes **0.25 BPW** of the old 0.26 Huffman-vs-bound gap. Whole-tensor
PBR-E now competes `bf16_exp_rans` against Huffman on complete bytes.
Still DF11-class. Not ≤4 BPW.

Full-checkpoint PBR-E (every 16-bit tensor, `pbre_whole`):

```bash
python scripts/run_pbre_full.py --model-dir /path/to/Qwen2.5-0.5B-Instruct
```

### Measured full-checkpoint PBR-E (this repo)

Same Qwen revision. **290 / 290** 16-bit tensors, **988,065,536 B** original
(the whole 0.5B Instruct weight set). Profile `pbre_whole`. **Every tensor
PASS** (uint16 / SHA-256). Winning mode: `bf16_exp_huffman`.

| | value |
| --- | ---: |
| original bytes | 988065536 |
| encoded bytes | 659740226 |
| complete-container BPW | **10.68** |
| ratio vs raw BF16 | 0.668 |
| encode | 6.62 MB/s (149 s) |
| decode | 7.42 MB/s (133 s) |

DF11/ZipNN-class on the complete checkpoint (~33% size cut). Tile headers
store rows/cols as uint16, so large tensors are Huffman-coded in stripes
(7779 tiles total) rather than one codebook each; the rate is still ~10.7
BPW. **Not ≤4 BPW. Not a 1–2 GB / 8 GB claim.**

Reports: `artifacts/pbre_full_qwen.{json,md}`.

### Exact base→finetune delta (related-checkpoint residual)

Bit-exact uint16 residual of a finetune relative to a same-arch base.
Qwen2.5-0.5B and Instruct share config (qwen2, hidden 896, 24 layers,
14 heads, intermediate 4864, vocab 151936, BF16) and all 290 tensor
names/shapes, so this pair is used as-is.

| | HF id | revision |
| --- | --- | --- |
| Base | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Finetune | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |

Comparisons (complete container bytes, all metadata): (1) target PBR-E
rANS standalone, (2) uint16 XOR then zlib / rANS, (3) sign/exp/mantissa
field deltas, (4) sparse changed-position patches, (5) default XOR
residual + dict/raw. Reports **standalone BPW**, **delta-only BPW**
(base already present), and **bundle = base PBR-E + delta** if the base
must be shipped too. Success is SHA-256 exact restore of the target, not
≤4 BPW.

```bash
python scripts/run_checkpoint_delta.py --tag qwen
```

Config: `configs/poc_delta.yaml`.

#### Measured Qwen 0.5B → Instruct (this repo)

**290 / 290** paired 16-bit tensors, **988,065,536 B** original. Every
method **PASS** (uint16 / SHA-256 reconstruct of Instruct from base+delta).

| metric | value |
| --- | ---: |
| % weights unchanged | 2.07 |
| mean XOR popcount (of 16 bits) | 3.71 |
| XOR Hamming fraction | 0.232 |
| H(uint16 XOR) | 8.09 |
| H(sign XOR) | 0.19 |
| H(exp XOR) | 1.41 |
| H(mant XOR) | 6.75 |

| method | complete B | delta-only BPW | bundle B | bundle BPW | exact |
| --- | ---: | ---: | ---: | ---: | --- |
| 1. target PBR-E rANS (standalone) | 655839276 | n/a | 655839276 | **10.62** | PASS |
| 2a. uint16 XOR + zlib | 644202227 | 10.43 | 1300022374 | 21.05 | PASS |
| 2b. uint16 XOR + rANS lo/hi | 516005521 | **8.36** | 1171825668 | 18.98 | PASS |
| 3. sign/exp/mantissa field deltas | 546290439 | 8.85 | 1202110586 | 19.47 | PASS |
| 4. sparse changed-position patches | 1029449871 | 16.67 | 1685270018 | 27.29 | PASS |
| 5. default XOR residual + dict/raw | 988076895 | 16.00 | 1643897042 | 26.62 | PASS |
| base PBR-E rANS (bundle addend) | 655820147 | n/a | 655820147 | 10.62 | PASS |

Standalone Instruct is **10.62 BPW** (same DF11-class band as rANS PBR-E).
Best residual is whole-word XOR + rANS (**8.36 delta-only BPW**) — cheaper
than the finetune alone *only if the base is already on disk*. If the base
must be shipped too, the honest total is **base PBR-E + delta ≈ 19 BPW**,
worse than sending Instruct standalone. Sparse patches lose: 98% of
weights change. Not ≤4 BPW. Not a 1–2 GB / 8 GB claim.

Reports: `artifacts/checkpoint_delta_qwen.{json,md}`.

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

## Disk tunnel PoC

Working-set demo, not a compression-ratio claim. Full Qwen2.5-0.5B-Instruct
BF16 weights stay on disk (Safetensors, optional per-tensor PBR-E). Peak RAM
holds one tensor or an embedding-row chunk plus activations, then frees it
after a GEMM. Compare that tunnel to loading every uint16 weight into RAM.

```bash
python scripts/disk_tunnel_infer.py --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct --encode
```

- Mode A (`mmap`): copy one BF16 tensor (or lm_head row slice) from Safetensors, matmul, discard.
- Mode B (`pbre`): read a per-tensor `.pbr`, bit-exact decode to uint16, matmul, discard. `embed_tokens` stays mmap because the vocab dimension exceeds the tile uint16 prefix.
- Baseline (`full`): all 16-bit tensors resident.
- Microbench: numpy Qwen2 forward on a short fixed token list (not HF `generate` quality). Tunnel logits must match the full-load path. PBR-E tiles SHA-256 match the BF16 source.
- Metrics: `artifacts/disk_ram_tunnel.md` and `.json` (peak RSS, disk-read bytes, decode/compute time, tok/s, exactness).

Measured on this Qwen 0.5B Instruct checkpoint (8 tokens, 24 layers, isolated processes): **full-load sampled peak 1006 MiB** vs **mmap 78 MiB** / **PBR-E 83 MiB**. Logits bit-identical (`max_abs=0`). PBR-E SHA **289/289 PASS**. PBR-E decode is slow (~124 s) because every tensor is rANS-decoded on the CPU; RAM stays in the mmap band.

### Faster PBR-E decode

C rANS (bit-exact vs the Python loop) plus a 2-slot decoded cache and one prefetch thread. Still `Decode(Encode(W))==W`.

```bash
python scripts/tunnel_fast_pbre.py --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct
```

Metrics: `artifacts/tunnel_fast_pbre.md`. Compare `pbre_fast` wall time to mmap / full / `pbre_slow` (original Python rANS).

Measured (8 tokens, 24 layers, isolated): **pbre_slow wall 130 s / decode 126 s** vs **pbre_fast wall 8.0 s / decode 6.0 s** (C rANS + prefetch). mmap **2.1 s**, full **3.2 s**. Logits bit-identical. Sampled peak RSS: full 1007 MiB, mmap 87 MiB, pbre_fast 153 MiB (2-slot cache).

### Even faster PBR-E decode

Fused C rANS+join (skip JSON header), 4 decode threads, ~2-layer cache, optional warm uint16 sidecar. Still `Decode(Encode(W))==W`. Single-stream rANS is not rewritten with SIMD (that would change the bitstream); parallelism is across tensors.

```bash
python scripts/tunnel_faster_pbre.py --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct
```

Metrics: `artifacts/tunnel_faster_pbre.md`. Compare `pbre_faster` / `pbre_faster_warm` wall time to `pbre_fast` / mmap / full.

Measured (same 8 tokens / 24 layers, isolated): prior `pbre_fast` **7.98 s** → this `pbre_fast` **3.21 s** (fused C + pending prefetch) → **`pbre_faster` cold 2.53 s** / **warm sidecar 0.75 s**. mmap **0.59 s**, full **1.16 s** this run (prior published mmap 2.13 s). Logits `max_abs=0`. Sampled peak RSS: full 1008 MiB, mmap 88 MiB, pbre_faster 320 MiB, warm 235 MiB. Warm sidecar is a decoded uint16 copy on disk (682 MiB, SHA-checked in 0.93 s); RAM stays a working set. Not an ≤8 BPW claim.

### Hybrid lossy tunnel (prototype)

Embeddings, norms, biases, and first/last layers stay BF16. Middle-layer attention/MLP weights are per-group **int4 + FP16 scales**. Same disk-resident RAM tunnel. **Not bit-exact** on quantized tensors; quality is logit max_abs / KL / argmax match vs full BF16.

```bash
python scripts/tunnel_hybrid_lossy.py --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct --encode
```

Metrics: `artifacts/tunnel_hybrid_lossy.md`. Not a 27B phone runtime. Not ≤8 BPW exact.

Measured (same prompt): hybrid pack **483 MiB** (317 MiB exact BF16 + 166 MiB int4). Sampled peak RSS **86 MiB** vs full **1006 MiB**, wall **2.2 s** (mmap-like). Quality vs full BF16 is **lossy**: logit max_abs 16.7, mean KL 1.01, argmax match 0.25 on 8 tokens. Uncalibrated int4 prototype, labeled clearly.

This does **not** claim phone-scale 27B, ≤8 BPW exact, or 1–2 GB / 8 GB. ≤8 BPW hunt notes stay in `artifacts/path_to_50pct.md` and are not the goal here.

## Tests

```bash
pytest
```

The suite fails loudly (`ExactnessError: EXACTNESS FAIL ...`) if any uint16 word
differs. Cases include special BF16 bit patterns (signed zero, Inf, NaN
payloads, subnormals) that an FP32 detour would be likely to destroy.

Stage 1B / Stage 2 / PBR-E / blocker-mitigation / hierarchical / Phase A /
checkpoint-delta / mantissa-zoo / disk-tunnel unit tests write tiny local
Safetensors fixtures. They do **not** download the 988 MB checkpoint. Live
download tests are skipped unless `PBR_LIVE_HF=1`.

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
                 bit-planes, grammar, transformed refs, position-value dict,
                 PBR-4 structured nibble (standalone container)
pbr_encoder/     cost-based search, decoder, Stage 1A/1B/PBR-E/Phase A CLIs
pbr_qualifier/   Stage 2 inventory, entropy, sample encode, BPW projection
scripts/         run_poc1.py, run_poc1b.py, run_qualifier.py, run_pbre.py,
                 run_pbre_full.py, run_blocker_diagnosis.py,
                 run_blocker_ablation.py, run_family_eval.py,
                 run_phase_a_mantissa_audit.py, run_exp_coder_ablation.py,
                 run_checkpoint_delta.py, run_mantissa_zoo.py, run_pbr4.py,
                 run_path_to_50pct.py, disk_tunnel_infer.py,
                 tunnel_fast_pbre.py, tunnel_faster_pbre.py, tunnel_hybrid_lossy.py
tests/           exactness, codecs, Stage 1B fixtures, qualifier math,
                 hierarchical leftovers, mantissa audit, checkpoint delta,
                 mantissa zoo, PBR-4, path-to-50pct, disk tunnel,
                 fast / faster PBR-E / hybrid lossy tunnel
configs/         poc_controlled.yaml, poc_real.yaml, poc_llama.yaml,
                 qualifier_default.yaml, poc_delta.yaml
artifacts/       measured diagnosis / ablation / Phase A / delta / zoo /
                 PBR-4 / path-to-50pct / disk-RAM tunnel / fast PBR-E /
                 faster PBR-E / hybrid-lossy reports
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
projection** for the whole 16-bit parameter set (pre-PBR-E codecs). The
base→Instruct delta is **8.36 BPW** only as a residual assuming the base
is already present; shipping both is ~19 BPW. None of these is a measured
8 GB-model result. PBR-E ≈ DF11/ZipNN-class; related-checkpoint XOR is
not a way to hide a second full checkpoint.

zlib / zstd columns, when present, are **general-purpose baselines**, not PBR
modes. Forced uint16-spatial rows in the blocker ablation exist to show
that predictor on mixed fields still loses; they are not a compression claim.

## License

This PoC is MIT. Research drafts that specify it remain separate documents.
Third-party checkpoints used in Stage 1B keep their own licenses (Qwen2.5
Instruct: Apache 2.0; Llama-3.2-1B-Instruct via the unsloth BF16 mirror:
see that model card / Llama 3.2 license).
