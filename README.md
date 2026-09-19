# PBR Stage 1A / 1B proof of concept

Position-Based Binary Reconstruction (PBR) encodes BF16 weights as a
**generated / predicted bit pattern plus an exact XOR residual**. Reconstruction
is `Decode(Encode(W)) == W` on the original uint16 words.

This repository implements **Stage 1A** (controlled tensors) and **Stage 1B**
(real public-checkpoint tensors). It is **not** a production model compressor,
not a full-model qualifier, and **not evidence that an 8 GB checkpoint becomes
1–2 GB**.

> **Research status:** concept and early proof of concept. Stage 1A uses
> synthetic tensors. Stage 1B checks that the same codecs stay bit-exact on
> real BF16/F16 weights and reports complete-container BPW. Neither stage is a
> production ratio claim.

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
- Not a model qualification scanner (PoC 2).
- Not a fused tile inference runtime.
- Not a claim of 1–2 GB storage for an 8 GB model. Do not scale Stage 1B BPW
  into that story.
- Hierarchical modes (cross-layer references, grammar coding, adaptive region
  trees) are out of scope. The encoder is Direct-first.

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

## Tests

```bash
pytest
```

The suite fails loudly (`ExactnessError: EXACTNESS FAIL ...`) if any uint16 word
differs. Cases include special BF16 bit patterns (signed zero, Inf, NaN
payloads, subnormals) that an FP32 detour would be likely to destroy.

Stage 1B unit tests write a tiny local Safetensors fixture (Qwen-like shapes,
a few kilobytes). They do **not** download the 988 MB checkpoint. A live
download test exists but is skipped unless `PBR_LIVE_HF=1`.

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
pbr_codecs/      raw, predictors, residuals, dictionaries, components
pbr_encoder/     cost-based search, decoder, Stage 1A/1B CLIs, HF download
scripts/         generate_controlled_data.py, run_poc1.py, run_poc1b.py
tests/           exactness, codecs, container, round-trip, Stage 1B fixtures
configs/         poc_controlled.yaml, poc_real.yaml
```

Later stages from the research drafts (full-model qualification scanner, fused
runtime) are intentionally absent.

## Interpreting numbers

A low BPW on `constant_block` or `previous_row` only shows that the codec
recognizes the pattern it was given. The `random_uint16` row is the honesty
check: PBR must not invent compression on unstructured bits.

Stage 1B BPW on real Qwen tensors is a measurement of **this encoder on those
tensors**, including headers. It is not a projection for an 8 GB model.

zlib / zstd columns, when present, are **general-purpose baselines**, not PBR
modes.

## License

This PoC is MIT. Research drafts that specify it remain separate documents.
Third-party checkpoints used in Stage 1B keep their own licenses (Qwen2.5
Instruct: Apache 2.0; see above).
