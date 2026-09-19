# PBR Stage 1A proof of concept

Position-Based Binary Reconstruction (PBR) encodes BF16 weights as a
**generated / predicted bit pattern plus an exact XOR residual**. Reconstruction
is `Decode(Encode(W)) == W` on the original uint16 words.

This repository implements **Stage 1A only**: a controlled-method validation of
that contract. It is **not** a model compressor, not a Safetensors qualifier,
and **not evidence of real-checkpoint compression**.

> **Research status:** concept and early proof of concept. Synthetic / controlled
> tensor results demonstrate reversible reconstruction and codec mechanics. They
> must never be presented as production ratios or as results on a real model.

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

- Not a real Safetensors / checkpoint experiment (PoC 1B / Gate 3).
- Not a model qualification scanner (PoC 2).
- Not a fused tile inference runtime.
- Not a claim of 1–2 GB storage for an 8 GB model.
- Hierarchical modes (cross-layer references, grammar coding, adaptive region
  trees) are out of scope. The encoder is Direct-first.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

Runtime dependency: `numpy`. Tests: `pytest`. Optional baseline: `zstandard`.

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

## Tests

```bash
pytest
```

The suite fails loudly (`ExactnessError: EXACTNESS FAIL ...`) if any uint16 word
differs. Cases include special BF16 bit patterns (signed zero, Inf, NaN
payloads, subnormals) that an FP32 detour would be likely to destroy.

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
pbr_core/        uint16 views, tiles, container, hashing, bit packing
pbr_codecs/      raw, predictors, residuals, dictionaries, components
pbr_encoder/     cost-based search, decoder, verification, Stage 1A CLI
scripts/         generate_controlled_data.py, run_poc1.py
tests/           exactness, codecs, container, round-trip
configs/         poc_controlled.yaml
```

Later stages from the research drafts (`pbr_qualifier`, fused runtime, real
checkpoints) are intentionally absent.

## Interpreting numbers

A low BPW on `constant_block` or `previous_row` only shows that the codec
recognizes the pattern it was given. The `random_uint16` row is the honesty
check: PBR must not invent compression on unstructured bits.

zlib / zstd columns, when present, are **general-purpose baselines**, not PBR
modes.

## License

MIT. Research drafts that specify this PoC remain separate documents.
