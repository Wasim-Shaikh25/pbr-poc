# PBR-Q4 / H95Q-S1 + selective 256-node X/Y post-codec

This line **does not** re-quantize Qwen from BF16 into a new groupwise-Q4
policy. It takes the frozen **H95Q-S1** quantized reference (container v2,
physical **7.420820 BPW**, SHA
`eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`) and
re-encodes its already-quantized mantissa K-codes.

**Product default is packed-K** (the existing S1 stream of true, unpadded
nodes). A 16×16 / 256-node X/Y matrix-family tile is stored only when its
**complete** physical cost — payload + 1-byte XY flag + that tile’s share of
the tensor mode map — is strictly smaller than packed-K. Arrays are **not**
padded to 16×16; ragged last tiles keep their true shape.

This is the follow-up to PR #15 (all-tile X/Y, 7.453890 BPW). Always-on X/Y
is not the product path.

H95Q S1 / container v2 code is unchanged.

## Hard rules

- Start from S1’s exact quantized words. Decode must match the frozen SHA.
- Emit X/Y (`XY_RANS` / `XY_BITPLANE` / `XY_PAIR` / `XY_RUN`, with causal
  predictors) **only if strictly smaller** than packed-K including map/flag
  bits. `XY_MATRIX` residual packing ties packed-K and is never selected.
- `actual_bpw = file_bytes * 8 / n_weights` from the physical `.h95x` file.
- Gate: `actual_bpw ≤ 7.420820` vs S1 v2. Report PASS/FAIL honestly.
- Quality is not re-run when the SHA matches; S1’s heldout proxy 0.990 still applies.

## Code

- `pbr_q4/predictors.py` — causal X/Y predictors + residuals
- `pbr_q4/codecs.py` — matrix-family tile codecs (candidates)
- `pbr_q4/selective.py` — packed-vs-XY choice, cheap maps (sparse / bitmap / RLE)
- `pbr_q4/container.py` — `H95X` v2 encode/decode (`python -m pbr_q4.container`)
- `pbr_q4/s1_source.py` — load S1 from a frozen `.h95q` or rebuild from the model
- `scripts/run_pbr_q4_s1_xy.py` — full S1 encode + artifacts
- Tests: `tests/test_pbr_q4_s1_xy.py`

## Run

```bash
# Preferred: transcode a frozen S1 container (v2 or v1)
PYTHONPATH=. python scripts/run_pbr_q4_s1_xy.py \
  --s1-container artifacts/pbr_h95/containers/H95Q-S1-v2.h95q

# Or rebuild Q(W) from the checkpoint (refuses if SHA drifts)
PYTHONPATH=. python scripts/run_pbr_q4_s1_xy.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct
```

Artifacts: `artifacts/pbr_q4/s1_xy_selective_qwen.{json,md}`. Large `.h95x` blobs are gitignored.

## Measured (Qwen2.5-0.5B-Instruct, frozen S1 SHA)

| Container | file_bytes | actual_bpw | exact Q(W)? |
| --- | ---: | ---: | --- |
| H95Q-S1 v2 (PR #14) | 458,266,017 | **7.420820** | yes |
| H95Q-S1 + all-tile X/Y (PR #15) | 460,308,254 | **7.453890** | yes |
| H95Q-S1 + selective X/Y (this) | 458,220,385 | **7.420081** | yes (SHA `eda64747…`) |

**PASS** vs ≤ 7.420820. Δ vs S1 v2: **−0.000739 BPW** / −45,632 bytes. 1,120 XY tiles (embed only: 1,080 rANS / 40 RUN); 1,933,000 tiles stay packed-K. Quality not re-run: Q(W) is bit-identical to S1, prior heldout proxy **0.990**.

## Honesty

- Not a production mobile runtime.
- Do not invent BPW numbers — read them from the physical file after encode.
- Do not invent quality numbers — same Q(W) ⇒ prior heldout proxy **0.990**.
