# PBR H95Q-S1 + selective 256-node X/Y

**Gate vs S1 v2 7.420820: PASS**

Post-codec of the **frozen H95Q-S1 quantized reference** (SHA `eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`). Not a new BF16→Q4 policy. Packed-K is the product default; X/Y only when strictly smaller.
Wall time: **153.5s**

## Honesty

- This is S1 + *selective* X/Y post-codec, not a new BF16→groupwise-Q4 line.
- Numerical values are the frozen H95Q-S1 quantized reference; decode must match that SHA.
- actual_bpw = physical file_bytes * 8 / n_weights (header, maps, mode IDs, checksums included).
- Packed-K of true (unpadded) nodes is the product default. X/Y is emitted only if strictly smaller including map + per-XY flag bits.
- Arrays are not padded to 16×16; ragged last tiles store their true shape.
- Always-on all-tile X/Y (PR #15, 7.453890 BPW) is not the product path.
- Quality: if SHA matches S1, Q(W) is identical; prior heldout proxy 0.990 is inherited, not re-evaluated.
- Not a production mobile runtime.
- Compare physical BPW to H95Q-S1 container v2 = 7.420820 (PR #14) and PR #15 all-tile = 7.453890.

## Physical rate

| Container | file_bytes | actual_bpw | notes |
| --- | ---: | ---: | --- |
| H95Q-S1 v2 (PR #14) | 458,266,017 | **7.420820** | packed-K mantissas + adaptive exp |
| H95Q-S1 + all-tile X/Y (PR #15) | 460,308,254 | **7.453890** | always-on matrix family |
| H95Q-S1 + selective X/Y (this) | 458,220,385 | **7.420081** | packed default, XY iff smaller |
| Δ (this − S1 v2) | -45,632 | **-0.000739** | negative = smaller; ≤0 is PASS |
| Δ (this − PR #15) | -2,087,869 | **-0.033809** | vs always-on |

**Verdict: PASS** — `actual_bpw` ≤ 7.420820.

## Selective mantissa vs packed-K (true nodes, no 16×16 pad)

- Packed-K whole-tensor mantissa (S1 stream, true nodes): 233,533,104 B (3.781662 BPW)
- Stored mantissa blobs: 233,487,408 B (3.780922 BPW)
- Saved vs packed mantissa: +45,696 B
- XY tiles / packed tiles / all tiles: 1,120 / 1,933,000 / 1,934,120
- Tensors with at least one XY blob: 1
- Mode-map bytes: 27
- XY flag bytes (XY tiles only): 1,120
- XY payloads: 59,669
- Product default packed: True
- Always-on all-tile matrix family: False

## Mode mix (XY tiles only; packed tiles have no predictor flags)

- Tile modes: `{'XY_RANS': 1080, 'XY_RUN': 40}`
- Predictors: `{'PAETH': 10, 'PREVIOUS': 1097, 'UP': 13}`
- Traversals: `{'ROW': 23, 'COL': 1087, 'COL_SERP': 10}`
- Exp modes: complete BPW 2.619289

## Exactness

- Source: `rebuilt_from_model` (`outputs/models/Qwen__Qwen2.5-0.5B-Instruct`)
- SHA-256 quantized ref: `eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`
- Matches frozen S1: **YES**
- decode(encode(Q_S1)) == Q_S1: **YES**
- Subprocess decode SHA: **YES**

## Quality

Not re-evaluated. S1 heldout proxy retention **0.990** still applies because Q(W) is bit-identical.

