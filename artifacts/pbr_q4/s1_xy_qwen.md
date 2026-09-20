# PBR H95Q-S1 + all-tile 256-node X/Y

Post-codec of the **frozen H95Q-S1 quantized reference** (SHA `eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`). Not a new BF16→Q4 policy.
Wall time: **119.2s**

## Honesty

- This is S1 + all-tile X/Y *post-codec*, not a new BF16→groupwise-Q4 line.
- Numerical values are the frozen H95Q-S1 quantized reference; decode must match that SHA.
- actual_bpw = physical file_bytes * 8 / n_weights (header, scales/maps, mode IDs, checksums included).
- Packed-K mantissa bytes are a *baseline metric only* — the container never stores packed-raw tiles.
- Every eligible tile uses the 256-node X/Y matrix family (MATRIX / RANS / BITPLANE / PAIR / RUN).
- Quality: if SHA matches S1, Q(W) is identical; prior heldout proxy 0.990 is inherited, not re-evaluated.
- Not a production mobile runtime. Not a claim that X/Y beats packed-K on every tile.
- Compare physical BPW to H95Q-S1 container v2 = 7.42082 (PR #14).

## Physical rate

| Container | file_bytes | actual_bpw | notes |
| --- | ---: | ---: | --- |
| H95Q-S1 v2 (PR #14) | 458,266,017 | **7.420820** | packed-K mantissas + adaptive exp |
| H95Q-S1 + X/Y (this) | 460,308,254 | **7.453890** | all-tile matrix family |
| Δ (X/Y − v2) | +2,042,237 | **+0.033070** | negative = smaller file |

## Mantissa path vs packed-K baseline (metric only)

- Packed-K mantissa payload (baseline, not stored): 234,493,952 B (3.797221 BPW)
- X/Y mantissa payload (stored): 233,601,989 B (3.782777 BPW)
- Payload Δ (X/Y − packed): -891,963 B
- Tile flag bytes: 1,934,120
- Tiles: 1,934,120 (all matrix-family: True)
- Tiles with X/Y payload < packed-K: 5,704 (saved 891,963 B on those tiles)
- Tiles with X/Y payload = packed-K: 1,928,416
- Tiles with X/Y payload > packed-K: 0
- Packed-K baseline is computed on **padded 16×16 tiles** (metric only), not the S1 v2 stream.

## Mode mix

- Tile modes: `{'XY_MATRIX': 1928416, 'XY_RANS': 1055, 'XY_RUN': 4649}`
- Predictors: `{'AVG': 485559, 'PREVIOUS': 1075399, 'LEFT': 90234, 'UP': 91644, 'PAETH': 191284}`
- Traversals: `{'ROW': 1156603, 'COL': 450095, 'COL_SERP': 158861, 'ROW_SERP': 168561}`
- Exp modes: complete BPW 2.619289

## Exactness

- Source: `rebuilt_from_model` (`outputs/models/Qwen__Qwen2.5-0.5B-Instruct`)
- SHA-256 quantized ref: `eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`
- Matches frozen S1: **YES**
- decode(encode(Q_S1)) == Q_S1: **YES**
- Subprocess decode SHA: **YES**

## Quality

Not re-evaluated. S1 heldout proxy retention **0.990** still applies because Q(W) is bit-identical.

