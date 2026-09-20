# Matrix Mantissa V3 — Qwen 2-D probe

Model: `Qwen/Qwen2.5-0.5B-Instruct` · tile `(16, 16)` · V3

## Aggregate (tiled region only)

- Values tiled: **98304**
- Candidate payload BPW: **7.1562**
- RAW candidate BPW: **7.1562**
- Implied total (sign1+exp8+cand): **16.1562**
- Mode counts: `{'raw': 384}`
- Exact round-trip: **True**
- Wall: 3.63s

Prior full-model adaptive mantissa (reference): ~7.0001 mant / ~10.62 total (ALL_RAW).

## Honesty

Probe only on a tiled subset of 2-D linear weights. candidate_payload_bpw is the evaluate_tile winner payload (not whole-model). implied_total assumes raw sign (1) + raw exp (8) + candidate mantissa — exp is usually ~2.6 BPW with rANS, so real totals would be lower by ~5.4 if exp coding is reused. Not a <=4 BPW claim. container_bpw includes Gate D framing + stored exp plane and is not comparable to PBR-E totals.

## Per tensor

| tensor | shape | tiles | cand BPW | raw BPW | vs raw | modes | exact |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `model.layers.0.self_attn.q_proj.weight` | [896, 896] | 64 | 7.1562 | 7.1562 | 1.0 | `{'raw': 64}` | True |
| `model.layers.0.self_attn.o_proj.weight` | [896, 896] | 64 | 7.1562 | 7.1562 | 1.0 | `{'raw': 64}` | True |
| `model.layers.0.mlp.up_proj.weight` | [4864, 896] | 64 | 7.1562 | 7.1562 | 1.0 | `{'raw': 64}` | True |
| `model.layers.0.mlp.down_proj.weight` | [896, 4864] | 64 | 7.1562 | 7.1562 | 1.0 | `{'raw': 64}` | True |
| `model.layers.12.mlp.down_proj.weight` | [896, 4864] | 64 | 7.1562 | 7.1562 | 1.0 | `{'raw': 64}` | True |
| `model.layers.23.mlp.gate_proj.weight` | [4864, 896] | 64 | 7.1562 | 7.1562 | 1.0 | `{'raw': 64}` | True |

