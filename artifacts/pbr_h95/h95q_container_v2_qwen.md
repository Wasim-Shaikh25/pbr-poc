# PBR-H95Q physical containerization (v2)

Model: `outputs/models/Qwen__Qwen2.5-0.5B-Instruct`
Container: magic `H95Q` version **2**; exp packing = **adaptive exact** (per-tensor mode competition).
Wall time: **360.9s**

## Honesty

- actual_bpw = physical file_bytes * 8 / n_weights (includes header/metadata) — file size only.
- v2 replaces only the raw-u8 exponent stream with adaptive exact coding (RAW8/rANS/Huffman/delta-rANS/run-rANS).
- Complete exponent cost = payload + probability table + mode_id + stream_length + final_state (+ section bytes).
- Reported exp section bytes == physical sum(exp_len); reported file bytes == physical file size.
- S1 precision policy / embed tiers / mid-MLP K3 / mantissa / signs / tensor order FROZEN — Q(W) SHA must match v1.
- Prior heldout retention 0.990 is proxy from prior stack — not re-evaluated here.
- Not production quality / multilingual / runtime RAM == BPW. Large .h95q blobs are gitignored.

## Results

| Candidate | v1 actual | v2 actual | exp BPW | modes | file_bytes | exact | ≤8? |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| H95Q-Conservative | 13.2932 | 7.9128 | 2.6192 | EXP_HUFFMAN:85,EXP_RANS:205 | 488,646,675 | yes | True |
| H95Q-Balanced | 13.0129 | 7.6325 | 2.6193 | EXP_HUFFMAN:85,EXP_RANS:205 | 471,340,875 | yes | True |
| H95Q-S1 | 12.8012 | 7.4208 | 2.6193 | EXP_HUFFMAN:85,EXP_RANS:205 | 458,266,017 | yes | True |

## Per-candidate notes

### H95Q-Conservative

- Stack: `T1_B1_embed_conservative`
- Container: `artifacts/pbr_h95/containers/H95Q-Conservative-v2.h95q` (gitignored if large)
- SHA-256 quantized ref: `9c0247096cdcaeee1e0cedc4d734396962aa0e43fc2959b602d8ba749479bc1a`
- SHA-256 file: `75b09e42acaea15e8152be0813b66e5243da1c77f3f524110b78f6f29d86b701`
- Exp complete BPW: 2.619225
- Mode histogram: `{'EXP_RANS': 205, 'EXP_HUFFMAN': 85}`
- Δ vs v1 actual: -5.380459
- Prior heldout retention (stack proxy): 0.9923
- Word diffs vs v1 decode: 0

### H95Q-Balanced

- Stack: `T1_B1_embed_aggressive`
- Container: `artifacts/pbr_h95/containers/H95Q-Balanced-v2.h95q` (gitignored if large)
- SHA-256 quantized ref: `3610ea3ccac36ac646bea028222e3900ca75448989f869b6165e297a50461ef8`
- SHA-256 file: `f17c34b124762d15b35e99f53e6338366454e91000a9803af1f97875d38f7887`
- Exp complete BPW: 2.619297
- Mode histogram: `{'EXP_RANS': 205, 'EXP_HUFFMAN': 85}`
- Δ vs v1 actual: -5.380388
- Prior heldout retention (stack proxy): 0.9897
- Word diffs vs v1 decode: 0

### H95Q-S1

- Stack: `S1_B1_aggr_embed_band815_k3`
- Container: `artifacts/pbr_h95/containers/H95Q-S1-v2.h95q` (gitignored if large)
- SHA-256 quantized ref: `eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`
- SHA-256 file: `b75b8c229bef50f474a625a176fbf70270631239155d8ca8de00cf31ad4b992f`
- Exp complete BPW: 2.619289
- Mode histogram: `{'EXP_RANS': 205, 'EXP_HUFFMAN': 85}`
- Δ vs v1 actual: -5.380395
- Prior heldout retention (stack proxy): 0.9900
- Word diffs vs v1 decode: 0

## S1 physical ≤8 gate

**PASS** — H95Q-S1 `actual_bpw=7.4208` ≤ 8 (v1 was 12.801215; exp complete 2.6193 BPW).

## Exactness gates

- Q idempotent on samples: PASS
- decode(encode(Q(W))) == Q(W) uint16: PASS
- SHA-256(ref) == SHA-256(decoded) == frozen v1: PASS
- Subprocess decode-only entrypoint: PASS
- decode(v1)==decode(v2) word-identical: PASS

