# PBR-H95Q physical containerization

Model: `outputs/models/Qwen__Qwen2.5-0.5B-Instruct`
Container: magic `H95Q` version 1; exp packing = **raw u8** (exactness).
Wall time: **117.3s**

## Honesty

- actual_bpw = physical file_bytes * 8 / n_weights (includes header/metadata).
- Exponents stored as raw u8 for exactness — NOT the ~2.62 BPW entropy reference used in estimated packed BPW.
- Therefore actual_bpw is expected >> est_bpw (~+5.38 on the exp term alone) even before metadata.
- Prior estimated packed BPW + heldout retention copied from H95Q-stack artifacts (proxy only).
- Exactness gate: decode(encode(Q(W))) == Q(W) uint16 bit-identical + matching SHA-256.
- Not production quality / multilingual / runtime RAM == BPW. Large .h95q blobs are gitignored.

## Results

| Candidate | est_bpw (prior) | actual_bpw | file_bytes | overhead | exact | sha | ≤8? |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| H95Q-Conservative | 7.8937 | 13.2932 | 820,912,080 | 1,055,360 | yes | ok | False |
| H95Q-Balanced | 7.6134 | 13.0129 | 803,601,824 | 1,055,488 | yes | ok | False |
| H95Q-S1 | 7.4017 | 12.8012 | 790,527,456 | 1,055,552 | yes | ok | False |

## Per-candidate notes

### H95Q-Conservative

- Stack: `T1_B1_embed_conservative`
- Container: `artifacts/pbr_h95/containers/H95Q-Conservative.h95q` (gitignored if large)
- SHA-256 quantized ref: `9c0247096cdcaeee1e0cedc4d734396962aa0e43fc2959b602d8ba749479bc1a`
- SHA-256 file: `00e1757b4db6971bd0831f3549140f809a0e506e4335b5e863bc659de1f18d0f`
- Stream BPW (payload only): 13.2762
- Δ(actual − est): +5.3995 BPW
- Prior heldout retention (stack): 0.9923 (proxy_GO)
- Gap vs est is dominated by raw_u8 exponents (8 BPW) vs EXP_BPW_REF=2.62 used in estimates; metadata/header overhead is secondary.

### H95Q-Balanced

- Stack: `T1_B1_embed_aggressive`
- Container: `artifacts/pbr_h95/containers/H95Q-Balanced.h95q` (gitignored if large)
- SHA-256 quantized ref: `3610ea3ccac36ac646bea028222e3900ca75448989f869b6165e297a50461ef8`
- SHA-256 file: `2b91e9f1870780a387d957e009db047175c2f2973df4bb8a9aa89cc2244bf7b7`
- Stream BPW (payload only): 12.9958
- Δ(actual − est): +5.3995 BPW
- Prior heldout retention (stack): 0.9897 (proxy_GO)
- Gap vs est is dominated by raw_u8 exponents (8 BPW) vs EXP_BPW_REF=2.62 used in estimates; metadata/header overhead is secondary.

### H95Q-S1

- Stack: `S1_B1_aggr_embed_band815_k3`
- Container: `artifacts/pbr_h95/containers/H95Q-S1.h95q` (gitignored if large)
- SHA-256 quantized ref: `eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`
- SHA-256 file: `7589e7aa73e7ce45a7d7f2adefdd4fdfadaff39a0b209c2e00418d49b90bde03`
- Stream BPW (payload only): 12.7841
- Δ(actual − est): +5.3995 BPW
- Prior heldout retention (stack): 0.9900 (proxy_GO)
- Gap vs est is dominated by raw_u8 exponents (8 BPW) vs EXP_BPW_REF=2.62 used in estimates; metadata/header overhead is secondary.

## S1 ≤8 physical?

**No** — H95Q-S1 `actual_bpw=12.8012` (> 8). Expected with raw-u8 exponents; prior ≤8 was on *estimated* packed BPW (EXP_BPW_REF=2.62). Container still proves exact independent decode of Q(W).

## Exactness gates

- Q idempotent on samples: PASS
- decode(encode(Q(W))) == Q(W) uint16: PASS
- SHA-256(ref) == SHA-256(decoded): PASS
- Subprocess decode-only entrypoint: PASS

