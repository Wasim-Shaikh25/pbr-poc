# Disk–RAM tunnel PoC

Disk-RAM tunnel PoC. Weights stay on disk; RAM holds a small working set (one tensor or embedding-row chunk + activations). Not a phone-scale 27B runtime. Not an ≤8 BPW exact-compression claim.

Model: `Qwen/Qwen2.5-0.5B-Instruct`  revision `7ae557604adf67be50417f59c2c2f167def9a775`  tokens=8  layers=24.

Full BF16 checkpoint stays on disk (Safetensors). The tunnel copies **one** weight tensor (or an embedding-row chunk) into RAM, converts BF16 bits to FP32 for GEMM, then frees it. PBR-E mode decodes a per-tensor container bit-exactly before the GEMM. Embedding / lm_head uses row chunks so the large table is not required resident. Not a 27B-on-phone result.

## Metrics

| mode | peak sampled RSS | ru_maxrss | disk read | decode s | compute s | wall s | tok/s | matmuls/s | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | 1005.879 MiB | 1005.629 MiB | 942.293 MiB | 0.913 | 0.806 | 2.093 | 3.823 | 109.90 | PASS |
| `mmap` | 78.102 MiB | 180.828 MiB | 942.306 MiB | 0.319 | 1.427 | 2.132 | 3.752 | 107.86 | PASS |
| `pbre` | 82.625 MiB | 180.828 MiB | 713.431 MiB | 124.082 | 2.495 | 127.676 | 0.063 | 1.80 | PASS |

mmap/full ru_maxrss ratio: **0.180**. Sampled peak mmap/full: **0.078** (78.102 MiB vs 1005.879 MiB).
pbre/full ru_maxrss ratio: **0.180**. Sampled peak pbre/full: **0.082**. PBR-E SHA checks: 289 (fail 0).

Each mode ran in its **own subprocess** so `ru_maxrss` is not a cumulative high-water mark of full-load + tunnel.

## Correctness

Logits vs full-load: mmap: PASS max_abs=0.000e+00; pbre: PASS max_abs=0.000e+00. Same uint16 weights and the same FP32 GEMM.

## Disk layout

- Safetensors on disk: 942.324 MiB (942.293 MiB weight payload, 290 tensors).
- PBR-E per-tensor dir: 289 files, 682.636 MiB → 453.761 MiB (10.635 complete BPW on encoded tensors). Skipped mmap: ['model.embed_tokens.weight'].

`ru_maxrss` is the kernel high-water for that isolated process. Sampled peak is max `/proc/self/status` VmRSS during the forward (the number that tracks the working set after malloc_trim).

Working-set demonstration only. PBR-E is the already-measured ~10.6 BPW DF11-class codec; this tunnel does not claim ≤8 BPW.
