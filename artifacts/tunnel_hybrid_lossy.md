# Hybrid lossy disk–RAM tunnel

Hybrid LOSSY tunnel PoC. Sensitive tensors stay BF16; bulk MLP/attn projections are per-group int4. Not bit-exact on quantized weights. Not a phone-scale 27B runtime. Not an ≤8 BPW exact-compression claim.

Model: `Qwen/Qwen2.5-0.5B-Instruct`  revision `7ae557604adf67be50417f59c2c2f167def9a775`  tokens=8  layers=24.

Full BF16 checkpoint stays on disk (Safetensors). The tunnel copies **one** weight tensor (or an embedding-row chunk) into RAM, converts BF16 bits to FP32 for GEMM, then frees it. PBR-E mode decodes a per-tensor container bit-exactly before the GEMM. Embedding / lm_head uses row chunks so the large table is not required resident. Not a 27B-on-phone result.

## Metrics

| mode | peak sampled RSS | ru_maxrss | disk read | decode s | compute s | wall s | tok/s | matmuls/s | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | 1006.246 MiB | 1006.090 MiB | 942.293 MiB | 0.886 | 1.038 | 2.709 | 2.953 | 84.89 | PASS |
| `mmap` | 85.949 MiB | 557.160 MiB | 942.306 MiB | 0.272 | 1.299 | 2.192 | 3.649 | 104.92 | PASS |
| `hybrid` | 86.398 MiB | 557.160 MiB | 482.866 MiB | 0.948 | 0.766 | 2.184 | 3.663 | 105.32 | LOSSY |

mmap/full ru_maxrss ratio: **0.554**. Sampled peak mmap/full: **0.085** (85.949 MiB vs 1006.246 MiB).
hybrid/full ru_maxrss ratio: **0.554**. Sampled peak hybrid/full: **0.086**. SHA checks: 135 (fail 0).

Each mode ran in its **own subprocess** so `ru_maxrss` is not a cumulative high-water mark of full-load + tunnel.

## Correctness

Logits vs full-load: mmap: PASS max_abs=0.000e+00; hybrid: LOSSY max_abs=1.674e+01 KL=1.007e+00 argmax=0.250. Same uint16 weights and the same FP32 GEMM.

## Disk layout

- Safetensors on disk: 942.324 MiB (942.293 MiB weight payload, 290 tensors).

`ru_maxrss` is the kernel high-water for that isolated process. Sampled peak is max `/proc/self/status` VmRSS during the forward (the number that tracks the working set after malloc_trim).

Working-set demonstration only. PBR-E is the already-measured ~10.6 BPW DF11-class codec; this tunnel does not claim ≤8 BPW.

## Lossy quality (vs full BF16, same tokens)

Quantized tensors are **not** bit-exact. Exact BF16 tensors (embed, norms, biases, first/last layer) still SHA-256 match.

- logit max_abs: **16.74**
- logit MAE: 1.13
- mean KL(ref || hyp): **1.007**
- argmax match rate: **0.250**

## Hybrid pack size

- exact BF16 copies: 136 tensors, 316.668 MiB
- int4 + FP16 scales: 154 tensors, 166.185 MiB
- pack total: **482.853 MiB** (not claimed as ≤8 BPW / 50%).

Working-set + lossy-prototype demonstration only. Not a 27B phone runtime.
