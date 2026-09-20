# Faster PBR-E disk–RAM tunnel

Disk-RAM tunnel PoC. Weights stay on disk; RAM holds a small working set (one tensor or embedding-row chunk + activations). Not a phone-scale 27B runtime. Not an ≤8 BPW exact-compression claim.

Model: `Qwen/Qwen2.5-0.5B-Instruct`  revision `7ae557604adf67be50417f59c2c2f167def9a775`  tokens=8  layers=24.

Full BF16 checkpoint stays on disk (Safetensors). The tunnel copies **one** weight tensor (or an embedding-row chunk) into RAM, converts BF16 bits to FP32 for GEMM, then frees it. PBR-E mode decodes a per-tensor container bit-exactly before the GEMM. Embedding / lm_head uses row chunks so the large table is not required resident. Not a 27B-on-phone result.

## Metrics

| mode | peak sampled RSS | ru_maxrss | disk read | decode s | compute s | wall s | tok/s | matmuls/s | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | 1007.414 MiB | 1007.176 MiB | 942.293 MiB | 1.545 | 1.094 | 3.190 | 2.508 | 72.11 | PASS |
| `mmap` | 87.359 MiB | 95.570 MiB | 942.306 MiB | 0.262 | 1.297 | 2.130 | 3.756 | 108.00 | PASS |
| `pbre_fast` | 152.840 MiB | 220.484 MiB | 713.431 MiB | 6.046 | 1.629 | 7.980 | 1.002 | 28.82 | PASS |
| `pbre_slow` | 87.387 MiB | 125.469 MiB | 713.431 MiB | 126.050 | 2.279 | 129.660 | 0.062 | 1.77 | PASS |

mmap/full ru_maxrss ratio: **0.095**. Sampled peak mmap/full: **0.087** (87.359 MiB vs 1007.414 MiB).
pbre_fast/full ru_maxrss ratio: **0.219**. Sampled peak pbre_fast/full: **0.152**. SHA checks: 289 (fail 0).
pbre_slow/full ru_maxrss ratio: **0.125**. Sampled peak pbre_slow/full: **0.087**. SHA checks: 289 (fail 0).

Each mode ran in its **own subprocess** so `ru_maxrss` is not a cumulative high-water mark of full-load + tunnel.

## Correctness

Logits vs full-load: mmap: PASS max_abs=0.000e+00; pbre_slow: PASS max_abs=0.000e+00; pbre_fast: PASS max_abs=0.000e+00. Same uint16 weights and the same FP32 GEMM.

## Disk layout

- Safetensors on disk: 942.324 MiB (942.293 MiB weight payload, 290 tensors).

`ru_maxrss` is the kernel high-water for that isolated process. Sampled peak is max `/proc/self/status` VmRSS during the forward (the number that tracks the working set after malloc_trim).

Working-set demonstration only. PBR-E is the already-measured ~10.6 BPW DF11-class codec; this tunnel does not claim ≤8 BPW.

## Decode path

rANS backend in this process catalog: `c`.
`pbre_slow` forces the original Python rANS loop (the ~128 s path).
`pbre_fast` uses the C decoder (GIL released) plus a 2-slot cache and one prefetch thread so the next tensor can decode during GEMM.
Reconstruction stays **bit-exact** (SHA-256 vs BF16 source).

Not an ≤8 BPW claim. Decode speed ≠ compression ratio.
