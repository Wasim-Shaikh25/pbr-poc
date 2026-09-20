# Faster PBR-E disk–RAM tunnel (stage 2)

Disk-RAM tunnel PoC. Weights stay on disk; RAM holds a small working set (one tensor or embedding-row chunk + activations). Not a phone-scale 27B runtime. Not an ≤8 BPW exact-compression claim.

Model: `Qwen/Qwen2.5-0.5B-Instruct`  revision `7ae557604adf67be50417f59c2c2f167def9a775`  tokens=8  layers=24.

Full BF16 checkpoint stays on disk (Safetensors). The tunnel copies **one** weight tensor (or an embedding-row chunk) into RAM, converts BF16 bits to FP32 for GEMM, then frees it. PBR-E mode decodes a per-tensor container bit-exactly before the GEMM. Embedding / lm_head uses row chunks so the large table is not required resident. Not a 27B-on-phone result.

## Metrics

| mode | peak sampled RSS | ru_maxrss | disk read | decode s | compute s | wall s | tok/s | matmuls/s | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | 1008.195 MiB | 1007.984 MiB | 942.293 MiB | 0.767 | 0.151 | 1.159 | 6.901 | 198.40 | PASS |
| `mmap` | 87.945 MiB | 96.219 MiB | 942.306 MiB | 0.223 | 0.137 | 0.591 | 13.535 | 389.14 | PASS |
| `pbre_fast` | 132.539 MiB | 140.652 MiB | 713.431 MiB | 3.094 | 0.567 | 3.209 | 2.493 | 71.68 | PASS |
| `pbre_faster` | 320.309 MiB | 319.848 MiB | 713.431 MiB | 7.678 | 1.512 | 2.525 | 3.168 | 91.08 | PASS |
| `pbre_faster_warm` | 235.066 MiB | 235.203 MiB | 942.306 MiB | 0.891 | 0.243 | 0.747 | 10.713 | 308.00 | PASS |

mmap/full ru_maxrss ratio: **0.095**. Sampled peak mmap/full: **0.087** (87.945 MiB vs 1008.195 MiB).
pbre_fast/full ru_maxrss ratio: **0.140**. Sampled peak pbre_fast/full: **0.131**. SHA checks: 289 (fail 0).
pbre_faster/full ru_maxrss ratio: **0.317**. Sampled peak pbre_faster/full: **0.318**. SHA checks: 289 (fail 0).
pbre_faster_warm/full ru_maxrss ratio: **0.233**. Sampled peak pbre_faster_warm/full: **0.233**. SHA checks: 0 (fail 0).

Each mode ran in its **own subprocess** so `ru_maxrss` is not a cumulative high-water mark of full-load + tunnel.

## Correctness

Logits vs full-load: mmap: PASS max_abs=0.000e+00; pbre_fast: PASS max_abs=0.000e+00; pbre_faster: PASS max_abs=0.000e+00; pbre_faster_warm: PASS max_abs=0.000e+00. Same uint16 weights and the same FP32 GEMM.

## Disk layout

- Safetensors on disk: 942.324 MiB (942.293 MiB weight payload, 290 tensors).

`ru_maxrss` is the kernel high-water for that isolated process. Sampled peak is max `/proc/self/status` VmRSS during the forward (the number that tracks the working set after malloc_trim).

Working-set demonstration only. PBR-E is the already-measured ~10.6 BPW DF11-class codec; this tunnel does not claim ≤8 BPW.

## Decode path

rANS backend in this process catalog: `c`.
`pbre_fast` is the previous C rANS + 2-slot cache + 1 prefetch thread (~8 s).
`pbre_faster` (cold): fused C tile decode (rANS + BF16 join, skip JSON header),
4 worker threads, prefetch depth 16, 32-slot LRU (~two layers of linears).
Single-stream rANS is not SIMD-rewritten (bitstream must stay bit-exact);
parallelism is across tensors while GEMM runs.
`pbre_faster_warm`: decoded uint16 sidecars materialized after SHA-checked
decode; the forward copies one tensor like mmap. RAM stays a working set,
not the full ~1 GB resident checkpoint. Sidecar is full uint16 (not exponents
only) because join is cheap next to rANS; disk holds the extra copy.

Reconstruction stays **bit-exact** (logits max_abs=0 vs full-load).
Cold path SHA-256 checks every PBR-E tensor. Warm path SHA ran at materialize.

Not an ≤8 BPW claim. Decode speed ≠ compression ratio.

## Before / after

Prior published harness (`artifacts/tunnel_fast_pbre.md`, same model/tokens): `pbre_fast` wall **7.980 s** / decode 6.046 s, mmap **2.130 s**, full **3.190 s**, sampled peak RSS pbre_fast 153 MiB.

This run (isolated processes). `pbre_fast` also picks up fused C decode, no `gc.collect` on the GEMM path, and a pending prefetch queue, so it is already faster than 8 s. `pbre_faster` adds 4-way decode + layer cache. `decode_s` is the **sum** of job times and exceeds wall when workers overlap.

| mode | peak sampled RSS | decode s | compute s | wall s | vs prior pbre_fast 7.98 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | 1008.195 MiB | 0.767 | 0.151 | **1.159** | 6.88× |
| `mmap` | 87.945 MiB | 0.223 | 0.137 | **0.591** | 13.50× |
| `pbre_fast` | 132.539 MiB | 3.094 | 0.567 | **3.209** | 2.49× |
| `pbre_faster` | 320.309 MiB | 7.678 | 1.512 | **2.525** | 3.16× |
| `pbre_faster_warm` | 235.066 MiB | 0.891 | 0.243 | **0.747** | 10.69× |

Cold `pbre_faster` wall 2.525 s (3.16× vs prior 7.98 s). Warm sidecar 0.747 s. mmap this run 0.591 s (page cache; prior published mmap was 2.13 s).

## Warm sidecar

Materialize: 289 tensors, 682.636 MiB decoded uint16, 0.925s wall, SHA checks 289.
