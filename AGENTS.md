# AGENTS.md — Highest-Priority Aims

> This file is the north star for any agent/session working in this repo. Read it first.
> Detailed results live in [`pbr_ladder/FINDINGS_AND_REPRODUCTION.md`](pbr_ladder/FINDINGS_AND_REPRODUCTION.md)
> and the plain-language [`pbr_ladder/SUMMARY_SIMPLE.md`](pbr_ladder/SUMMARY_SIMPLE.md).

## RULE 0 (HIGHEST — never skip) — Document everything in the repo
Every experiment, decision, result, failure, command, and integration step gets written down
**in the repo**, as it happens — not left in chat. The running journal is
[`pbr_ladder/DEV_LOG.md`](pbr_ladder/DEV_LOG.md) (append newest at top, dated). Findings go in
`FINDINGS_AND_REPRODUCTION.md`; plans in `DEPLOYMENT_PLAN.md`. If it isn't in the repo, it
didn't happen. Commit + push after each meaningful step. Failures are logged with the same
care as successes (they stop us re-running dead ends).

## THE GOAL — cross from "nice project" to "people care" (all 4 must land)
1. **3-bit quality holds usable** — the best-quality 3-bit run tells us this.
2. **LUT + tunnel gives real phone speed** — measured tokens/sec on an actual Android/iPhone.
3. **Generalizes to 3B–7B** — where the embedding tax vanishes and the method looks *better*.
4. **A clean shareable story** — "our allocation recipe + LUT format runs model X at N tok/s
   on a $200 phone at quality Y." A demo people share.

## Mission
Turn the PBR-Ladder quantization research (which already beats GGUF q2_k at a matched
~5-bit budget on Qwen2.5-0.5B) into a **deployable, fast, on-device LLM format** — not a
research artifact that dequantizes back to fp16 in RAM.

## The 3 hard objectives (all must hold at once)
1. **Stays packed in RAM.** Weights live as ~3–4-bit codebook indices at runtime and are
   decoded on the fly. They must NEVER expand to fp16 after loading. (The current
   `save_pretrained()` output fails this — it is research-only.)
2. **Fast on CPU.** Real tokens/sec on a laptop CPU, competitive with llama.cpp.
3. **Runs on mobile.** Compiles and runs on Android/iOS phones.

## The plan — build a new file type the *smart* way (`.pbrq`)
Do **not** write a from-scratch runtime + kernels (months, and we'd lose on speed). Instead:

- **Our RVQ codebook == a lookup table.** The entire 2024–2026 edge-LLM frontier is
  lookup-table inference. Register our format as a **custom llama.cpp quant type**
  (the pattern bitnet.cpp already proved) and decode it with a **Vec-LUT / T-MAC-style
  vector-lookup kernel** (matmul with NO dequantization).
- **Tunnel / stream** weights from flash via mmap (llama.cpp already does this); decode
  per-tensor on demand for small RAM.
- We inherit llama.cpp's mmap, threading, NEON/AVX/Metal, and mobile build for free.

`.pbrq` = RVQ codebook (the LUT) + 8-bit embeddings + per-layer bit allocation + importance
matrix, as a custom llama.cpp type, LUT-decoded, mmap-streamed.

### Research anchors (adopt, don't reinvent)
- **T-MAC** (Microsoft, EuroSys 2025) — table-lookup mpGEMM on CPU, no dequant, up to 6.6×
  vs llama.cpp on edge. https://arxiv.org/abs/2407.00088 · https://github.com/microsoft/T-MAC
- **Vec-LUT** (MobiSys 2026) — *vector* table lookup = our D=8 codebook; up to 4.2× over SOTA;
  code integrated into llama.cpp. https://arxiv.org/abs/2512.06443
- **bitnet.cpp** (ACL 2025) — proves "custom quant type + kernel added to llama.cpp".
  https://aclanthology.org/2025.acl-long.457.pdf
- **ActiveFlow / LLM-in-a-Flash** — active-weight swapping DRAM↔flash (the "tunnel"), −40% DRAM.
  https://arxiv.org/abs/2504.08378

## Roadmap (priority order)
- **P0 — 3-bit push (in progress).** Best-quality 3-bit model (beam=4 + full k-means +
  8-bit embeds). Low bits matter most: LUT kernels give their biggest speedup at 2–3 bit.
  `PBR_STAGES=256,256,256 PBR_BEAM=4 PBR_KMEANS_IT=10 python quantize_full_model.py ...`
- **P0 — best 4-bit model (in progress).** Same knobs at 4-bit → `qwen05b_best`.
- **P1 — GGUF feasibility spike.** Convert with our imatrix + bit-allocation; measure REAL
  in-RAM size, CPU tok/s, and PPL vs stock GGUF. This proves the win survives packing and
  ships mobile immediately. (Standard IQ types, zero kernel work.)
- **P2 — `.pbrq` custom llama.cpp type** *only if* P1 shows our codebook beats IQ4 enough to
  justify the kernel. First check whether Vec-LUT's vector-LUT layout can ingest our RVQ
  codebooks (D=8, K=256) directly.
- **P3 — mobile build** (Android NDK / iOS) + on-device tok/s.
- **P4 — scale + robustness:** validate at 3B/7B, diverse calibration (WikiText+C4+code),
  task-accuracy benchmarks (not just PPL).

## Quality knobs are OFFLINE-only (never cost inference speed)
`PBR_BEAM` and `PBR_KMEANS_IT` only slow the one-time compression; they do NOT affect runtime.
Always turn them UP for the final artifact. They were cut earlier only to iterate fast.

## Deployability change required
Replace the random-orthogonal rotation with a **Hadamard (structured) rotation** — O(n log n),
no stored matrix, mobile-cheap, same incoherence benefit. Random-orthogonal is research-only.

## Honest constraints (do not overclaim)
- Win is on 0.5B, WikiText PPL, partly in-domain-flattered. Not yet validated at scale/tasks.
- RVQ is ~0.27 dB from the Shannon limit — the codebook is not magic; the **allocation** is.
- "Beat GGUF" was largely fixing our own fp16-embedding mistake, and vs q2_k specifically.
- The core kernels/tech are from Microsoft/Apple teams; our edge is integration + the
  allocation recipe + the importance-matrix + the rigorous negative-results map.

## Current status (2026-09-23)
- `pbr_ladder/qwen05b_rematch` = 4-bit winner (PPL 15.210 @ ~5.1 eff bpw with 8-bit embeds).
- Best-quality 4-bit re-run → `qwen05b_best`; 3-bit best run → `qwen05b_best3` (chained after).
