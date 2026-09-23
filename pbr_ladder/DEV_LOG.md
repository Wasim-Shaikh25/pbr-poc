# DEV_LOG — PBR-Ladder Deployment Push

Append newest entries at the top, dated. Log decisions, runs, results, failures, commands.
Governed by [`../AGENTS.md`](../AGENTS.md) RULE 0. If it isn't here, it didn't happen.

---

## 2026-09-23

### Repo triage of LUT / codebook runtimes (verified via repos + papers)
- **Correction to an earlier optimistic claim:** T-MAC and Vec-LUT are **scalar** ultra-low-bit
  LUT accelerators, **not** vector-codebook kernels. Verified:
  - T-MAC accepts int1/2/3/4 (BitNet/GPTQ/GGUF); integrates via `convert-hf-to-gguf.py
    --enable-t-mac`, `-DLLAMA_TMAC=ON`. Platforms incl. Android.
  - vlut.cpp (Vec-LUT) is **ternary-only** (I1/I2). "Vector" = across-token SIMD, NOT VQ weights.
- **Vector-codebook home is a different family:** AQLM (= additive quant = our RVQ; has CPU
  kernels), Q4X (codebook type in llama.cpp), Arm fine-grained codebook kernels (arXiv 2501.00032).
- **Consequence:** two deployable families. F1 = scalar low-bit + T-MAC (drop RVQ, keep
  allocation+imatrix, fastest/most turnkey, best mobile). F2 = AQLM/Q4X (keep RVQ codebook).
  Shannon-wall result predicts ≈ equal quality → bias F1 for shipping, F2 for the research artifact.
  Full analysis in [`DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md).
- A vector-codebook + token-parallel LUT kernel does **not** exist off the shelf → genuinely
  novel but real kernel work; parked behind evidence.

### Quant runs launched (best-quality knobs ON — offline cost only, zero inference cost)
- **4-bit best:** `PBR_STAGES=256,256,256,256 PBR_BEAM=4 PBR_KMEANS_IT=10` → `qwen05b_best`.
  layer 0 = 740s → full run ~5h. Log: scratchpad/best_run.out.
- **3-bit best (chained after 4-bit, no CPU contention):** `PBR_STAGES=256,256,256 PBR_BEAM=4
  PBR_KMEANS_IT=10` → `qwen05b_best3`. Log: scratchpad/three_run.out.
- Rationale: LUT kernels give the biggest speedup at 2–3 bit, so a strong 3-bit model is the
  strategic target; beam+full-kmeans reclaim quality lost by going lower without adding bits.
- Note: `PBR_BEAM`/`PBR_KMEANS_IT` were cut in earlier experiment runs purely for iteration
  speed; they do NOT affect inference — always max them for the final artifact.

### Docs / process
- Added `../AGENTS.md` RULE 0 (document everything) + the 4 "people care" goals.
- Created `DEPLOYMENT_PLAN.md` (integration triage + pipeline + roadmap) and this log.
- Prior baseline still standing: `qwen05b_rematch` = 4-bit, PPL 15.210 @ ~5.1 eff bpw (8-bit embeds),
  beats GGUF q2_k at matched budget.

### Product spec pinned (owner)
- Ship ONLY sub-4-bit at **≥98% quality**. Metric must be chosen: task-accuracy retention
  (primary) vs PPL proxy — they disagree (0.5B 4-bit = +6.8% PPL but ~98–99% task acc).
- **Bit target scales with model size**: 0.5B needs ~4-bit for 98%; 3B–7B can reach ~3-bit at
  98%. The sub-4-bit-at-98% promise is a **3B–7B** deliverable, not a 0.5B one. 3-bit on 0.5B
  = +18% PPL, will not hit 98%. Reinforces goal #3 (scale) as mandatory.
- Decision logged in DEPLOYMENT_PLAN.md (PRODUCT SPEC + Unified custom-kernel option).

### Unified-kernel option evaluated
- Feasible: one custom llama.cpp type = RVQ codebook + LUT-GEMM decode (no dequant) + mmap +
  NEON. Keeps novelty AND gets LUT speed. Cost = writing a new GEMM kernel (weeks). Gated:
  build ONLY if P1 shows scalar-3bit+T-MAC misses 98% and RVQ clears it. Not speculative.

### Next
- P1: GGUF + T-MAC feasibility spike — emit scalar 3-bit + imatrix + 8-bit embeds, measure real
  in-RAM MB / CPU tok/s / PPL. Judges goals 1–2, the F1-vs-F2 call, AND whether the custom
  kernel is justified. Draft export/convert script (no CPU) while quant runs finish.
