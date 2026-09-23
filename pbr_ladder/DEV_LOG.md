# DEV_LOG — PBR-Ladder Deployment Push

Append newest entries at the top, dated. Log decisions, runs, results, failures, commands.
Governed by [`../AGENTS.md`](../AGENTS.md) RULE 0. If it isn't here, it didn't happen.

---

## 2026-09-23

### Defensible pipeline BUILT + proven end-to-end (pbr_pipeline.py)
- Built [`pbr_pipeline.py`](pbr_pipeline.py): one CPU-only command chain, base fp16 GGUF +
  calib -> standard sub-4-bit GGUF + auditable manifest.json. Subcommands: quantize / validate
  / run / recipes. Recipes encode findings (ship-3bit default; iq/2bit gated to >=1.5B/3B).
  Binary discovery via --llama-bin/$LLAMA_BIN/PATH. `--reuse-imatrix` skips the slow rebuild.
- PROVEN on 0.5B (laptop CPU, sharing cores with other jobs): quantize+imatrix 145 s ->
  432.0 MB, 5.485 eff bpw (gguf stored-element count 630M); `run` generates coherent text at
  **75.1 tok/s**; matches prior validated PPL 16.509 for the same recipe/artifact.
- Manifest = the defensible handoff: base sha256, exact quantize command, output sha256, size,
  eff bpw, PPL. Anyone can reproduce + re-verify. Shipped calib/eval slices in pipeline_data/.
- Timing measured for the OWNER's machine: 0.5B full cycle ~3-5 min idle; **3B ~1-1.5 h**
  (imatrix ~10-20 min, quantize ~2-5 min, PPL ~30-60 min) -- all CPU, free, no infra. GPU
  would NOT speed quantize (CPU-bound); only helps lm-eval task-accuracy later.
- RAM property CONFIRMED (answering owner): output stays ~3-bit in RAM, never re-expands to
  fp16. Components applied: packed low-bit GGUF + mmap (only packed pages resident) + per-block
  on-the-fly dequant inside the matmul (no full fp16 tensor ever materialized). T-MAC LUT =
  optional drop-in later. Proof: artifacts/disk_ram_tunnel.md (0.078x peak RSS, bit-exact).
  Our own PBR-E per-tensor tunnel works bit-exactly but 0.06 tok/s in Python -> parked research.
- Added [`PIPELINE.md`](PIPELINE.md) (how-to + reproduced result) and [`POSITIONING.md`]
  (POSITIONING.md) (honest wedge: BYO-model PTQ+DX+provenance; PrismML=QAT fixed models;
  "sell for millions" = acquisition, needs traction or standout tech; path = prove 3B ->
  task-accuracy -> on-phone -> package DX -> ship reference models+demo).
- MCP/free-platform check: available MCPs (Render/Canva/Docs/scheduling/browser) are NOT free
  GPU boxes; but the recipe path is CPU-only so no GPU platform is needed on the critical path.


### IQ / below-3-bit RESULTS — codebook thesis failed on 0.5B (decisive)
- Q3_K_M+imatrix 16.509@432MB BEATS IQ3_M+imatrix 16.933@419MB. IQ (codebook) does NOT help at
  3-bit on 0.5B; IQ shines at 2-bit. IQ2_M tanks to 19.2 PPL for only ~27MB saved (embeddings
  dominate). q8-embed lever still not worth it.
- NET: at 0.5B we have NO quant-quality edge over well-tuned stock GGUF beyond stock imatrix
  (which is llama.cpp's, not ours -> not a differentiator). Only untested lever = fused rotation,
  whose benefit is largest at 2-bit = pointless on 0.5B.
- DECISION: STOP optimizing on 0.5B (wrong testbed, embedding-dominated). Next real test = 3B on
  free Colab/Kaggle, where embeddings ~4% and low-bit+rotation+imatrix can matter. Table:
  gguf_validation_results.md.
- Business context (from research): PrismML/Deepgrove hit 98% at 27B-on-phone via ternary QAT
  (trained-from-scratch, needs GPU infra) -- a bigger effort than our PTQ. Our realistic wedge =
  "quantize + deploy the customer's OWN model on-device, dead simple" (PTQ+DX), which PrismML
  (sells fixed models) does not serve. Monetization here = API/licensing/complement/acquisition;
  "sell for millions" = acquisition, needs traction or standout tech.


### DECISION LOCKED — fused offline quantizer -> standard IQ-GGUF (owner)
- Build ONE offline quantizer that fuses the cheaply-ownable pieces and emits a STANDARD
  IQ-GGUF (no new kernel, runs on every llama.cpp backend today):
  Hadamard rotation (baked/fused into adjacent layers, QuaRot-style) -> GPTQ error feedback ->
  IQ codebook quant (llama.cpp IQ2/IQ3, already kernel-backed on AVX/NEON/Metal) ->
  imatrix weighting -> per-tensor bit allocation.
- DROP AQLM + T-MAC (need runtime kernels / weeks / infra — not doable easily). PARKED.
- Streaming: use llama.cpp mmap now; "our own" streaming layer is a LATER, separate effort.
- Target regime = below 3-bit (IQ2/IQ3), which is exactly where these components pay off.
- Honest boundary: beats stock GGUF on quality/size (rotation+feedback+alloc the default lacks);
  beats AQLM on speed/portability/mobile (it's standard GGUF); does NOT strictly beat AQLM on
  pure quality-per-bit (that needs AQLM's parked kernel).
- Licenses: llama.cpp/T-MAC/QuaRot MIT, AQLM Apache-2.0 — all permissive; algorithms not
  copyrightable, reimplement cleanly with paper attribution; verify each before copying code.
- Deliverable framing: NOT a new file FORMAT (needs a kernel) but a new QUANTIZER/recipe that
  produces best-in-class standard IQ-GGUF files.


### GGUF recipe small-validation RESULTS (0.5B, laptop CPU, b11138)
- Pipeline works end-to-end on consumer hardware (base->imatrix->quantize->PPL->packed gguf).
- stock Q3_K_M 16.549 @432MB | +imatrix 16.509 @432MB (-0.04 FREE) | +q8 embed/out 16.468 @500MB (+68MB).
- HONEST: imatrix = free win; q8-embed lever NOT worth it vs a *well-tuned* stock GGUF (k-quants
  already quantize embeddings). Edge over tuned stock GGUF is small. Earlier "beat GGUF" was vs
  downloaded defaults / our own fp16-embed mistake, not a tuned baseline.
- SCALE THESIS CONFIRMED: on 0.5B even Q3_K_M is ~7 eff bpw (embeddings 27.5% of params dominate).
  Sub-4-bit-at-quality only real at 3B-7B. Full table: gguf_validation_results.md. Script:
  run_gguf_validation.sh.
- Next: matched-size --tensor-type allocation test (<=432MB); then repeat on 3B (free Colab/Kaggle).


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

### PIVOT (owner constraint: no weeks for a kernel, no industry infra)
- **Decision: NO custom kernel, recipe-on-GGUF only.** Drop RVQ codebook + custom LUT-GEMM
  entirely. Express our method through llama.cpp's EXISTING `llama-quantize`: `llama-imatrix`
  + `--token-embedding-type q8_0` + `--tensor-type` allocation → standard GGUF → existing
  kernels (CPU/Android/iOS/Metal). Zero new code. Shannon-wall → scalar ≈ RVQ quality, ~no loss.
- **Validation = consumer/free hardware:** llama-perplexity (laptop), lm-eval-harness
  (laptop/free Colab/Kaggle), on-phone tok/s (owner's phone), 3B on free Colab T4/Kaggle P100.
- **Honest:** novelty narrows to recipe + imatrix + findings; edge over a *well-tuned* stock
  GGUF may be modest (our old win was vs downloaded defaults). Dropping rotation may cost some
  advantage. P1 must measure this honestly. Parked: custom kernel, AQLM. See DEPLOYMENT_PLAN.md.

### Next
- P1 GGUF recipe spike (no kernel, mostly not CPU-heavy): get llama.cpp, build imatrix on our
  calib text, quantize sub-4-bit with embed+allocation overrides, compare vs well-tuned stock
  GGUF on in-RAM MB / CPU tok/s / PPL. Heavy steps wait for the quant runs to free the CPU.
