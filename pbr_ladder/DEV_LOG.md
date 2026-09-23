# DEV_LOG — PBR-Ladder Deployment Push

Append newest entries at the top, dated. Log decisions, runs, results, failures, commands.
Governed by [`../AGENTS.md`](../AGENTS.md) RULE 0. If it isn't here, it didn't happen.

---

## 2026-09-24

### 3B PHASE 2 — matched baselines CORRECT the phase-1 read (honest revision)
- Filled the matched baselines (reused imatrix). Full frontier, best-first:
  | config | PPL | size | bpw |
  |---|---|---|---|
  | stock Q4_K_M (ref) | 9.302 | 1930MB | 5.00 |
  | iq3-imat-q8e | 9.734 | 1564MB | 4.055 |
  | **iq3-imat (IQ3_M+imatrix) STAR** | **9.804** | **1489MB** | **3.860** |
  | ship-3bit (Q3_K_M+imat) | 12.992 | 1590MB | 4.12 |
  | stock Q3_K_M | 13.333 | 1590MB | 4.12 |
  | ship-2bit-iq (IQ2+imat+q8e) | 12.285 | 1257MB | 3.26 |
  | stock-iq2m-imat (IQ2+imat) | 12.398 | 1140MB | 2.957 |
- REAL WINNER: **IQ3_M + imatrix @ 3.86 bpw = near-Q4 quality (+5.4% PPL vs Q4_K_M), 23%
  smaller, genuinely sub-4-bit.** THIS is the shippable sub-4-bit-at-quality model.
- IQ codebooks WIN at 3B (IQ3 9.80 crushes Q3_K_M 13.0 at same ~4bpw) -- OPPOSITE of 0.5B
  where scalar Q3_K beat IQ3. Scale thesis fully confirmed; codebook thesis was right, just
  needed scale.
- TWO CORRECTIONS to yesterday's phase-1 read (I was wrong, matched baselines prove it):
  (1) The q8-EMBED LEVER IS MARGINAL, not "decisive": IQ2+q8e vs IQ2 = -0.9% PPL for +10%
      size; IQ3 case -0.7% for +0.2bpw. DROPPED from the default recipe.
  (2) IQ2 IS TOO AGGRESSIVE for the >=98% bar (+33% PPL vs Q4). I mis-anchored IQ2 vs
      Q3_K_M; vs the real target it fails quality. IQ2 = extreme-size only. IQ3 = sweet spot.
- HONEST SCOPE: every config is stock llama.cpp type+imatrix+flags. No novel quantizer. Our
  value = turnkey reproducible measured pipeline + the sweet-spot finding, not a quant moat.
- ACTIONS: rewrote results_3b.md (unified table + corrections); updated pbr_pipeline.py
  recipes (new `ship-sub4` = IQ3_M+imatrix default; ship-2bit-iq -> extreme-size, embed
  dropped; embed notes corrected). NEXT: lm-eval task-accuracy on iq3-imat to prove >=98%.

### Ops (autonomous, overnight)
- 0.5B best-quality 4-bit reference `qwen05b_best` COMPLETED (~6.7h; beam=4, kmeans=10,
  stages=[256]x4; ~4 bpw weights only, embeds/head/norms full precision). Research
  upper-bound reference, NOT the shipped artifact. PPL eval deferred (0.5B = wrong testbed;
  not worth CPU now). Eval later with: python eval_ppl.py ./qwen05b_best.
- STOPPED the chained 0.5B 3-bit run (`qwen05b_best3`, PID 13828) right after it started:
  it would hog CPU for ~6h and contend with the meaningful 3B phase-2 analysis. Low-value
  0.5B reference on the wrong testbed; nothing lost (0% progress), trivially re-runnable
  (`python quantize_full_model.py ./qwen05b ./qwen05b_best3` with PBR_STAGES=256,256,256
  PBR_BEAM=4 PBR_KMEANS_IT=10). Protecting CPU for the 3B work is the right call.

### 3B RESULT — recipe WINS at scale (the 0.5B null was the testbed, not the method)
- Ran full end-to-end on Qwen2.5-3B-Instruct f16 (bartowski GGUF, downloaded via robust
  resume-retry loop after curl AND huggingface_hub both dropped ~640-805MB on the flaky CDN).
- Same base + same imatrix + same 150KB eval slice, ctx=512. Sizes/bpw from gguf stored count.
  | config | PPL | size | eff bpw |
  |---|---|---|---|
  | stock Q3_K_M (no imatrix) | 13.333 | 1590MB | 4.12 |
  | ship-3bit (Q3_K_M+imatrix) | 12.992 | 1590MB | 4.12 |  (-2.6% PPL, FREE, same size)
  | ship-2bit-iq (IQ2_M+imatrix+q8embed) | 12.285 | 1257MB | 3.26 | (-7.9% PPL AND -21% size)
  | stock Q4_K_M (ref) | 9.302 | 1930MB | 5.00 |
- HEADLINE: ship-2bit-iq is Pareto-better than stock Q3_K_M -- SMALLER (1257 vs 1590MB) AND
  lower PPL (12.29 vs 13.33), and genuinely sub-4-bit (3.26 bpw = product spec). This was
  IMPOSSIBLE at 0.5B. Scale thesis CONFIRMED.
- WHY: ship-2bit-iq uses IQ2 weights (lower precision than Q3) yet wins -> the 8-bit embed
  lever + imatrix allocation is decisive at 3B (Qwen large vocab). The lever that was
  worthless at 0.5B matters at scale. Full analysis: results_3b.md.
- HONEST caveats: ship-3bit's win = imatrix (llama.cpp's, not ours). ship-2bit-iq lacks a
  matched baseline (stock IQ2 needs imatrix; runner made it without -> failed). PENDING:
  re-run stock-IQ2+imatrix (no q8embed) to isolate the embed lever. PPL only, 150KB slice,
  task-accuracy still unproven. stock Q4_K_M (5bpw) still clearly better quality (9.30) --
  sub-4-bit trades quality for size, does NOT beat 4-bit quality.
- Runner bugs fixed: mb() used a /d/ path Windows-Python couldn't read (sizes blank live) ->
  switched to stat; stock IQ2 now built WITH imatrix (= the matched baseline). run_3b_test.sh.
- NEXT (autonomous): re-download f16, run matched IQ2+imatrix + IQ3 frontier baselines
  (reuse saved imatrix), then task-accuracy (lm-eval) + on-phone tok/s.


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
