# PBR-Ladder → Deployable Edge LLM: Integration Plan, Pipeline & Triage

Owner doc for the deployment push. Governed by [`../AGENTS.md`](../AGENTS.md) RULE 0
(document everything). Running journal: [`DEV_LOG.md`](DEV_LOG.md).

## The goal (all 4 must land — from AGENTS.md)
1. 3-bit quality usable · 2. LUT+tunnel real phone tok/s · 3. scales to 3B–7B · 4. shareable demo.

## PRODUCT SPEC (owner's target)
Ship **only sub-4-bit models** that retain **≥98% quality** vs fp16.
- **Metric must be pinned:** primary = downstream **task-accuracy retention ≥98%**; dev proxy =
  PPL increase ≤ ~3%. (Strict PPL ratio and task-accuracy disagree — 0.5B 4-bit is +6.8% PPL
  but typically ~98–99% task accuracy. Decide the metric before claiming pass/fail.)
- **Bit target scales with model size** to hold the 98% bar: 0.5B needs ~4-bit; 3B–7B can reach
  ~3-bit at the same 98% (more weight redundancy). This is *why* goal #3 (scale) is mandatory —
  the sub-4-bit-at-98% promise is delivered on 3B–7B, not on 0.5B.
- Measured so far (0.5B, PPL): 4-bit rematch 15.210 (+6.8%); 3-bit VQ 16.829 (+18%, will NOT
  hit 98% on 0.5B). 3-bit at 98% is a *scale* result, not a 0.5B result.

## DECISION (2026-09-23): NO custom kernel, NO industry infra — recipe-on-GGUF only
Owner constraint: cannot spend weeks writing a GEMM/NEON kernel, and has no industry-level
validation infra. Resolution: **contribute at the quantization-recipe layer, use only
existing tools & kernels.** We DROP the RVQ codebook and the custom kernel entirely (Shannon-
wall result → scalar quant at equal bits ≈ same quality, so ~no real loss).

**Our method = a recipe on top of llama.cpp's existing `llama-quantize`:**
- importance matrix → `llama-imatrix` (built from our activation analysis)
- per-tensor bit allocation → `--tensor-type` overrides
- 8-bit embeddings → `--token-embedding-type q8_0`
Output = a **standard GGUF** that runs on every llama.cpp target (CPU/Android/iOS/Metal) with
ZERO new code. T-MAC is an optional drop-in accelerator, also no code from us.

**Validation on consumer/free hardware only:** PPL via `llama-perplexity` (laptop CPU);
task-accuracy via `lm-eval-harness` (laptop / free Colab/Kaggle); mobile tok/s via a prebuilt
llama.cpp Android app on the owner's own phone; 3B via free Colab T4 / Kaggle P100. Never
quantize huge models ourselves — the recipe is model-agnostic.

**Honest scope after pivot:** novelty narrows to the allocation recipe + imatrix tuning +
findings. A *well-tuned* stock GGUF already does some of this, so the quality edge may be
MODEST; the product value is packaging (sub-4-bit @ ≥98%, runs on a phone, exact recipe).

### Parked (do NOT build unless this whole path fails AND funding/infra appears)
- Unified custom llama.cpp type (RVQ codebook + LUT-GEMM + NEON): keeps codebook + LUT speed
  but needs a new kernel (weeks) — out of scope under current constraints.
- AQLM path (keeps RVQ, kernels already exist CPU/GPU): viable for desktop, weak on mobile.

---

## Repo triage — what actually exists, verified (2026-09-23)

**Critical correction to an earlier optimistic claim:** T-MAC and Vec-LUT are **NOT**
vector-codebook kernels. They accelerate **scalar** ultra-low-bit weights only. This splits
the deployable world into two families, and we must consciously pick.

### Family 1 — Scalar low-bit + bit-serial LUT kernels (fastest, most turnkey)
| Tool | What it accepts | Platforms | Integration surface |
|---|---|---|---|
| **T-MAC** (arXiv 2407.00088) | int1/2/3/4 × int8/fp16 — **BitNet, GPTQ, GGUF** scalar formats | x86, ARM (Apple Silicon), **Android**, ARM64 Win | `convert-hf-to-gguf.py --enable-t-mac`, build `-DLLAMA_TMAC=ON`, `llama-bench` |
| **Vec-LUT / vlut.cpp** (arXiv 2512.06443) | **ternary (1.58-bit) only** — I1/I2 packings; "vector"=across-token SIMD, NOT VQ weights | Intel/AMD/ARM, Linux/Android/macOS/Win | fork of llama.cpp, pure C/C++ |
| **bitnet.cpp** (ACL 2025) | ternary (TL / I2_S) | edge CPU | proves "custom quant type + kernel bolted onto llama.cpp" |

→ To use Family 1 we **drop our RVQ codebook** and emit a **scalar** 2/3/4-bit model (GPTQ
format), but we **keep the real win**: our bit-allocation recipe + 8-bit embeddings + the
activation importance matrix (imatrix). Since our RVQ is already ~0.27 dB from the Shannon
wall, scalar quant at the same bits costs little quality — and we get the fastest existing
CPU/mobile kernels for free. **This is the fastest route to "runs fast on a phone."**

### Family 2 — Vector-codebook quantization (keeps our RVQ novelty)
| Tool | What it is | Kernels | Notes |
|---|---|---|---|
| **AQLM** (arXiv 2401.06118) | Additive quantization = **exactly our RVQ**; Pareto-optimal <3-bit | GPU **and CPU** (splits 16-bit codebook into 8-bit sub-codebooks for L1/L2) | HF-integrated; our codebooks map onto its format |
| **Q4X** (10xengineers) | codebook quant **integrated into llama.cpp** | RISC-V vector CPU (extendable) | precedent: codebook type inside llama.cpp |
| **Arm fine-grained codebook kernels** (arXiv 2501.00032) | optimized **ARM CPU** codebook kernels + custom file format | ARM/mobile | closest thing to a mobile VQ-codebook kernel |
| **Llama-Mobile** (arXiv 2608.21134) | 2.7-bit VLM quant on mobile | mobile | reference for mobile low-bit |

→ Family 2 preserves our RVQ codebook end-to-end. AQLM is the natural home (our RVQ *is*
additive quantization). Mobile codebook kernels exist (arXiv 2501.00032) but are younger and
less turnkey than T-MAC.

### The genuinely-novel (hard) option
A **vector-codebook + across-token-parallel LUT kernel** — i.e. apply Vec-LUT's 1→N
token-parallel trick to a D=8/K=256 *vector* codebook — **does not exist off the shelf**.
Building it would be a real, novel kernel contribution, not a weekend. Park it behind evidence.

---

## Chosen path (locked): recipe-on-GGUF, no kernel, consumer hardware
Family 1 only. We express our method entirely through llama.cpp's existing quantize tool and
run on existing kernels. RVQ codebook + custom kernel are PARKED (see DECISION above). The
Shannon-wall result says scalar ≈ RVQ quality, so we lose ~no quality by dropping the codebook.

---

## Pipeline (end-to-end, NO new code)
```
calib data ──► llama-imatrix                 (importance matrix; our activation insight)
base fp16  ──► llama-quantize                 (scalar sub-4-bit)
                 --imatrix <file>
                 --token-embedding-type q8_0  (the embedding lever)
                 --tensor-type <regex:type>   (our per-layer bit allocation)
           ──► standard .gguf                 (stays packed; never fp16 in RAM; mmap'd)
           ──► llama.cpp (+ optional T-MAC)    (existing kernels: CPU/Android/iOS/Metal)
           ──► llama-perplexity + lm-eval + on-phone tok/s   ← judges all 4 goals
```
Note: arbitrary rotation is NOT expressible in stock GGUF without runtime support, so the
recipe drops rotation (or uses only GGUF-native transforms). Re-test whether imatrix +
allocation + 8-bit embeds alone still beats a *well-tuned* stock GGUF — that is the open question.

## Roadmap (priority) — all on consumer/free hardware
- **P0** (running): best-quality reference models 3-bit + 4-bit (`qwen05b_best3/best`) — for our
  own upper-bound reference; the shipped models come from the GGUF recipe below.
- **P1 — GGUF recipe spike (no kernel):** build/download llama.cpp, `llama-imatrix` on our
  calib data, `llama-quantize` at sub-4-bit with `--token-embedding-type q8_0` + `--tensor-type`
  allocation, vs a *well-tuned* stock GGUF baseline. Measure **in-RAM MB, CPU tok/s, PPL**.
  Open question: does our recipe beat a well-configured stock quant, and by how much?
- **P2 — task-accuracy harness:** `lm-eval-harness`, a few tasks, to state the real "≥98%".
- **P3 — on-phone tok/s:** prebuilt llama.cpp Android app on the owner's own phone.
- **P4 — scale to 3B** on free Colab/Kaggle; confirm sub-4-bit @ ≥98% holds where it should.

## Honest risks
- The pivot narrows novelty to "a better llama.cpp quant recipe + imatrix + findings." Edge over
  a *well-tuned* stock GGUF may be MODEST — P1 must honestly measure this, not assume the old win.
- Our documented GGUF win was vs *downloaded defaults*; a properly-tuned baseline is a harder bar.
- Dropping rotation may cost some of our quality advantage — P1 tells us.
- Everything is 0.5B / PPL / in-domain so far. Scale + task numbers still unproven.

## Sources
- T-MAC arXiv 2407.00088 · github.com/microsoft/T-MAC
- Vec-LUT arXiv 2512.06443 · github.com/Cipherxzc/vlut.cpp
- bitnet.cpp ACL 2025.acl-long.457
- AQLM arXiv 2401.06118
- Arm codebook kernels arXiv 2501.00032 · Q4X (10xengineers.ai) · Llama-Mobile arXiv 2608.21134
- ActiveFlow arXiv 2504.08378 · LLM in a Flash
