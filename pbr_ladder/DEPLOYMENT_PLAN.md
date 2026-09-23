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

## Unified custom-kernel option (the novel path)
One custom llama.cpp quant type = our RVQ vector codebook + a LUT-GEMM decode kernel
(T-MAC/AQLM-style: precompute activation·codebook partial dot-products, matmul becomes
lookups+adds, no dequant) + mmap tunnel + NEON for mobile. Keeps our codebook AND gets LUT
speed. Cost: writing a new GEMM kernel (+ NEON) — weeks, not a flag. **Build only if the P1
numbers show scalar-3bit+T-MAC misses the 98% bar and RVQ clears it.** Starting points: AQLM
CPU kernel, Arm codebook kernels (arXiv 2501.00032). Do NOT build speculatively.

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

## Decision framework (resolve after the numbers come in)
- **If scalar-3bit + our recipe (Family 1) matches our RVQ quality** → ship Family 1 (T-MAC),
  fastest path, all 4 goals reachable. Our novelty = the recipe + imatrix + findings.
- **If RVQ meaningfully beats scalar-3bit** → go Family 2 (AQLM/Q4X), keep the codebook.
- **Only if both fall short** → invest in the novel vector-LUT kernel.

The Shannon-wall result predicts Family 1 ≈ Family 2 in quality → **bias toward Family 1 for
shipping, Family 2 for the "our method" research artifact.** Numbers decide.

---

## Pipeline (end-to-end, target state)
```
calib data ──► [imatrix build]            (our activation importance)
base fp16 ──► [Hadamard rotate]           (deployable, replaces random-orthogonal)
          ──► [quantize]  ── Family 1: scalar 2/3/4-bit (GPTQ fmt) + per-layer allocation
                          └─ Family 2: RVQ D=8 codebook (AQLM fmt) + per-layer allocation
          ──► [8-bit embeddings]          (the lever)
          ──► [pack to GGUF / AQLM]       (stays packed; never fp16 in RAM)
          ──► [LUT decode kernel]         F1: T-MAC/Vec-LUT · F2: AQLM/Arm-codebook
          ──► [mmap tunnel]               (page in only touched weights)
          ──► [CPU tok/s + mobile tok/s + PPL]   ← the numbers that judge all 4 goals
```

## Roadmap (priority)
- **P0** (running): best-quality 3-bit + 4-bit models (`qwen05b_best3`, `qwen05b_best`).
- **P1 — GGUF + T-MAC spike (Family 1):** emit scalar 3-bit GPTQ + our imatrix + 8-bit embeds,
  convert to GGUF, build llama.cpp (+T-MAC), measure **real in-RAM MB, CPU tok/s, PPL** vs stock.
  *This is the fastest way to get the judging numbers.*
- **P1b — AQLM sanity (Family 2):** confirm our RVQ codebooks load into AQLM; get its PPL/CPU
  numbers to compare against Family 1. Decides the family question.
- **P2 — Hadamard rotation swap** in `quantize_full_model.py` (mobile-cheap incoherence).
- **P3 — mobile build** (Android NDK / iOS) of the winning family; on-device tok/s.
- **P4 — scale (3B/7B) + diverse calibration + task-accuracy benchmarks.**

## Honest risks
- LUT speedup is biggest at 2-bit (6.7×) vs 4-bit (2.8×) → pushes us to 2–3 bit.
- Tunneling beyond mmap needs activation sparsity we don't yet have (dense model).
- Family 1 means our RVQ codebook is not in the shipped model (recipe survives, codebook doesn't).
- Everything is 0.5B / PPL / in-domain so far. Numbers at scale + on tasks are unproven.

## Sources
- T-MAC arXiv 2407.00088 · github.com/microsoft/T-MAC
- Vec-LUT arXiv 2512.06443 · github.com/Cipherxzc/vlut.cpp
- bitnet.cpp ACL 2025.acl-long.457
- AQLM arXiv 2401.06118
- Arm codebook kernels arXiv 2501.00032 · Q4X (10xengineers.ai) · Llama-Mobile arXiv 2608.21134
- ActiveFlow arXiv 2504.08378 · LLM in a Flash
