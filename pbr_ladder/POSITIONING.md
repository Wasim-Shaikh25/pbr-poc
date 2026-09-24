# POSITIONING — where PBR-Ladder actually stands, and how it becomes sellable

Owner doc. Governed by [`../AGENTS.md`](../AGENTS.md) RULE 0. Honest by design:
this file exists so we never sell a story the results don't back. Running
journal: [`DEV_LOG.md`](DEV_LOG.md). Evidence: [`gguf_validation_results.md`](gguf_validation_results.md).

---

## 1. What we honestly have (as of 2026-09-24)

**★ A PROVEN sub-4-bit-at-quality result at 3B** ([`results_3b_taskacc.md`](results_3b_taskacc.md)):
`IQ3_M + imatrix` on Qwen2.5-3B is **3.86 bpw (genuinely sub-4-bit), 23% smaller than Q4_K_M,
and retains 99.1% of Q4-level HellaSwag accuracy** (1000 tasks) — statistically
indistinguishable from Q4. This clears the ≥98% product bar on a real downstream task, not
just PPL. It was impossible at 0.5B (embedding-dominated), which is why scale was mandatory.
Honest asterisks: retention is vs a Q4 near-lossless proxy (f16 deleted to save disk; vs true
fp16 likely ~98.5–99%), 1000/10042 tasks, one task family. But the core claim now has teeth.

**A working, CPU-only, no-infra pipeline** ([`pbr_pipeline.py`](pbr_pipeline.py)) that turns
any base model into a standard sub-4-bit GGUF + an auditable manifest, in minutes on a
laptop. Proven end-to-end on 0.5B (145 s, 432 MB) and 3B (the result above), model-agnostic
(Qwen + phi3 tested). Ship recipe: `ship-sub4` = IQ3_M + imatrix.

**A real "tunnel" result** ([`../artifacts/disk_ram_tunnel.md`](../artifacts/disk_ram_tunnel.md)):
weights stay on disk, RAM holds only the working set — **0.078× peak RSS vs full-load
(78 MiB vs 1006 MiB), bit-exact logits**. This is the "never re-expands to fp16 in RAM"
property, and llama.cpp's mmap delivers it for free on our output. Our own PBR-E per-tensor
decode tunnel also works bit-exactly but is slow in pure Python (0.06 tok/s) — research, not
shippable yet.

**A body of quantization research** (RVQ ~0.27 dB from the Shannon distortion-rate bound;
rotation; GPTQ error feedback; bit-allocation; the embedding lever).

**★ CONFIRMED (2026-09-24): IQ3_M is the training-free ceiling — and we are ON it.**
We rigorously tried to beat IQ3 with rotation + vector/lattice codebooks + imatrix + RVQ (5 experiments,
[`SUB3BIT_FINDINGS.md`](SUB3BIT_FINDINGS.md)). All lose; the heroic matched-bit RVQ loses 46%. IQ3 sits
on the rate-distortion floor (0.099 weighted err ≈ theoretical ~0.125). This is a POSITIVE for
positioning: our shipped 99%-quality 3-bit model is provably near information-theoretic optimal for
training-free PTQ. Beating it needs QAT (training) — a different, heavier product. We stop out-engineering
the quantizer and ship IQ3.

### What we do NOT have (say it out loud)
- **No proven quant-quality edge over a *well-tuned* stock GGUF at 0.5B beyond `imatrix`** —
  and `imatrix` is llama.cpp's, not ours. Our earlier "beat GGUF" win was against *downloaded
  defaults* / our own fp16-embed mistake, not a tuned baseline.
- ~~No 3B/7B result yet~~ → **DONE at 3B** (above). 7B still untested, but the thesis now has one solid proof point.
- ~~No task-accuracy numbers~~ → **DONE** (HellaSwag 99.1% retention). Broader tasks (MMLU/ARC) still worth adding before a hard marketing claim.
- **Still no *novel quantizer*** — the win uses stock llama.cpp IQ3 + imatrix. The moat is the pipeline (DX + provenance) + the validated recipe/findings, not a new algorithm.
- **No on-phone tok/s measured** on a real device yet.
- **No custom kernel, no mobile tok/s measured on-device yet.**

Selling anything beyond this list today would be dishonest. The next section is how we
*earn* the right to sell.

---

## 2. The wedge — why anyone would pick us

We are not going to out-quantize Apple/Microsoft/Meta on raw bits-per-quality, and we don't
need to. The differentiation is **workflow, not a magic number**:

> **"Bring your own model. One CPU-only command turns it into a sub-4-bit build that runs on
> a phone, with a manifest proving exactly how it was made and how to re-verify it."**

Why that's a real gap:
- **PrismML / Deepgrove Bonsai** sell *their* fixed ternary models, trained from scratch with
  QAT on GPU infra. If you want *your* model on-device, they don't serve you.
- **Free quantizers** (llama.cpp, AutoGPTQ, AWQ) give you knobs, not a defensible, reproducible,
  measured artifact. No manifest, no recipe-as-product, no "here's the ≥98% proof."
- **The big players** optimize their own first-party stack; a neutral, model-agnostic
  "quantize + deploy anyone's model, provably" layer is not their priority.

Our edge is **PTQ + developer experience + provenance**, on **consumer hardware**. That is a
narrow, honest, real slice — enough to "stand out," which is the stated goal.

### PTQ vs QAT (know which game we're in)
- **QAT** (PrismML): retrain with quantization in the loop → best quality at extreme bits, but
  needs GPUs, data, and per-model effort. Not us.
- **PTQ** (us): quantize an already-trained model, no retraining → cheap, fast, model-agnostic,
  runs on a laptop. Weaker at the extreme low-bit frontier, but *infinitely more accessible*.
  Our bet: PTQ + a great pipeline is what most teams actually need.

---

## 3. How the comparable companies earn (grounded, not hype)

| Company / thing | What they sell | How money moves |
|---|---|---|
| **PrismML / Deepgrove (Bonsai)** | Their own ultra-low-bit (ternary/1.58b) models + on-device runtime | License/SDK to device & app makers; enterprise deals; likely acquisition target |
| **Free quantizers** (TheBloke-style, llama.cpp) | Nothing directly | Reputation → consulting, jobs, sponsorship, upstream influence; not a direct-revenue model |
| **Neural Magic / OctoML (pre-acquisition)** | Inference speedups + tooling | Enterprise licensing, then **acquired** (NM→Red Hat, Octo→Nvidia) |
| **Us (target)** | The pipeline/SDK + a few reference sub-4-bit models + provenance | Licensing/SDK, per-model service, or **acquisition** once traction/tech is proven |

**Blunt read on "sell for millions":** that outcome is almost always an **acquisition**, and
acquisitions buy one of two things — **traction** (users/revenue) or **standout tech a bigger
player wants to absorb**. We don't have either *yet*. This doc's job is to make the path to one
of them concrete, not to pretend we're already there.

---

## 4. The path to sellable (in order)

1. **Prove the engine where it matters — 3B.** Run the pipeline on a 3B model (laptop CPU, ~1 h,
   free). If sub-4-bit holds quality where 0.5B couldn't, *that* is the first real proof point.
2. **State quality in task terms, not just PPL.** `lm-eval-harness` on a few tasks → an honest
   "retains X% of fp16 accuracy at Y bits." This is the number buyers/partners care about.
3. **Measure on-device.** Prebuilt llama.cpp Android app on the owner's own phone → real tok/s +
   RAM. Turns "runs on mobile" from a claim into a screen recording.
4. **Package the DX.** The one-command pipeline + manifest *is* the product surface. Make it
   trivial: `pbr quantize <model> --recipe ship-3bit` → shippable file + proof.
5. **Ship 2–3 reference models + a demo.** A shareable on-phone demo is what makes a partner
   reply to an email.

Only after 1–3 do we have anything defensible to *sell* or to attract an acquirer with.

---

## 5. Honest risks to the whole thesis
- The 3B result might show our recipe still ≈ a well-tuned stock GGUF. Then the product is DX +
  provenance only — thinner, but still real.
- Mobile tok/s at sub-4-bit might disappoint without a low-bit kernel (T-MAC is the fallback,
  drop-in, no code from us).
- "Standing out against Apple/MSFT/Meta" is a *distribution/trust* problem as much as a tech one;
  a small player wins by being the neutral, model-agnostic, provable option — not the fastest.
