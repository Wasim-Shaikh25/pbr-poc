# PBR-Ladder — Complete Findings, Methods & Reproduction

**Model under test:** Qwen2.5-0.5B-Instruct (`./qwen05b/model.safetensors`, 24 layers).
**Metric:** WikiText-2 *test* perplexity (299k–336k tokens, seq 2048, non-overlapping).
**Baseline (fp16):** PPL **14.247**.  <5% quality gate = 14.959.
**Environment:** CPU only (torch cpu build), 16 cores, numpy. No CUDA.

> Effective bpw is quoted **over all model parameters** (incl. embeddings) so it is
> directly comparable to GGUF file bpw. "proj bpw" = bits over the quantized projection
> tensors only (what our stages control); embeddings are counted separately.

---

## 1. HEADLINE ACHIEVEMENT — we beat GGUF at equal-or-smaller size

Final winning configuration: **per-tensor rotation → 4-stage RVQ (4-bit projections) +
GPTQ column feedback, and 8-bit quantized embeddings.**

| model | eff bpw | ~size | WikiText PPL | C4 PPL (cross-domain) |
|---|---|---|---|---|
| **OURS proj@4.0 + embed@8** | **5.10** | **300 MB** | **15.210** | **22.811** |
| ours proj@4.0 + embed@6 | 4.55 | 268 MB | 15.268 | — |
| GGUF q2_k | 5.27 | 390 MB | 16.527 | 23.394 |
| GGUF q3_k_m | 5.49 | ~340 MB | 15.722 | 22.381 |
| GGUF q4_k_m | 6.24 | ~385 MB | 15.327 | 21.905 |
| baseline fp16 | 16.0 | ~988 MB | 14.247 | — |

**Robust claim (holds in AND out of domain):** at our 5.10 bpw operating point we beat
GGUF q2_k (nearest bit-budget) on both WikiText (+1.32) and C4 (+0.58), at fewer bits and
smaller size.

**Honest caveat (found by the cross-domain test, §5.6):** on WikiText we beat *every*
GGUF quant incl. the higher-bit q3_k_m/q4_k_m; on C4 those two (which spend more bits)
edge ahead of us. That extra margin was partly WikiText-calibration overfit. The
matched-budget win is real; the "beats everything" version was in-domain-flattered.

**What actually flipped the result** was not a new algorithm — it was **fixing our own two
bit-allocation mistakes** (we had been starving projections at 3.125-bit and leaving
embeddings at fp16 = 27% of a 0.5B model). Principle: **flat within a tensor, adaptive
across the model (projections vs embeddings).**

---

## 2. KEY SCIENTIFIC FINDINGS (each measured)

1. **Whole-model failure is a bit-BUDGET problem, not structural.** 3.875 bpw whole-model
   = +7.4% PPL; the catastrophic failures were sub-4-bit rate, not accumulation.
2. **After incoherence rotation, weights are EXACTLY memoryless Gaussian** — kurtosis
   3.000, skew 0.000, adjacent-weight correlation −0.002. Rotation+VQ @3 bpw sits **0.27 dB
   from the Shannon distortion-rate bound** (17.79 vs 18.06 dB). => *Any* memoryless
   quantizer / preconditioner is tapped out; you cannot beat Shannon and we're on it.
3. **The one exploitable structure is the activations, not the weights.** Input activation
   Gram has **participation ratio 9.17** (effectively rank-9). Output error is dominated by
   ~9 directions; a white VQ error cannot steer away from them. This sits *outside* the
   memoryless-Gaussian bound (which assumes a white input).
4. **VQ error propagates more gently across layers than GPTQ error.** VQ @3.125 whole-model
   = **16.829** beats int3+GPTQ @3.125 = **20.083**, *despite* GPTQ having higher per-tensor
   SQNR. => **Per-tensor SQNR is a misleading proxy for whole-model PPL.** (This reversed an
   earlier "swap RVQ→GPTQ" recommendation.)
5. **Bit allocation dominates the algorithm.** Our loss vs GGUF was self-inflicted; sensible
   allocation (proj 4-bit + embed 8-bit) flipped it (see §1).
6. **Calibration domain matters.** WikiText-only calibration overfits; a cross-domain (C4)
   eval shrinks the margin vs higher-bit GGUF quants. Production fix = diverse calibration.

---

## 3. THE ALGORITHM WE CONVERGED ON (reproducible pipeline)

Implemented in [`quantize_full_model.py`](quantize_full_model.py). Per linear weight `W`
(out×in), given the real input Gram `H = XᵀX/n`:

1. **Incoherence rotation** `Wr = Ro · W · Ri` (random orthogonal; regenerable from a seed,
   ~free to store). `PBR_ROT_PERLAYER=1` gives per-(proj,layer) rotations.
2. **Per-row RMS scale** `s`.
3. **Residual Vector Quantization (RVQ)**, D=8, K=256 per stage; each stage = 1 bit/weight.
   `PBR_STAGES=256,256,256,256` → 4-bit projections. Codebooks fit once over the whole
   tensor (pass 1), then a column-block pass applies **GPTQ error feedback** (pass 2).
4. **Optional free-quality knobs:** `PBR_BEAM=B` (beam-search assignment, +0.65 dB),
   Hessian-Mahalanobis assignment (+0.57 dB, see [`mahalanobis_assign.py`](mahalanobis_assign.py)).
5. **Activation low-rank correction** (`PBR_LOWRANK=r`): add `U Vᵀ` (rank r, 8-bit) fit to
   minimize the *activation-metric* residual `‖A(W−Wq−UVᵀ)ᵀ‖²`. Additive to VQ (+2.9 dB on
   a tensor), largely redundant with GPTQ (+0.26 dB). See §5.5.
6. **Quantize embeddings** (the overlooked lever): per-row 8-bit (or 6-bit) instead of fp16.
   Applied via [`embed_quant_eval.py`](embed_quant_eval.py); folds embed from 16→8 bpw.
7. **Nested / multi-quality single file** capability: one file decodes at multiple bit tiers
   near-free (+0.03 dB), see [`nested_multiquality.py`](nested_multiquality.py).

---

## 4. REPRODUCTION — exact commands

```bash
# --- winning model: 4-bit projections, whole model (~1.7h on CPU) ---
PBR_STAGES=256,256,256,256 PBR_KMEANS_IT=6 python quantize_full_model.py ./qwen05b ./qwen05b_rematch

# --- embed sweep (16/8/6/5/4-bit) + WikiText PPL at each; reports eff bpw + MB ---
PBR_PROJ_BPW=4.0 python embed_quant_eval.py ./qwen05b ./qwen05b_rematch

# --- GGUF baseline: download official quants, eval through the SAME harness ---
python gguf_eval.py q2_k q3_k_m q4_k_m

# --- cross-domain calibration check (C4) ---
python eval_on_corpus.py ./qwen05b ./qwen05b_rematch c4

# --- plain WikiText PPL of any saved model ---
python eval_ppl.py ./qwen05b_rematch
```

Key env vars for `quantize_full_model.py`: `PBR_STAGES` (codebook sizes per stage),
`PBR_KMEANS_IT` (k-means iterations, default 10), `PBR_BEAM` (beam width), `PBR_LOWRANK`
(low-rank correction rank), `PBR_ROT_PERLAYER`, `PBR_LAYER_STAGES` (per-layer bit
allocation), `PBR_INT4`/`PBR_INT4_BITS`/`PBR_INT4_G` (strong int-baseline mode),
`PBR_SAMPLES`, `PBR_SEQ`, `PBR_TEXT` (calibration text), `PBR_INIT_CKPT` (seed layers).

---

## 5. EVERY METHOD TRIED — verdict + script (so failures aren't re-run)

### 5.1 Core pipeline — KEPT
| method | script | result |
|---|---|---|
| Rotation + RVQ + GPTQ feedback | `quantize_full_model.py` | the pipeline; VQ@3.125 = 16.829 |
| Beam-search assignment | `beam_vq.py` (`PBR_BEAM`) | **+0.65 dB, free** |
| Hessian-Mahalanobis assignment | `mahalanobis_assign.py` | **+0.57 dB** |
| Nested multi-quality single file | `nested_multiquality.py` | **+0.03 dB penalty** (near-free capability) |
| Embedding quantization (8/6-bit) | `embed_quant_eval.py` | **the lever that beat GGUF** |
| Bit reallocation (proj 4-bit) | `PBR_STAGES=256,256,256,256` | 16.829 → 15.210 |

### 5.2 Mixed-precision family — FAILED (rotation flattens importance)
| method | script | result |
|---|---|---|
| Mixed precision, NO rotation | `mixed_precision_noR.py` | +3.2 dB vs naive, but **−6 dB vs pipeline** |
| Mixed precision AFTER rotation | `mixed_after_rotation.py` | **+0.01 dB** (rotation flattens all cols to 3-bit) |
| Block-local rotation + mixed | `block_rot_mixed.py` | no sweet spot; between-group var ratio 1.03–1.34× |
| Reverse water-fill allocation | `adaptive_waterfill.py` | loses to flat |
| Outlier-split (adaptive-first) | `outlier_split.py` | dominated by flat VQ |

### 5.3 Rotation variants — FAILED
| method | script | result |
|---|---|---|
| Per-(proj,layer) rotation | `PBR_ROT_PERLAYER=1` | full-24 PPL worse (18.4–18.8) |
| Partial rotation | `partial_rotation.py` | FALSIFIED — no residual corr (0.02) at D=8 |

### 5.4 Cross-layer / accumulation — FALSIFIED
| method | script | result |
|---|---|---|
| Coherent error accumulation | `coherence_probe.py` | ρ≈0.005, amplification 1.005× — refuted |
| Per-layer error growth | `eval_accum.py` | exponent 0.06 (incoherent walk) |
| Drift-correction W_eff=W·Cᵀ·H⁻¹ | `quantize_teacher.py`, `validate_teacher.py` | worse on every tensor — refuted |
| Degeneracy-manifold steering | `degeneracy_steer.py` | nets −6%..−18% error but NOT free (30–45% local cost) |

### 5.5 Structure-exploiting (the "escape the Shannon wall" ideas)
| method | script | result |
|---|---|---|
| Space-filling / memoryless VQ ceiling | `diagnostics_spacefill.py` | VQ capped at ~scalar parity (dead end) |
| Activation low-rank correction (on VQ) | `lowrank_actcorr.py`, `apply_lowrank.py`, `onepass_sweep.py` | **+2.9 dB tensor**, +0.28 PPL whole-model |
| Same, stacked on GPTQ | `gptq_lowrank.py` | **+0.26 dB** — subsumed by GPTQ (below kill threshold) |
| Trellis-coded quant (QTIP-style) | `trellis_quant.py` | +2–3.7 dB vs RVQ, loses to int3+GPTQ without error feedback |
| Weight low-rank + residual | (agent-measured) | DEAD — net −0.10 bpw (spectrum too flat) |
| Cross-layer shared subspace | (agent-measured) | DEAD — subspace overlap ~random |

### 5.6 Baselines & evaluation harnesses
| tool | script | purpose |
|---|---|---|
| GGUF comparison (real downloaded quants) | `gguf_eval.py` | q2_k 16.527 / q3_k_m 15.722 / q4_k_m 15.327 |
| Cross-domain PPL (C4 / PTB) | `eval_on_corpus.py` | calibration-overfit stress test |
| WikiText PPL of any model | `eval_ppl.py` | primary metric |
| INT3/INT4 + GPTQ baseline | `PBR_INT4=1`, `int4_rtn_model.py` | int3+GPTQ @3.125 = 20.083 |
| Transcript → markdown | `dump_chat.py` | this session's full log |

---

## 6. HONEST LIMITATIONS (what is NOT proven)

1. **0.5B only.** Embeddings are 27% of params here (pathological); at 7B they're ~4%, so
   the projection method dominates and the embedding tax vanishes — conclusions from 0.5B
   are likely *pessimistic* for real models, but untested at scale.
2. **PPL only**, no downstream task-accuracy benchmarks.
3. **No inference kernels / speed numbers.** This is a quality+size result, not a launchable
   product — real deployment needs GPU decode kernels and speed benchmarks.
4. **Cross-domain margin is thinner** than in-domain (§5.6); a diverse calibration set
   (WikiText+C4+code) is the next step to make a clean cross-domain claim.
5. The big sub-4-bit gains in the literature (trellis/QTIP) are published work we'd adopt,
   not invent. Our genuine contributions: the reallocation win, the rank-9 activation
   diagnosis, the VQ-vs-GPTQ propagation finding, and the rigorous negative-results map.

---

## 7. Older phase docs
See `stability_findings.md`, `EXPERT_FINDINGS.md`, `UNIFIED_ALGORITHM.md`,
`PBR_Ladder_Consolidated_Findings.md` for the earlier phase history.
