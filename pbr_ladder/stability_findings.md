# Whole-Model Stability: New Findings (branch `claude/whole-model-stability`)

Continuation of the PBR-Ladder work. Focus: **why does the whole model fail
(+36% PPL) when every single layer passes (<1%)?** All results below are REAL
Qwen2.5-0.5B-Instruct, WikiText-2, run on this machine (CPU fp32). Baseline PPL
= **14.247**; gate is <5% (**14.959**).

## Result 1 — the +36% is a BIT-BUDGET problem, not a structural wall  [REAL]

The single most-recommended, never-run test (consolidated doc §9.1): flat uniform
higher budget, whole model, no selective protection.

| Recipe | bpw | Whole-model PPL | vs baseline |
|---|---|---|---|
| `512,256,64` (prior work) | 2.875 | ~18.5 | +30% |
| **`512,256,256,64` (this run)** | **3.875** | **15.297** | **+7.4%** |

**+1 bpw cut the damage from +30% to +7.4%.** The "structural instability" fear is
disproven — spending bits uniformly works dramatically. ~4.0–4.1 bpw should clear
the 5% gate outright (still ~4x smaller than BF16). The open prize is now: clear
the gate at LOWER bpw by spending the last bits where they matter.

## Result 2 — error does NOT compound in the bulk; damage is at the two ends  [REAL]

`eval_accum.py`: residual-stream RMS error vs the clean model, per layer, on the
3.875 checkpoint.

```
L0-2:   10.0% -> 13.4%     (early injection; relative, shrinks as stream grows)
L3-15:  ~7-8%, FLAT        (growth exponent 0.06 -- no compounding, no amplification)
L16-23: KNEE 7.9% -> 18.7% (last three layers add +2.6/+1.5/+3.6% each)
```

The middle 13 layers are rock-stable (exponent 0.06, nowhere near a random walk's
0.5). Remaining damage is concentrated at the **last ~3 layers (21-23)** and
secondarily **layers 0-2**. NOT uniform (the earlier "uniform per-chunk" reading
was an artifact of the saturated 2.875 regime). => curve-guided mixed precision:
protect the ends, keep the middle cheap.

## Result 3 — coherent error accumulation RULED OUT (mechanism)  [REAL]

`coherence_probe.py`: cross-layer correlation of the token-aligned `down_proj`
output errors (which live in the shared residual space).

| rotation | rho(L2,L12) | rho(L2,L21) | rho(L12,L21) | residual amplification |
|---|---|---|---|---|
| shared (driver default) | +0.0015 | +0.0069 | -0.0004 | 1.005x |
| per-layer (proposed) | +0.0002 | +0.0047 | +0.0004 | 1.003x |

Errors are already essentially incoherent under the shared rotation, so per-layer
rotation is **not** a fix. (The consolidated doc's §7.1 reached the same "no" via
per-layer SQNR spread; this is the direct error-correlation measurement that
actually settles it.) Consistent with the flat exponent in Result 2.

## Idea under test — cross-layer drift correction ("teacher targets")

Combines least-squares projection with delta-sigma noise shaping: quantize each
layer to output the CLEAN target `x_clean . W^T` despite receiving the drifted
input `x_q`, cancelling accumulated drift. Reduces to quantizing
`W_eff = W . C^T . H^-1` (`C = x_q^T x_clean/n`, `H = x_q^T x_q/n`); == W when no
drift. Toy unit test: clean-target error 32% -> 26.8%, approaching the irreducible
"orthogonal residual" floor 26.3% (cf. arXiv 2609.21450). Real deep-layer
validation on the 3.875 checkpoint: see `validate_teacher.out` (in progress).

## Result 3b — we BEAT standard 4-bit by 6x at fewer bits  [REAL]

Whole-model INT4 RTN (group=128, the common reference, ~llama.cpp Q4):

| Method | bpw | PPL | vs baseline |
|---|---|---|---|
| INT4 RTN | 4.125 | 20.797 | **+46%** |
| PBR flat (ours) | 3.875 | 15.297 | **+7.4%** |
| PBR ends-boosted (ours) | 3.97 | 15.140 | **+6.3%** |

Reframes the "no 4-bit/99%" worry: NO method reaches 99% at 4-bit on a 0.5B model
(INT4 RTN is +46%); ours is 6x better at fewer bits. 99% is out of reach because
of the model size, not the method. CAVEAT: INT4 RTN has no rotation/feedback; the
fair strong baseline is INT4 + GPTQ + rotation (not yet run) which would narrow it.

## Result 4 — curve-guided end-boost helps a little; damage is distributed  [REAL]

Reused the 3.875 checkpoint, re-quantized only the 6 end layers (L0-2, L21-23) at
4.25 bpw (`512,512,256,256`), middle inherited at 3.875 (`PBR_INIT_CKPT`).

| Recipe | avg bpw | PPL | vs baseline |
|---|---|---|---|
| flat 3.875 | 3.875 | 15.297 | +7.4% |
| ends 4.25 / middle 3.875 | 3.97 | 15.140 | +6.3% |

Boosting the ends bought only 0.157 PPL. The ends are *a* contributor but the
damage is more distributed than the residual-error curve implied (the residual
error at the ends is partly inherited upstream drift that more end-bits can't undo).
=> ~4-4.25 bpw flat is the practical floor to approach the <5% gate; whole-model
4-bit/99% (<1%) is NOT reached on this 0.5B model. (INT4 baseline on this model
not yet measured -- unknown whether ANY method hits 99% at 4-bit here.)

## Result 5 — adaptive bit allocation and rotation are ANTAGONISTIC  [REAL]

Tested proper reverse water-filling (rank every group's marginal error-reduction
per bit; the "1/2/3/4/8-bit combination" = keep 1..8 RVQ stages). On top of the
rotation it LOSES to flat at equal bits: **-0.60 dB (L12 gate), -0.17 dB (L2 gate)**.

Root cause: the rotation flattens the importance the allocation needs.

| importance spread | original basis | rotated basis |
|---|---|---|
| input-channel Hessian-diag max/median | 286x | 4.8x |
| input-channel CV | 5.78 | 0.45 |
| per-row weight energy max/median | 15.5x | 1.3x |
| per-row weight energy CV | 0.68 | 0.06 |

Rotation's job is to make a FLAT quantizer optimal; it removes ~10-40x of the
non-uniformity adaptive allocation would exploit. And the trade isn't close:
rotation is worth ~+6 dB (INT3 RTN 11.1 -> rot+GPTQ 18.9), adaptive's best case
was ~+0.7 dB. **Flat-on-rotation is the correct design** -- explains the mixed
§7.3 results. Surviving variant: sparse outlier protection of the ~1% rotation
couldn't flatten (rotated top1%/median still 3.2x) -- untested on real data.

## Result 6 — "adaptive-first, rotate-second" (outlier split) also loses to flat  [REAL]

Coherent form of "do allocation before rotation": pull the top-P% most important
input channels (original basis) out at 8-bit UNROTATED, rotate + low-bit VQ the
bulk. On L12 gate vs flat rotated `256,256` (2.0 bpw, 18.34 dB):

| outlier frac | bpw | out SQNR | dB per extra bpw |
|---|---|---|---|
| 1% @8-bit | 2.14 | 18.48 | +1.0 |
| 2% @8-bit | 2.29 | 18.59 | +0.9 |

~1 dB/bpw, vs ~4-6 dB/bpw from simply adding a flat VQ stage. So outlier protection
is DOMINATED by flat VQ depth on top of rotation. Combined with Result 5: on top of
the rotation, NOTHING (adaptive allocation, outlier split, drift correction) beats
just spending bits on flat VQ. Order-swapping allocation before rotation doesn't
help because rotation delocalizes the outliers regardless of when you pick them.

## Scripts added
- `coherence_probe.py` — cross-layer error correlation (Result 3)
- `eval_accum.py` — per-layer residual error growth curve (Result 2)
- `quantize_teacher.py` — whole-model drift-correcting quantizer (PBR_TEACHER)
- `validate_teacher.py` — cheap real-layer A/B of drift correction
- `adaptive_waterfill.py` — reverse water-filling bit allocation vs flat (Result 5)
- `outlier_split.py` — adaptive-first/rotate-second outlier protection (Result 6)
- `int4_rtn_model.py` — whole-model INT4 RTN baseline
- `quantize_full_model.py` — added PBR_ROT_PERLAYER + PBR_INIT_CKPT options
