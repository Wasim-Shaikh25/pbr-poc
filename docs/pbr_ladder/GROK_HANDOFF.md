# PBR-Ladder operational handoff (Phase 4 through PR #33)

**Date:** 2026-09-22  
**Status:** **STOPPED.** User said **Stop all** (~16:07 IST). Do not auto-resume compute.

This file is the current operational handoff for a fresh session. It replaces the stale Claude→Grok Phase 0–2 intro. Detail logs live in `docs/pbr_ladder/README.md` and `artifacts/pbr_ladder/`; roadmap scope stays in `docs/pbr_ladder/PBR_Ladder_Whats_Next.docx`. Read this first, not the full PR history.

---

## Snapshot (what is true now)

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` BF16 |
| Eval | WikiText-2 raw v1 test, **299078** tokens via `pbr_ladder/eval_ppl.py` (`PBR_CPU_FP32=1`) |
| Baseline PPL | **14.247** |
| Phase 2 / Phase 4 gate | **≤14.959** (+5%) |
| Latest full-24 PPL | **18.492** (PR #33) — **+3.533** over gate |
| Avg index bpw @ #33 | ~**2.849** |
| Phase 4 done? | **No** |
| L8–12 all-bump run | **Cancelled mid-flight — no result** |

`quantize_full_model.py` saves a **dequantized FP** checkpoint for PPL only. It is **not** a smaller packed file. Size wins are a later phase. Do not claim size from these checkpoints.

### Recipes

| Name | Stages | Index bpw (approx) |
| --- | --- | ---: |
| Default | `256,256,64` | ~2.75 |
| Bumped | `512,256,64` | ~2.875 |

---

## Proven milestones (on `main`)

| Milestone | Result |
| --- | --- |
| Phase 0 | **PASS** — 9-tensor dB table used real activations |
| Phase 1 L12 all mats | **PASS** — 14.247 → **14.354** (+0.75% &lt;1%) |
| Phase 2 flat all-24 @ default | **FAIL** — **19.427** (+36.4%) |
| L0/L1/L2 isolation (#25) | Nearly additive; no single villain |
| L0–2 @ bumped only, rest FP (#26) | **PASS** 14.873 |
| Full-24 early bump L0–2 (#27) | **FAIL** **19.280** |
| L3–23 four-block map (#28/#30) | Alone Δ%: L19–23 hottest (+8.4%), then 13–18 (+5.7%), 3–7 (+4.6%), 8–12 (+3.7%) |
| Protect L19–23 full-24 (#31) | **FAIL** **19.039** (−0.24 vs early-mix) |
| Also protect L13–18 (#32) | **FAIL** **18.803** (−0.24 vs #31) |
| Also protect L3–7 (#33) | **FAIL** **18.492** (−0.311 vs #32); avg index ~**2.849** bpw; **+3.533** over gate |

### Full-24 stack (same gate each time)

flat **19.427** → early-mix **19.280** → +late **19.039** → +L13–18 **18.803** → +L3–7 **18.492**. Gate still far.

---

## Stopped work (important)

- User: **Stop all** on **2026-09-22 ~16:07 IST**.
- Cloud agent for **L8–12 / all-24 @ `512,256,64`** was **cancelled mid-flight**.
- **No result** from that run. Do **not** claim all-bump / uniform-`512,256,64` numbers.
- Do **not** auto-resume that run unless the user explicitly asks.

---

## Locked Phase 4 iterate (Wasim via GitHub) — status

Protect hot chunks at `512,256,64` in map order, remasure full-24 each time:

1. L19–23 → **done** (#31)
2. L13–18 → **done** (#32)
3. L3–7 → **done** (#33)
4. L8–12 → **started then cancelled** — unfinished (last map chunk at this rung)

Expectation from the #33 agent: even finishing L8–12 at this rung likely will **not** clear the remaining **+3.533**. If/when that PR is resumed and merges, **stop** inventing the next recipe without a new locked step from the user.

---

## Explicit non-retries (user)

- No KLT / shared rotation retries (transform storage tax).
- Don’t bother weak per-channel scale/permute for a ~30%+ hole.
- Outlier correction only later, aimed at concentrated damage — not as global bit-reduction now.

---

## Product / strategy (short)

- Compressed file ≠ lower RAM unless packed-resident or disk tunnel.
- vs PrismML Ternary Bonsai / GGUF: Bonsai wins tiny fixed SOTA; the wedge is BYO weights + tunnel. The core puzzle is a **low-bit recipe the model absorbs**, not more packing of noisy codes.
- Do **not** claim size wins from dequant checkpoints.

---

## Claims discipline

Only claim what the current phase measured. **Phase 4 is not done.**

Fair claim now: selective `512,256,64` bumps on most layers still leave full-24 PPL ~**18.5** vs **14.25** baseline (~**+30%**).

Do **not** claim: whole-model gate pass, packed-file size wins, beating GGUF/AQLM/QuIP#/Bonsai, or any all-24 @ uniform `512,256,64` result.

---

## Immediate next steps (wait for user)

1. Optionally finish the last map chunk: all L0–23 @ `512,256,64`, remasure vs 14.959 (honest **FAIL** expected; closes the rung).
2. Then **stop and decide** a new locked recipe (higher stages / different method / outlier correction) — do not invent unilaterally.
3. Later roadmap phases unchanged: quality broaden (3), adaptive bits (4 formal), multi-quality (5), baselines (6), packed runtime (7), prior art (8).

---

## Repo pointers

| What | Where |
| --- | --- |
| Quantize / eval / acts | `pbr_ladder/quantize_full_model.py`, `eval_ppl.py`, `capture_activations.py`, `push_below4.py`, … |
| Long-form results | `docs/pbr_ladder/README.md` |
| Roadmap | `docs/pbr_ladder/PBR_Ladder_Whats_Next.docx` |
| Phase logs | `artifacts/pbr_ladder/` (where present on `main`) |

GitHub agent owns review/merge under Wasim policy (**honest FAIL OK to merge**).
