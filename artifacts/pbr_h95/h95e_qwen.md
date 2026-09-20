# PBR-H95E — Embedding mantissa compression (on Phase C body)

Model: `Qwen/Qwen2.5-0.5B-Instruct` (local BF16, CPU)
Body: Phase C frozen = uniform mid keep=4, protected@7 (emb/norm/bias/first/last)
Calibration: **calib-v2** tokens=1925, max_length=256
Held-out: **heldout-v1** tokens=1094, max_length=256

## Honesty

- Proxy PPL on in-repo calib/heldout only — not a multilingual/production bench.
- Frequency tiers fit on calib-v2 — risk of calib overfitting for rare tokens.
- Padded-row cleanup ≠ meaningful BPW win (padded rows are nonzero; omitting them from storage estimate is bookkeeping).
- est BPW is not a physical container (1 sign + 2.62 exp ref + avg mantissa keep).
- Not claiming ≤8 BPW product unless heldout retention ≥ 0.95 AND numbers support it — if heldout ≥ 0.95 say proxy GO for that map; else NO-GO.
- English-heavy calib ≠ multilingual retention proof (Qwen is multilingual).
- Body = Phase C frozen map (mid bands keep=4, emb/norm/bias/first/last protected@7) with only embed rows varied.

## Gate (best tier / primary map)

- Primary map: **tier_aggressive**
- heldout ppl_retention vs BF16: **0.9896** (threshold 0.95)
- calib ppl_retention vs BF16: **0.9883**
- est_total_bpw (omit padded storage): **7.53** (Phase C was **8.63**)
- **Decision: GO**

## E1 Uniform embed ladder (Phase C body fixed)

| embed_keep | calib ret | heldout ret | Δnll calib | est BPW (omit pad) | est BPW (full) | wall s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.9933 | 0.9934 | +0.006750 | 8.63 | 8.63 | 24.39 |
| 6 | 0.9890 | 0.9859 | +0.011055 | 8.35 | 8.35 | 28.68 |
| 5 | 0.9925 | 0.9890 | +0.007517 | 8.08 | 8.08 | 27.05 |
| 4 | 0.9849 | 0.9914 | +0.015211 | 7.80 | 7.80 | 27.54 |
| 3 | 0.9875 | 0.9806 | +0.012570 | 7.53 | 7.53 | 27.24 |

## E2 Frequency-tiered embed keeps

| schedule | calib ret | heldout ret | avg embed row keep | est BPW (omit pad) | vs Phase C BPW | wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tier_default | 0.9822 | 0.9898 | 4.0204 | 7.81 | -0.82 | 27.92 |
| tier_aggressive | 0.9883 | 0.9896 | 3.0207 | 7.53 | -1.10 | 27.63 |
| tier_conservative | 0.9813 | 0.9914 | 4.0380 | 7.81 | -0.82 | 28.0 |

### Tier band summaries

**default** — band row counts: `{'special': 14, 'top': 1000, 'band2': 162, 'band3': 0, 'seen_rest': 0, 'unseen': 150489, 'padded': 271}`
- keeps_by_band: `{'special': 7, 'top': 7, 'band2': 6, 'band3': 5, 'seen_rest': 4, 'unseen_reachable': 4, 'padded': 3}`
- keep_hist_rows: `{'0': 0, '1': 0, '2': 0, '3': 271, '4': 150489, '5': 0, '6': 162, '7': 1014}`
- top_cut_n=1000, calib_tokens=1983, unique_seen=1162

**aggressive** — band row counts: `{'special': 14, 'top': 500, 'band2': 100, 'band3': 397, 'seen_rest': 165, 'unseen': 150489, 'padded': 271}`
- keeps_by_band: `{'special': 7, 'top': 7, 'band2': 6, 'band3': 5, 'seen_rest': 3, 'unseen_reachable': 3, 'padded': 3}`
- keep_hist_rows: `{'0': 0, '1': 0, '2': 0, '3': 150925, '4': 0, '5': 397, '6': 100, '7': 514}`
- top_cut_n=500, calib_tokens=1983, unique_seen=1162

**conservative** — band row counts: `{'special': 14, 'top': 2000, 'band2': 0, 'band3': 0, 'seen_rest': 0, 'unseen': 149651, 'padded': 271}`
- keeps_by_band: `{'special': 7, 'top': 7, 'band2': 6, 'band3': 5, 'seen_rest': 5, 'unseen_reachable': 4, 'padded': 3}`
- keep_hist_rows: `{'0': 0, '1': 0, '2': 0, '3': 271, '4': 149651, '5': 0, '6': 0, '7': 2014}`
- top_cut_n=2000, calib_tokens=1983, unique_seen=1162

## Phase C reference (embed@7)

- Phase C heldout ret: **0.9934**
- Phase C est BPW: **8.63**

## BF16 baselines

- calib mean_nll=3.160408, ppl=23.5802
- heldout mean_nll=3.664400, ppl=39.0327

## Shortcuts / notes

- Tier head uses top_n_cap rank floor (expanded only if mass < target); avoids 1-id head when one token exceeds 1% mass on tiny calib.
- Calib-v2 sees only ~1.2k unique ids; vast majority of reachable rows are never-seen → tier maps ≈ uniform embed keep for that unseen band.
- Wall time: **239.25s**
