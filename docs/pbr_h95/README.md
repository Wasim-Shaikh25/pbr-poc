# PBR-H95 — Adaptive Mantissa Quantization

Lossy path: deterministically quantize BF16 mantissas to `k` retained bits, then store the quantized checkpoint exactly. Goal (later phases): minimize BPW subject to **≥95% of BF16 model-quality score** on held-out eval — not weight similarity.

Guide: `PBR_H95_Adaptive_Mantissa_Quantization_Guide.pdf`

## Phase A

- `pbr_h95/quantize.py` — round-to-nearest mantissa keep-bits; Inf/NaN exact; signed zeros exact
- `pbr_h95/policy.py` — first-experiment map (emb/lm/norm/bias/first/last → 7; large attn/MLP → 5)
- `scripts/run_pbr_h95_phase_a.py` — full-model policy probe + uniform ladder k=7..3
- Artifacts: `artifacts/pbr_h95/phase_a_qwen.{json,md}`

### Qwen2.5-0.5B-Instruct (measured)

| Mode | Est. total BPW | Notes |
| ---: | ---: | --- |
| Policy (7/5) | **9.29** | avg mant bits 5.67; RAW pack + 2.62 exp ref |
| Uniform k=7 | 10.62 | lossless mantissa reference |
| Uniform k=5 | 8.62 | |
| Uniform k=3 | 6.62 | |

Quantize idempotent on all tensors: **True**.

### Honesty (Phase A)

- Phase A does **not** measure ≥95% quality — that needs held-out eval (B/C).
- `est_total_bpw` is RAW mantissa packing + reference exp rate, not a physical H95 container.
- Not a ≤4 BPW product claim. Lossless PBR-E reference remains ~10.6 BPW.

## Phase B (layer / tensor-family sensitivity)

- `pbr_h95/eval_nll.py` — mean token NLL / PPL under teacher forcing
- `pbr_h95/apply_policy.py` — apply keep-bits maps onto a live BF16 state_dict
- `pbr_h95/policy.py` — family helpers (`embed`, `norm`, `attn_mid`, `mlp_mid`, `first_block`, `last_block`; `lm_head_tied` ≡ embed when tied)
- `scripts/run_pbr_h95_phase_b.py` — calibration sweep + utility ranking (`--calib` selects corpus)
- Calibration corpora:
  - `docs/pbr_h95/calibration_v1.json` (`calib-v1`, historical; ~155 scored tokens @ max_length=128)
  - `docs/pbr_h95/calibration_v2.json` (`calib-v2`, enlarged; includes all v1 texts + longer passages; target ≥1500 scored tokens @ max_length=256)
- Artifacts: `artifacts/pbr_h95/phase_b_qwen.{json,md}` (v1) and `artifacts/pbr_h95/phase_b_qwen_calib_v2.{json,md}` (v2)

### Metric (honest labeling)

- Score = mean token NLL on the chosen calib corpus; `ppl = exp(mean_nll)`.
- `ppl_retention = bf16_ppl / quant_ppl` (reported for ranking only).
- **Calibration proxy only — not held-out eval.** Phase C owns the ≥95% held-out gate.
- Do **not** claim ≥95% GO from Phase B numbers.

### Non-claims

- No held-out ≥95% quality acceptance from Phase B.
- No ≤4 BPW product claim.
- Utility rankings are calibration-local and may not transfer to held-out sets.

## Phase C (greedy precision allocation + held-out gate)

- `scripts/run_pbr_h95_phase_c.py` — greedy global precision allocation over layer-band (or family) units
- Held-out corpus: `docs/pbr_h95/heldout_v1.json` (`heldout-v1`; **disjoint** from calib; never used for search)
- Search uses **calib-v2 only**; final **GO/NO-GO** uses **heldout-v1** `ppl_retention ≥ 0.95`
- Artifacts: `artifacts/pbr_h95/phase_c_qwen.{json,md}`

### Honesty (Phase C — must read)

- Held-out set is small/in-repo — **not** a production LM benchmark (no MMLU/HellaSwag/etc.).
- ppl_retention on heldout-v1 is a **proxy** quality score for this PoC.
- Claim GO only if heldout ppl_retention ≥ 0.95 AND map is reported; if <0.95, say NO-GO / needs restore.
- Not a physical container BPW; est only.
- Not ≤4 BPW product claim.

### Search sketch

1. Start all-protect keep=7 (BF16 mantissa).
2. Decision units: mid-MLP / mid-Attn layer bands (layers 1–7, 8–15, 16–22) when runtime allows; else family-level `mlp_mid` / `attn_mid`.
3. Greedy: lower one unit by one candidate keep (6→5→4) maximizing `bytes_saved / max(Δnll, 1e-6)` subject to calib `ppl_retention ≥ 0.97`.
4. Freeze map; evaluate held-out; also report policy_7_5 and uniform_mid_k{6,5,4} on held-out for Pareto reference.

## Run

```bash
# Phase A (storage / numeric probe)
PYTHONPATH=. python scripts/run_pbr_h95_phase_a.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct

# Phase B on calib-v1 (historical)
PYTHONPATH=. python scripts/run_pbr_h95_phase_b.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --max-length 128

# Phase B on calib-v2 (enlarged)
PYTHONPATH=. python scripts/run_pbr_h95_phase_b.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --calib docs/pbr_h95/calibration_v2.json \
  --max-length 256 \
  --out artifacts/pbr_h95/phase_b_qwen_calib_v2.json \
  --md artifacts/pbr_h95/phase_b_qwen_calib_v2.md

# Phase C (greedy + held-out gate)
PYTHONPATH=. python scripts/run_pbr_h95_phase_c.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --calib docs/pbr_h95/calibration_v2.json \
  --heldout docs/pbr_h95/heldout_v1.json \
  --max-length 256 \
  --units bands
```

## H95E (vocabulary / frequency-aware embedding mantissa)

Stacks **on top of Phase C** (mid bands keep=4, protected emb/norm/bias/first/last @7) and only varies **embedding rows**.

- `pbr_h95/embed_tiers.py` — inventory, calib token-frequency counts, tier assignment, per-row embed quantize
- `scripts/run_pbr_h95e.py` — E1 inventory + uniform embed ladder + E2 frequency tiers; calib+heldout PPL
- Artifacts: `artifacts/pbr_h95/h95e_inventory.{json,md}`, `artifacts/pbr_h95/h95e_qwen.{json,md}`

### E1 / E2 sketch

1. **Inventory** — embed shape/share, tie status, configured vs reachable vocab, padded-row norms (not zero), class histogram.
2. **Uniform ladder** — Phase C body fixed; all embed rows at keep ∈ {7,6,5,4,3}; report calib+heldout retention and est BPW.
3. **Frequency tiers** — fit token counts on **calib-v2 only**; specials/top-mass → higher keep; unseen/padded lower; try default / aggressive / conservative schedules.

### Honesty (H95E — must read)

- Proxy PPL on in-repo calib/heldout only — **not** a multilingual/production bench.
- Frequency tiers fit on calib-v2 — risk of calib overfitting for rare tokens.
- Padded-row cleanup ≠ meaningful BPW win (padded rows are nonzero; omitting them from a storage estimate is bookkeeping).
- `est_total_bpw` is not a physical container (1 sign + 2.62 exp ref + avg mantissa keep).
- Not claiming ≤8 BPW product unless heldout retention ≥ 0.95 **and** numbers support it — if heldout ≥ 0.95 say **proxy GO** for that map; else **NO-GO**.
- English-heavy calib ≠ multilingual retention proof (Qwen is multilingual).

### Run

```bash
PYTHONPATH=. python scripts/run_pbr_h95e.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --calib docs/pbr_h95/calibration_v2.json \
  --heldout docs/pbr_h95/heldout_v1.json \
  --max-length 256
```

## H95Q (mixed-precision base + sparse recovery path)

Guide: `PBR_H95Q_Mixed_Precision_Sparse_Recovery_Guide.pdf`

Stage 1: quality-controlled mantissa keep \(K \in \{7,6,5,4,3\}\) (idempotent Q). Stage 2: exact coding of the **quantized reference** (original BF16 need not round-trip). Target ~8 total BPW ≈ 1 sign + 2.62 exp + ≤4.38 mant(+maps). Primary lever is K; codecs only if they beat packed-K.

### This PoC (first recommended run §18)

- `pbr_h95/policy.py` — named maps `h95q_A_7_5`, `h95q_B1_mlp_k4`, `h95q_B2_embed_k5`
- `pbr_h95/packed_rate.py` — packed mantissa bits + `packed_K_total_bpw` (vs entropy-lb)
- `pbr_h95/entropy_diag.py` — H(retained symbols), bit-plane bias, simple rANS table-cost estimate
- `scripts/run_pbr_h95q.py` — run A/B1/B2 (+ optional `--with-c-sketch`), calib-v2 + heldout-v1
- Artifacts: `artifacts/pbr_h95/h95q_qwen.{json,md}`

| Candidate | Precision design |
| --- | --- |
| **A** | Existing 7/5 policy; packed retained bits; reproduce ~9.29 BPW |
| **B1** | mlp_mid K5→K4; attn_mid stays K5; emb/norm/bias/first/last @K7 |
| **B2** | embed @K5; attn_mid @K6; mlp_mid @K5; norm/bias/first/last @K7 |
| **C sketch** (optional) | mid @K4 + restore top-magnitude % mlp rows to K7 (+ row bitmap cost) |

### Honesty (H95Q — must read)

- Proxy PPL on in-repo calib-v2 / heldout-v1 only — **not** a production LM benchmark.
- heldout ppl_retention ≥ 0.95 ⇒ **proxy GO** for that map; else **NO-GO**. Not MMLU/etc.
- Separate **packed_K total BPW** from **entropy lower-bound BPW**. Do not claim physical container size unless real bytes are written.
- Reject rANS / advanced codecs unless complete cost (incl. tables) beats packed K.
- Not a ≤8 BPW product claim unless packed rate **and** proxy quality both support it.
- Full Candidate C/D codecs (matrix residuals, pair dict) are out of scope unless A/B leave clear headroom.

### Run

```bash
PYTHONPATH=. .venv/bin/python scripts/run_pbr_h95q.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --calib docs/pbr_h95/calibration_v2.json \
  --heldout docs/pbr_h95/heldout_v1.json \
  --max-length 256 \
  --with-c-sketch
```


## H95Q stack follow-up (Track1 / Track2 / Track3)

Follow-ups on `feat/pbr-h95q-stack` stacking toward packed ≤8 with heldout ≥0.95 (proxy dual-gate):

| Track | Idea |
| --- | --- |
| **T1** | B1 body + H95E embed frequency tiers (default / aggressive / conservative), fit calib-v2 only |
| **T2** | Proper Candidate C: aggressive body (mlp@K4, attn@K4) + sparse |w|-magnitude row/channel recovery to K7; **map_bpw charged** (bitmap vs absolute indices, auto=min) + correction bits |
| **T3** | Selective mlp_mid@K3 (all / bands 1–7, 8–15, 16–22 / robust-50% low-|w| rows) |
| **Stacks** | B1+embed tiers+band K3; C with K3 base + sparse K7 recovery |

- Code: `pbr_h95/h95q_stack.py`, `scripts/run_pbr_h95q_stack.py`
- Artifacts: `artifacts/pbr_h95/h95q_stack_qwen.{json,md}`
- Packed = 1 + 2.62 + avg_K (+ map for C). Not a physical container. Dual-gate = packed≤8 **and** heldout≥0.95.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_pbr_h95q_stack.py   --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct   --calib docs/pbr_h95/calibration_v2.json   --heldout docs/pbr_h95/heldout_v1.json   --max-length 256
```

## H95Q physical container (freeze)

Independent-decode proof for three freeze candidates on Qwen2.5-0.5B-Instruct:

| Candidate | Policy | Prior *estimated* packed BPW |
| --- | --- | ---: |
| **H95Q-Conservative** | B1 + conservative embed tiers | ~7.89 |
| **H95Q-Balanced** | B1 + aggressive embed (**no** mid-MLP K3) | ~7.61 |
| **H95Q-S1** (primary) | B1 + aggressive embed + mlp band 8–15 @K3 | ~7.40 |

### Format (`H95Q` v1)

- Global header: magic, version, model id/hash, dtype=bf16, n_tensors, total_weights, endianness, checksum=SHA-256
- Per-tensor: name, shape, layout, base_keep / mode, optional embed row-tier table, optional MLP K3 band mask
- Packed streams: sign bitplane (1 bit/w), **raw u8 exponents** (exactness), K-bit mantissa streams (K∈{3..7})
- Precision map + SHA-256 of quantized reference state; absolute offsets for seeking

**Exponent choice:** raw 8-bit/weight (not the ~2.62 BPW entropy reference used in estimated packed BPW). Physical `actual_bpw` is therefore expected to exceed estimated packed BPW by ~5.38 BPW on the exp term alone, before metadata.

### Code

- `pbr_h95/bitpack.py` — K-bit pack/unpack (MSB-first)
- `pbr_h95/container_h95q.py` — encode/decode API + `python -m pbr_h95.container_h95q` decode-only CLI
- `scripts/run_pbr_h95q_container.py` — build three candidates, encode, decode-verify, report
- Artifacts: `artifacts/pbr_h95/h95q_container_qwen.{json,md}`; containers under `artifacts/pbr_h95/containers/*.h95q` (**gitignored** if large)

### Exactness gates

- Q idempotent on samples
- `decode(encode(Q(W))) == Q(W)` uint16 bit-identical for every tensor
- SHA-256(quantized reference) == SHA-256(decoded)
- Fail hard on mismatch

### Honesty

- `actual_bpw = file_bytes * 8 / n_weights` from the physical file
- Compare to prior estimated packed BPW; call out metadata overhead
- Still **not** claiming production quality / multilingual / runtime RAM == BPW
- If S1 `actual_bpw > 8` due to raw-u8 exponents (+ metadata), say so clearly — container still proves exact independent decode

```bash
PYTHONPATH=. .venv/bin/python scripts/run_pbr_h95q_container.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --calib docs/pbr_h95/calibration_v2.json
```

## H95Q container v2 (adaptive exact exponents)

Same frozen S1/Conservative/Balanced precision policies and **identical** Q(W)
uint16 words as v1 (`sha256_quantized_reference` unchanged). Only the exponent
stream changes: per-tensor competition among `EXP_RAW8` / `EXP_RANS` /
`EXP_HUFFMAN` / `EXP_DELTA_RANS` / `EXP_RUN_RANS` by **complete physical
section bytes** (payload + table + mode_id + length fields + final rANS state).

| Candidate | v1 actual BPW | v2 actual BPW | exp complete BPW | ≤8? |
| --- | ---: | ---: | ---: | --- |
| H95Q-Conservative | ~13.29 | **~7.91** | ~2.62 | PASS |
| H95Q-Balanced | ~13.01 | **~7.63** | ~2.62 | PASS |
| H95Q-S1 | ~12.80 | **~7.42** | ~2.62 | **PASS** |

### Code

- `pbr_h95/exp_codec.py` — RAW8 / Huffman / rANS / delta / run encode+decode + cost helpers
- `pbr_h95/container_h95q.py` — version 2 wire format; decoder still reads v1
- `scripts/run_pbr_h95q_container.py --version 2` (default; `--from-v1` transcode)
- Tests: `tests/test_h95q_container_v2.py` (gates E1–E4)
- Artifacts: `artifacts/pbr_h95/h95q_container_v2_qwen.{json,md}`; `*-v2.h95q` gitignored

### Honesty

- Physical BPW from file size only; exp section bytes == Σ `exp_len`
- Quality retention still prior stack proxy (≈0.990) — not re-evaluated
- Not production / multilingual / RAM == BPW

```bash
PYTHONPATH=. .venv/bin/python scripts/run_pbr_h95q_container.py --version 2 \
  --candidates H95Q-Conservative H95Q-Balanced H95Q-S1
```

## S1 + selective 256-node X/Y (post-codec)

Follow-up to the all-tile X/Y experiment (PR #15, 7.453890 BPW). Re-encode
the frozen S1 quantized mantissa codes; **packed-K is the product default**.
X/Y is stored only when it is strictly smaller including map/flag bits.
No 16×16 padding. Does not change Q(W). See `docs/pbr_q4/README.md`.

```bash
PYTHONPATH=. python scripts/run_pbr_q4_s1_xy.py \
  --s1-container artifacts/pbr_h95/containers/H95Q-S1-v2.h95q
```

