# Zoo vs PBR-E: why 10.585 vs 10.616

Zoo vs PBR-E accounting. Held-out NLL is not a bitstream. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW.

## Headline

The mantissa zoo did **not** find a new ~8 BPW codec. It found the DF11 move already used on exponents, applied to the mantissa: a 256×128 exp-conditional table plus a 1.6 KB GBDT. Codec-view total **10.5850 BPW** vs synthetic raw-M PBR-E **10.6470** (Δ **-0.0620**) and vs measured PBR-E rANS **10.6161** (Δ **-0.0311**). Toward 8 BPW (2.6 bits missing): **dead-end micro-gain**.

## Who won how many tensors

Per-tensor argmin on held-out mantissa NLL. Only two models were ever selected; the mixture pays the union of their tables (67124 B = 65536 + 1588).

| choice | tensors | orig words | orig bytes | holdout words | holdout share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exp_cond` | 34 / 42 | 81534169 | 163068338 | 16306560 | 97.3% |
| `gbdt` | 8 / 42 | 2302759 | 4605518 | 460544 | 2.7% |

GBDT wins are small attention projections (k/q/v), not the MLP walls. Names: `model.layers.0.self_attn.k_proj.weight`, `model.layers.0.self_attn.q_proj.weight`, `model.layers.1.self_attn.k_proj.weight`, `model.layers.1.self_attn.q_proj.weight`, `model.layers.1.self_attn.v_proj.weight`, `model.layers.2.self_attn.k_proj.weight`, `model.layers.2.self_attn.v_proj.weight`, `model.layers.3.self_attn.k_proj.weight`.

CTW, PPM, tiny AR, and IDF-lite won **0** tensors in the mixture.

## Exact breakdown

| piece | zoo mixture (codec-view) | PBR-E split (raw M) | PBR-E measured rANS |
| --- | ---: | ---: | ---: |
| sign | 1.0000 raw (10479616 B) | 1.0000 raw | 1.000 in packed SM |
| exponent | 2.6469 NLL+table (27738658 B) | 2.6469 NLL+table | ~2.6161 bitstream (27416101 B w/ headers) |
| mantissa | 6.9380 NLL+table (72707503 B) | 7.0000 raw (73357312 B) | 7.000 raw |
| packed SM | (sign+mant separate) | (sign+mant separate) | 83836928 B |
| tables | 67124 B (0.0064 BPW, in mant) | 0 (M raw) | exp table in exp stream |
| headers | 0.0001 (714 B) | 0.0001 | tile + JSON |
| **total** | **10.5850** (110926491 B) | **10.6470** (111576300 B) | **10.6161** (111253029 B) |

Measured PBR-E rANS container: **111253029 B** / 83836928 words = 10.6161 BPW (entropy bound 10.6127).

## Is 0.03 BPW real or accounting?

The 0.03–0.06 BPW is a real mantissa entropy gap (H(M|exp)≈6.93 vs raw 7), not random noise. It is NOT a like-for-like container comparison: zoo totals use held-out NLL plus amortized tables; PBR-E 10.616 is an on-disk bitstream over all words. Charge the 67 KB tables to the 20% holdout only and the zoo total is ~10.611, within 0.005 of measured PBR-E. Realizing NLL as rANS would add ~0.003 BPW (the exponent overhead).

- Headline **-0.0620 BPW** is vs synthetic split PBR-E (raw-M 7.0), codec-view amortization.
- **-0.0311 BPW** is vs measured PBR-E rANS 10.616, still NLL not bitstream.
- Holdout-charged zoo total **10.6106** vs measured -0.0055 BPW.
- Mantissa ideal gap vs raw 7: **-0.0684** BPW; after amortized tables **-0.0620**.
- Mixture vs H(M\|exp) alone: **+0.0010 BPW** (GBDT is a rounding error).
- Adding the Job-3 rANS overhead to zoo NLL: **10.5884 BPW**.

Zoo scores **held-out last 20% of rows** (`n_hold=16767104`). PBR-E 10.616 encodes **all** `83836928` words. Complete zoo BPW amortizes one global table over the full set (`8 × model_bytes / n_all` added to holdout NLL/n_hold). That is honest as a *codec view*, not as an on-disk file of the holdout.

## Transferable trick vs 8 BPW

Entropy-code the 7-bit mantissa, preferably 256 exp-conditional tables (DF11-class, same move already used on exponents). ~0.06 BPW vs raw-M, ~0.03 vs current PBR-E. Not a new axis.

Bits needed to go from measured PBR-E 10.616 to 8.0: **2.616**. Bits found in the zoo: **0.0620**. Two orders of magnitude short. CTW/AR/IDF/PBR-4 did not add any.

**Verdict:** dead-end micro-gain for an 8 GB → 4 GB bit-exact target. Worth shipping as an optional DF11 mantissa rANS in the PBR-E product (~0.03–0.06 BPW), not as a new research axis.

This is not a 1–2 GB / 8 GB result and not ≤4 BPW.
