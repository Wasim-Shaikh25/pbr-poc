# Path to 8 BPW (bit-exact)

Target: **8 complete BPW** on a standalone dense BF16 checkpoint (8 GB → 4 GB).
PBR-E today is **10.616** rANS / **10.68** Huffman on Qwen2.5-0.5B-Instruct.
That is a **2.6 bit/weight** gap. Not ≤4 BPW. Not a 1–2 GB / 8 GB claim.

Evidence already in this repo: Phase A miss, mantissa zoo micro-gain,
PBR-4 negative (~20.6 BPW), related-checkpoint delta 8.36 **only if the
base is free**.

## Ranked levers (evidence only)

1. **Related-checkpoint residual (already measured ~8.36 delta-only BPW).**
   Qwen-0.5B → Instruct, XOR then rANS, reconstruct PASS. Bundle with a
   shipped base is ~19 BPW. This is the only measured number in the 8 BPW
   band, and it does **not** apply to a standalone checkpoint. Use it when
   the user already has the base.

2. **rANS-code the mantissa with H(M|exp) tables (~0.03–0.06 BPW).**
   Zoo mixture 10.585 vs synthetic raw-M PBR-E 10.647; vs measured PBR-E
   10.616 the gap is ~0.03 and partly accounting (NLL vs bitstream,
   holdout vs full set). Same DF11 move already used on exponents.
   Ship as an optional PBR-E extra, not a research program. Does not
   reach 8 BPW.

3. **Keep exponent rANS, not Huffman (already shipped, +0.25 vs Huffman).**
   Job 3: rANS 10.616 vs Huffman 10.866. Already the product default
   (`pbre_whole` competes Huffman and rANS). No further 2.6 bits here.

4. **Exact duplicates / tying / vocab projections.**
   Value-dict, duplicate-tile, and transformed-ref codecs exist and lost
   the complete-byte contest on these dense Qwen/Llama samples. Revisit
   only on architectures with documented tying or huge embedding overlap,
   with the same complete-byte gate.

5. **Sparsity, MoE experts, activation-gated storage.**
   Not present in the dense Qwen/Llama samples. Do not project 8 BPW from
   them onto dense weights.

6. **Lossy / NF4 / GPTQ / residual quantization.**
   Would leave the bit-exact contract. Out of scope for this product.

## Rejected as 8 BPW paths (measured)

- **PBR-4** structured nibble + node formulas: **20.58 BPW**, 8.59% R=0,
  worse than raw 16. Experimental / negative.
- **Phase A** spatial/layer mantissa contexts and cheap reversibles: gate
  MISS, H(M|exp) still 6.93/7.
- **CTW / PPM / tiny AR / IDF-lite:** sat on the unigram; 0 mixture wins
  except GBDT on small k/q/v slices (noise-scale).

## Honest conclusion

There is **no measured bit-exact lever** that takes a *standalone* dense
LLM from 10.6 BPW to 8 BPW. The missing 2.6 bits are residual mantissa
entropy of trained weights, not a missing tile codec. Ship PBR-E at
~10.6–10.7 BPW (~30% vs raw BF16). Treat related-checkpoint delta as a
separate product when a base is already stored. Do not claim ≤4 BPW.
