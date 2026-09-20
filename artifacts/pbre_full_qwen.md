# Full-checkpoint PBR-E (Qwen2.5-0.5B-Instruct)

Every 16-bit tensor. Profile `pbre_whole` (raw + `bf16_exp_huffman` only).
Lossless BF16. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW.

- repo: `Qwen/Qwen2.5-0.5B-Instruct`
- revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- tensors: **290 / 290** 16-bit
- original bytes: **988065536** (942.3 MiB of BF16 weights)
- encoded bytes: **659740226**
- BPW: **10.6833**
- ratio vs raw BF16: **0.6677**
- exact: **PASS** (uint16 / SHA-256 per tensor)
- winning mode: `bf16_exp_huffman` (7779 tiles; one Huffman table per stripe because the on-wire tile header stores rows/cols as uint16)
- encode: 149.2 s, **6.62 MB/s**
- decode: 133.2 s, **7.42 MB/s**

This is the DF11/ZipNN-class bar on the complete 0.5B Instruct checkpoint, not a novel rate and not ≤4 BPW.
