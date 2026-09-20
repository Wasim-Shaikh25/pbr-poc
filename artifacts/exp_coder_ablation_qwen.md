# Exponent coder ablation (Qwen/Qwen2.5-0.5B-Instruct)

Exponent coder ablation (canonical Huffman vs rANS). Complete bytes include table + stream + packed sign/mantissa + one tile header. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW.

- revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- tensors: **42**
- words: **83836928**
- original bytes: **167673856**
- H(exp) weighted: **2.6124**

| coder | enc B | BPW | vs bound | exact |
| --- | ---: | ---: | ---: | --- |
| entropy bound (table+stream+SM+hdr) | 111216676 | 10.6127 | 0 | n/a |
| canonical Huffman | 113875515 | 10.8664 | +0.2537 | PASS |
| rANS | 111253029 | 10.6161 | +0.0035 | PASS |

Huffman − rANS: **0.2502 BPW** (2622486 bytes).

rANS is smaller by 0.2502 BPW on this sample. Still DF11-class; not ≤4 BPW.
