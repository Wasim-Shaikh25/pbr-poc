# Blocker-fix ablation (same 42-tensor Qwen set)

Blocker-fix ablation on the Stage 1B Qwen tensor set. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW.

| profile | orig B | enc B | BPW | ratio | exact | winning modes | zlib B |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| pbre | 167673856 | 113900442 | 10.8688 | 0.6793 | PASS | bf16_exp_huffman:42 | 133488038 |
| zlib_baseline | 167673856 | 133488038 | 12.7379 | 0.7961 | n/a (not PBR) | zlib (baseline, not PBR) | 133488038 |
| new_modes | 167673856 | 113900652 | 10.8688 | 0.6793 | PASS | bf16_exp_huffman:42 |  |
| uint16_spatial | 167673856 | 176866107 | 16.8772 | 1.0548 | PASS | raw_bf16:327488 |  |
