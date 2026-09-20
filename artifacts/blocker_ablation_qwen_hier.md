# Blocker-fix ablation (Qwen/Qwen2.5-0.5B-Instruct / qwen_hier)

Hierarchical leftovers vs PBR-E on the Stage 1B Qwen tensor set. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW.

| profile | orig B | enc B | BPW | ratio | exact | winning modes | zlib B |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| hierarchical | 167673856 | 113900778 | 10.8688 | 0.6793 | PASS | bf16_exp_huffman:42 |  |
