# Blocker-fix ablation (unsloth/Llama-3.2-1B-Instruct / llama)

Llama-family method validation only. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW.

| profile | orig B | enc B | BPW | ratio | exact | winning modes | zlib B |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| pbre | 167772160 | 113694055 | 10.8427 | 0.6777 | PASS | bf16_exp_huffman:11 | 133302428 |
| zlib_baseline | 167772160 | 133302428 | 12.7127 | 0.7945 | n/a (not PBR) | zlib (baseline, not PBR) | 133302428 |
| hierarchical | 167772160 | 113694143 | 10.8427 | 0.6777 | PASS | bf16_exp_huffman:11 |  |
