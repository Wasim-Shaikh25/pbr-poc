# Adaptive 7-bit mantissa codec (real BF16)

Adaptive 7-bit mantissa codec PoC. Exact uint16 round-trip. Complete bytes include modes, codebook, directory, and padding. Not a ≤4 BPW or ≤8 BPW total claim unless the measured complete BPW is that low. Phase A found real LLM mantissas near 7 BPW; this codec is allowed to confirm that.

Model dir: `outputs/models/Qwen__Qwen2.5-0.5B-Instruct`. Tensors=290  words=494032768.

The guide PDF was not on this agent VM; `predict()` is documented in `pbr_adaptive_mantissa/predict.py` (copy-prev, local pos, ramp, exp, xor, avg).

## Aggregate

- Raw BF16: 942.293 MiB (**16.000 BPW**)
- Adaptive mantissa field: 412.258 MiB (**7.0001 mantissa BPW**)
- Sign raw: 58.893 MiB
- Exp rANS: 154.255 MiB
- Complete bundle: 625.413 MiB (**10.6194 total BPW**)
- vs raw 16: **0.664×**
- vs PBR-E ~10.62: **1.000×** (same complete-byte discipline; PBR-E is sign+mantissa raw + exp entropy).
- Tensor strategies: {'ALL_RAW': 290}
- Winning MIXED/ALL block modes: {'RAW': 1929876, 'HUFFMAN': 0, 'CONTEXT': 0}
- Wall: 192.85s

**Honesty:** adaptive mantissa stays ~7 BPW (mostly RAW). That matches Phase A: real Qwen mantissas are not a compact-context win. This is not ≤4 or ≤8 **total** BPW.

## Synthetic sanity

| case | strategy | mantissa BPW | total BPW | Huffman table | exact |
| --- | --- | ---: | ---: | --- | --- |
| `random` | ALL_RAW | 7.037 | 13.186 | False | True |
| `skewed` | ALL_HUFFMAN | 1.156 | 7.305 | True | True |
| `structured_pos` | ALL_CONTEXT | 1.037 | 7.186 | False | True |

## Per tensor

| tensor | words | strategy | mant BPW | total BPW | huff table B | exact |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `model.embed_tokens.weight` | 136134656 | ALL_RAW | 7.000 | 10.584 | 0 | True |
| `model.layers.0.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.295 | 0 | True |
| `model.layers.0.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.603 | 0 | True |
| `model.layers.0.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.606 | 0 | True |
| `model.layers.0.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.571 | 0 | True |
| `model.layers.0.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 8.812 | 0 | True |
| `model.layers.0.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 17.625 | 0 | True |
| `model.layers.0.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.908 | 0 | True |
| `model.layers.0.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.719 | 0 | True |
| `model.layers.0.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.420 | 0 | True |
| `model.layers.0.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.862 | 0 | True |
| `model.layers.0.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.250 | 0 | True |
| `model.layers.0.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.663 | 0 | True |
| `model.layers.1.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.964 | 0 | True |
| `model.layers.1.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.589 | 0 | True |
| `model.layers.1.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.588 | 0 | True |
| `model.layers.1.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.572 | 0 | True |
| `model.layers.1.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.830 | 0 | True |
| `model.layers.1.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 17.500 | 0 | True |
| `model.layers.1.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.702 | 0 | True |
| `model.layers.1.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.674 | 0 | True |
| `model.layers.1.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.955 | 0 | True |
| `model.layers.1.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.702 | 0 | True |
| `model.layers.1.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.188 | 0 | True |
| `model.layers.1.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.655 | 0 | True |
| `model.layers.10.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.777 | 0 | True |
| `model.layers.10.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.673 | 0 | True |
| `model.layers.10.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.655 | 0 | True |
| `model.layers.10.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.616 | 0 | True |
| `model.layers.10.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.438 | 0 | True |
| `model.layers.10.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.438 | 0 | True |
| `model.layers.10.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.709 | 0 | True |
| `model.layers.10.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.783 | 0 | True |
| `model.layers.10.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.982 | 0 | True |
| `model.layers.10.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.706 | 0 | True |
| `model.layers.10.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.625 | 0 | True |
| `model.layers.10.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.682 | 0 | True |
| `model.layers.11.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.580 | 0 | True |
| `model.layers.11.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.671 | 0 | True |
| `model.layers.11.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.636 | 0 | True |
| `model.layers.11.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.617 | 0 | True |
| `model.layers.11.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.473 | 0 | True |
| `model.layers.11.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.750 | 0 | True |
| `model.layers.11.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.896 | 0 | True |
| `model.layers.11.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.782 | 0 | True |
| `model.layers.11.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.312 | 0 | True |
| `model.layers.11.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.903 | 0 | True |
| `model.layers.11.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.438 | 0 | True |
| `model.layers.11.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.707 | 0 | True |
| `model.layers.12.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.866 | 0 | True |
| `model.layers.12.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.665 | 0 | True |
| `model.layers.12.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.644 | 0 | True |
| `model.layers.12.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.613 | 0 | True |
| `model.layers.12.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.482 | 0 | True |
| `model.layers.12.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.812 | 0 | True |
| `model.layers.12.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.703 | 0 | True |
| `model.layers.12.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.753 | 0 | True |
| `model.layers.12.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.018 | 0 | True |
| `model.layers.12.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.709 | 0 | True |
| `model.layers.12.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 15.875 | 0 | True |
| `model.layers.12.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.638 | 0 | True |
| `model.layers.13.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.786 | 0 | True |
| `model.layers.13.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.666 | 0 | True |
| `model.layers.13.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.645 | 0 | True |
| `model.layers.13.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.613 | 0 | True |
| `model.layers.13.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.223 | 0 | True |
| `model.layers.13.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.375 | 0 | True |
| `model.layers.13.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.783 | 0 | True |
| `model.layers.13.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.759 | 0 | True |
| `model.layers.13.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.902 | 0 | True |
| `model.layers.13.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.822 | 0 | True |
| `model.layers.13.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 15.938 | 0 | True |
| `model.layers.13.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.695 | 0 | True |
| `model.layers.14.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.375 | 0 | True |
| `model.layers.14.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.664 | 0 | True |
| `model.layers.14.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.645 | 0 | True |
| `model.layers.14.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.611 | 0 | True |
| `model.layers.14.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.688 | 0 | True |
| `model.layers.14.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.375 | 0 | True |
| `model.layers.14.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.706 | 0 | True |
| `model.layers.14.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.747 | 0 | True |
| `model.layers.14.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.911 | 0 | True |
| `model.layers.14.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.736 | 0 | True |
| `model.layers.14.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 15.938 | 0 | True |
| `model.layers.14.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.735 | 0 | True |
| `model.layers.15.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.491 | 0 | True |
| `model.layers.15.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.685 | 0 | True |
| `model.layers.15.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.654 | 0 | True |
| `model.layers.15.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.627 | 0 | True |
| `model.layers.15.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.598 | 0 | True |
| `model.layers.15.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.438 | 0 | True |
| `model.layers.15.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.727 | 0 | True |
| `model.layers.15.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.774 | 0 | True |
| `model.layers.15.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.009 | 0 | True |
| `model.layers.15.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.721 | 0 | True |
| `model.layers.15.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.062 | 0 | True |
| `model.layers.15.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.644 | 0 | True |
| `model.layers.16.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.143 | 0 | True |
| `model.layers.16.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.627 | 0 | True |
| `model.layers.16.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.660 | 0 | True |
| `model.layers.16.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.624 | 0 | True |
| `model.layers.16.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.366 | 0 | True |
| `model.layers.16.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.062 | 0 | True |
| `model.layers.16.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.813 | 0 | True |
| `model.layers.16.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.739 | 0 | True |
| `model.layers.16.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.071 | 0 | True |
| `model.layers.16.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.939 | 0 | True |
| `model.layers.16.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 15.375 | 0 | True |
| `model.layers.16.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.888 | 0 | True |
| `model.layers.17.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.473 | 0 | True |
| `model.layers.17.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.607 | 0 | True |
| `model.layers.17.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.624 | 0 | True |
| `model.layers.17.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.592 | 0 | True |
| `model.layers.17.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.670 | 0 | True |
| `model.layers.17.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.312 | 0 | True |
| `model.layers.17.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.707 | 0 | True |
| `model.layers.17.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.661 | 0 | True |
| `model.layers.17.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.857 | 0 | True |
| `model.layers.17.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.731 | 0 | True |
| `model.layers.17.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.250 | 0 | True |
| `model.layers.17.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.677 | 0 | True |
| `model.layers.18.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.438 | 0 | True |
| `model.layers.18.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.609 | 0 | True |
| `model.layers.18.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.599 | 0 | True |
| `model.layers.18.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.579 | 0 | True |
| `model.layers.18.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.857 | 0 | True |
| `model.layers.18.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.438 | 0 | True |
| `model.layers.18.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.672 | 0 | True |
| `model.layers.18.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.666 | 0 | True |
| `model.layers.18.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.884 | 0 | True |
| `model.layers.18.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.714 | 0 | True |
| `model.layers.18.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.125 | 0 | True |
| `model.layers.18.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.682 | 0 | True |
| `model.layers.19.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.768 | 0 | True |
| `model.layers.19.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.640 | 0 | True |
| `model.layers.19.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.632 | 0 | True |
| `model.layers.19.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.602 | 0 | True |
| `model.layers.19.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.589 | 0 | True |
| `model.layers.19.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.125 | 0 | True |
| `model.layers.19.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.642 | 0 | True |
| `model.layers.19.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.675 | 0 | True |
| `model.layers.19.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.946 | 0 | True |
| `model.layers.19.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.739 | 0 | True |
| `model.layers.19.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 15.562 | 0 | True |
| `model.layers.19.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.629 | 0 | True |
| `model.layers.2.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.661 | 0 | True |
| `model.layers.2.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.600 | 0 | True |
| `model.layers.2.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.586 | 0 | True |
| `model.layers.2.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.581 | 0 | True |
| `model.layers.2.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.491 | 0 | True |
| `model.layers.2.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.750 | 0 | True |
| `model.layers.2.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.707 | 0 | True |
| `model.layers.2.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.685 | 0 | True |
| `model.layers.2.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.973 | 0 | True |
| `model.layers.2.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.759 | 0 | True |
| `model.layers.2.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.062 | 0 | True |
| `model.layers.2.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.695 | 0 | True |
| `model.layers.20.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.625 | 0 | True |
| `model.layers.20.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.642 | 0 | True |
| `model.layers.20.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.629 | 0 | True |
| `model.layers.20.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.605 | 0 | True |
| `model.layers.20.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.652 | 0 | True |
| `model.layers.20.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.312 | 0 | True |
| `model.layers.20.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.757 | 0 | True |
| `model.layers.20.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.697 | 0 | True |
| `model.layers.20.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.848 | 0 | True |
| `model.layers.20.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.831 | 0 | True |
| `model.layers.20.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.062 | 0 | True |
| `model.layers.20.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.776 | 0 | True |
| `model.layers.21.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.196 | 0 | True |
| `model.layers.21.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.613 | 0 | True |
| `model.layers.21.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.593 | 0 | True |
| `model.layers.21.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.584 | 0 | True |
| `model.layers.21.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.768 | 0 | True |
| `model.layers.21.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 15.875 | 0 | True |
| `model.layers.21.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.763 | 0 | True |
| `model.layers.21.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.689 | 0 | True |
| `model.layers.21.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.946 | 0 | True |
| `model.layers.21.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.906 | 0 | True |
| `model.layers.21.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 15.750 | 0 | True |
| `model.layers.21.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.737 | 0 | True |
| `model.layers.22.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.116 | 0 | True |
| `model.layers.22.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.585 | 0 | True |
| `model.layers.22.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.588 | 0 | True |
| `model.layers.22.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.581 | 0 | True |
| `model.layers.22.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.625 | 0 | True |
| `model.layers.22.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.312 | 0 | True |
| `model.layers.22.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.754 | 0 | True |
| `model.layers.22.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.603 | 0 | True |
| `model.layers.22.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.857 | 0 | True |
| `model.layers.22.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.841 | 0 | True |
| `model.layers.22.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.375 | 0 | True |
| `model.layers.22.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.739 | 0 | True |
| `model.layers.23.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.304 | 0 | True |
| `model.layers.23.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.611 | 0 | True |
| `model.layers.23.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.595 | 0 | True |
| `model.layers.23.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.584 | 0 | True |
| `model.layers.23.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 8.902 | 0 | True |
| `model.layers.23.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.500 | 0 | True |
| `model.layers.23.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.797 | 0 | True |
| `model.layers.23.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.719 | 0 | True |
| `model.layers.23.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.902 | 0 | True |
| `model.layers.23.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.864 | 0 | True |
| `model.layers.23.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 15.812 | 0 | True |
| `model.layers.23.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.677 | 0 | True |
| `model.layers.3.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.080 | 0 | True |
| `model.layers.3.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.607 | 0 | True |
| `model.layers.3.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.587 | 0 | True |
| `model.layers.3.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.585 | 0 | True |
| `model.layers.3.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.500 | 0 | True |
| `model.layers.3.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.812 | 0 | True |
| `model.layers.3.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.745 | 0 | True |
| `model.layers.3.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.710 | 0 | True |
| `model.layers.3.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.080 | 0 | True |
| `model.layers.3.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.851 | 0 | True |
| `model.layers.3.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 17.125 | 0 | True |
| `model.layers.3.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.677 | 0 | True |
| `model.layers.4.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.455 | 0 | True |
| `model.layers.4.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.623 | 0 | True |
| `model.layers.4.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.618 | 0 | True |
| `model.layers.4.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.594 | 0 | True |
| `model.layers.4.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.509 | 0 | True |
| `model.layers.4.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 17.000 | 0 | True |
| `model.layers.4.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.698 | 0 | True |
| `model.layers.4.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.715 | 0 | True |
| `model.layers.4.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.920 | 0 | True |
| `model.layers.4.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.708 | 0 | True |
| `model.layers.4.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 17.062 | 0 | True |
| `model.layers.4.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.643 | 0 | True |
| `model.layers.5.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.161 | 0 | True |
| `model.layers.5.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.628 | 0 | True |
| `model.layers.5.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.620 | 0 | True |
| `model.layers.5.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.588 | 0 | True |
| `model.layers.5.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.750 | 0 | True |
| `model.layers.5.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.250 | 0 | True |
| `model.layers.5.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.719 | 0 | True |
| `model.layers.5.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.693 | 0 | True |
| `model.layers.5.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.795 | 0 | True |
| `model.layers.5.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.715 | 0 | True |
| `model.layers.5.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.875 | 0 | True |
| `model.layers.5.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.654 | 0 | True |
| `model.layers.6.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.188 | 0 | True |
| `model.layers.6.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.621 | 0 | True |
| `model.layers.6.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.602 | 0 | True |
| `model.layers.6.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.592 | 0 | True |
| `model.layers.6.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.464 | 0 | True |
| `model.layers.6.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.000 | 0 | True |
| `model.layers.6.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.721 | 0 | True |
| `model.layers.6.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.717 | 0 | True |
| `model.layers.6.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.080 | 0 | True |
| `model.layers.6.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.752 | 0 | True |
| `model.layers.6.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.188 | 0 | True |
| `model.layers.6.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.638 | 0 | True |
| `model.layers.7.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.562 | 0 | True |
| `model.layers.7.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.634 | 0 | True |
| `model.layers.7.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.612 | 0 | True |
| `model.layers.7.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.599 | 0 | True |
| `model.layers.7.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.527 | 0 | True |
| `model.layers.7.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.250 | 0 | True |
| `model.layers.7.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.713 | 0 | True |
| `model.layers.7.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.712 | 0 | True |
| `model.layers.7.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 11.946 | 0 | True |
| `model.layers.7.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.712 | 0 | True |
| `model.layers.7.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.312 | 0 | True |
| `model.layers.7.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.653 | 0 | True |
| `model.layers.8.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.357 | 0 | True |
| `model.layers.8.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.662 | 0 | True |
| `model.layers.8.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.649 | 0 | True |
| `model.layers.8.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.621 | 0 | True |
| `model.layers.8.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 9.786 | 0 | True |
| `model.layers.8.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 17.188 | 0 | True |
| `model.layers.8.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.702 | 0 | True |
| `model.layers.8.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.756 | 0 | True |
| `model.layers.8.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.045 | 0 | True |
| `model.layers.8.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.721 | 0 | True |
| `model.layers.8.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.312 | 0 | True |
| `model.layers.8.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.670 | 0 | True |
| `model.layers.9.input_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.696 | 0 | True |
| `model.layers.9.mlp.down_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.662 | 0 | True |
| `model.layers.9.mlp.gate_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.621 | 0 | True |
| `model.layers.9.mlp.up_proj.weight` | 4358144 | ALL_RAW | 7.000 | 10.604 | 0 | True |
| `model.layers.9.post_attention_layernorm.weight` | 896 | ALL_RAW | 7.170 | 10.330 | 0 | True |
| `model.layers.9.self_attn.k_proj.bias` | 128 | ALL_RAW | 8.188 | 16.250 | 0 | True |
| `model.layers.9.self_attn.k_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.793 | 0 | True |
| `model.layers.9.self_attn.o_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.751 | 0 | True |
| `model.layers.9.self_attn.q_proj.bias` | 896 | ALL_RAW | 7.170 | 12.196 | 0 | True |
| `model.layers.9.self_attn.q_proj.weight` | 802816 | ALL_RAW | 7.000 | 10.857 | 0 | True |
| `model.layers.9.self_attn.v_proj.bias` | 128 | ALL_RAW | 8.188 | 16.500 | 0 | True |
| `model.layers.9.self_attn.v_proj.weight` | 114688 | ALL_RAW | 7.001 | 10.731 | 0 | True |
| `model.norm.weight` | 896 | ALL_RAW | 7.170 | 8.991 | 0 | True |

SHA-256 is over restored uint16 words vs the Safetensors source. No FP32 field extract.
