# PBR-H95E E1 — Embedding inventory (Qwen2.5-0.5B-Instruct)

- Embed param: `model.embed_tokens.weight`
- Shape: **151936 × 896** (bfloat16)
- Embed params: **136,134,656** (27.56% of unique storage)
- BF16 MiB (embed only): **259.66**
- tie_word_embeddings: **True**
- Tied aliases: `[]`
- Separate lm_head params: `[]`
- Configured vocab_size: **151936**
- Tokenizer len (reachable): **151665** (vocab_size attr=151643)
- Padded / unreachable rows: **271**
- Padded rows are zero: **False** (mean L2 norm=0.296459, max=0.296478, zero_rows=0)

## Vocab class histogram

| class | count |
| --- | ---: |
| ASCII_LETTER | 87327 |
| OTHER_MULTILINGUAL | 59067 |
| PUNCTUATION | 5062 |
| RESERVED_OR_UNUSED | 271 |
| WHITESPACE | 185 |
| SPECIAL | 14 |
| NUMBER | 10 |

## Softmax mass on padded rows (probe)
- mean=1.741898e-03, max=1.757204e-03, n_probes=8

## Notes

Padded/unreachable rows are NOT zero — mean L2 norm reported. Zeroing them is not a meaningful BPW win for H95E quality claims.
