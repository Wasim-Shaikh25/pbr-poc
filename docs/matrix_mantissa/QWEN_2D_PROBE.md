# Qwen 2-D Matrix Mantissa probe

## Command

```bash
python3 scripts/run_matrix_qwen_2d_probe.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --max-tensors 6 --max-tiles-per-tensor 64
```

## Result

| metric | value |
| --- | ---: |
| Tiled values | 98,304 |
| Candidate payload BPW | **7.1562** |
| RAW candidate BPW | **7.1562** |
| Mode counts | **raw 384/384** |
| Exact round-trip | yes |

Prior full-model adaptive mantissa: ~7.0001 mant / ~10.62 total (ALL_RAW).

## Reading

On these Qwen linear tiles, **V3 matrix predictors never beat RAW**. Same class as Phase A / adaptive mantissa: dense BF16 mantissas stay near 7 BPW.

Not a whole-checkpoint encode. Not a ≤4 BPW claim.
