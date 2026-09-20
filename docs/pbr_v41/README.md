# PBR V4.1 — bit-packed residual-pair codec

Synthetic exact PoC and statistics probe. **Not a Qwen result.**

## Quick start

```bash
python -m pip install -r pbr_v41/requirements.txt
python pbr_v41/pbr_v41_bitpacked_pair_poc.py --self-test
python pbr_v41/pbr_v41_bitpacked_pair_poc.py --dataset all --height 64 --width 64 --output artifacts/pbr_v41/pbr_v41_results.csv
```

Guide: `docs/pbr_v41/PBR_V41_BitPacked_Pair_Codec_Guide.docx`

## Status

Synthetic self-test + 1800-row benchmark executed. See `artifacts/pbr_v41/`.
Qwen remains **untested** for V4.1 — next step is a stats-only held-out probe only if we continue this branch.
