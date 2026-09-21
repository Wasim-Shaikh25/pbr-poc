# PBR-Ladder — script guide run (environment check only)

Source: user-supplied `PBR_Ladder_Script_Guide.docx`, plus 4 of the 9 scripts
it describes (`push_below4.py`, `push_below3.py`, `capture_activations.py`,
`eval_ppl.py`). The other 5 scripts named in the guide
(`formula_probe.py`, `node_combo_q4.py`, `node_structures.py`,
`push_nested.py`, `ladder_exact.py`) were **not supplied** and are not in
this repo yet.

## What ran

Only the synthetic-surrogate self-check (guide §2.3), which needs nothing
but `numpy`:

```bash
cd pbr_ladder
pip install numpy
python3 push_below4.py
python3 push_below3.py
```

**Result: PASS.** Output matches the guide's Appendix A exactly (seed 0,
512×896 synthetic Student-t tensor, synthetic activations with 8 outlier
channels):

| Method | bpw | output dB (measured) | output dB (Appendix A) |
| --- | ---: | ---: | ---: |
| INT3 RTN | 3.250 | 11.11 | 11.11 |
| INT3 GPTQ | 3.250 | 13.52 | 13.52 |
| INT3 rot+GPTQ | 3.250 | 18.91 | 18.91 |
| INT2 rot+GPTQ | 2.250 | 10.53 | 10.53 |
| 256+256+64 (push_below3) | 2.929 | 18.61 | 18.61 |

Full logs: `artifacts/pbr_ladder/push_below4_synthetic.txt`,
`artifacts/pbr_ladder/push_below3_synthetic.txt`.

This confirms the scripts run correctly and are seeded/deterministic. **It
is not a Qwen result** — same caveat the guide itself states in its status
line: "every result in this guide comes from a synthetic Qwen-like tensor."

## What did not run: the real-Qwen procedure (guide §4–§5)

`capture_activations.py` and `eval_ppl.py` both require downloading
`Qwen/Qwen2.5-0.5B-Instruct` and the `wikitext-2-raw-v1` dataset from
`huggingface.co`. In this sandbox, outbound HTTPS to `huggingface.co` is
**blocked by the egress proxy's organization policy** (`403` on `CONNECT`,
confirmed via the proxy status endpoint — `pypi.org`/`npmjs.org`/etc. are
allowed, `huggingface.co` is not). No weights, no activations, no real
perplexity number was obtained or fabricated.

This is the same limitation the guide already flags: both helper scripts
"were written without Hugging Face access and have not been run yet."
That remains true here.

Consequently none of the following from the guide could be executed:

- §3.1–3.7 tensor tests on real Qwen weights (`formula_probe.py`,
  `node_combo_q4.py`, `node_structures.py`, `push_below4.py`,
  `push_nested.py`, `push_below3.py`, `ladder_exact.py` against
  `qwen05b/model.safetensors`) — and 5 of those 7 scripts weren't
  supplied to begin with.
- §4 `capture_activations.py` — real calibration activations.
- §5.1–§5.4 the whole real-model procedure, perplexity gates, and
  retention targets.

## Go/no-go gates (guide §7.2) — status

| Gate | Status |
| --- | --- |
| Environment check (synthetic, §2.3) | **PASS** — matches Appendix A |
| Tensor quality (real weights) | **NOT RUN** — no network access to `huggingface.co` |
| Model quality / perplexity | **NOT RUN** — same reason |
| Retention | **NOT RUN** — same reason |

## To actually get real-Qwen numbers

Run this on a machine/environment with `huggingface.co` egress (this
sandbox does not have it):

```bash
pip install numpy safetensors torch transformers datasets huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir qwen05b
python3 capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 mlp.gate_proj acts_L12_gate.npy
M=qwen05b/model.safetensors; T=model.layers.12.mlp.gate_proj.weight; A=acts_L12_gate.npy
python3 push_below4.py $M $T $A
python3 push_below3.py $M $T $A
python3 eval_ppl.py Qwen/Qwen2.5-0.5B-Instruct
```

## Honesty

- No claim is made here about real Qwen compression, quality, or
  perplexity. Only the synthetic environment check ran.
- 5 of the 9 scripts the guide describes were not supplied and are not in
  this repo.
