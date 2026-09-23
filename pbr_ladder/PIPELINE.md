# PBR-Ladder Pipeline — base model → shippable sub-4-bit GGUF

Governed by [`../AGENTS.md`](../AGENTS.md) RULE 0. The pipeline itself:
[`pbr_pipeline.py`](pbr_pipeline.py). Positioning: [`POSITIONING.md`](POSITIONING.md).
Evidence: [`gguf_validation_results.md`](gguf_validation_results.md).

## What it is
One CPU-only, no-infra command chain that hands you a **standard GGUF** (runs on every
llama.cpp backend today) plus an **auditable manifest** recording exactly how it was made.
No custom kernel, no GPU, no cloud.

**The RAM property (the "tunnel"):** the output stays packed at ~3-bit in RAM under mmap —
it never re-expands to fp16. llama.cpp dequantizes one block at a time *inside* the matmul
and discards it; a full fp16 tensor is never materialized. Proven:
[`../artifacts/disk_ram_tunnel.md`](../artifacts/disk_ram_tunnel.md) — 0.078× peak RSS vs
full-load, bit-exact logits.

## Prerequisites
- Python 3 with the `gguf` package (`pip install gguf`) for the eff-bpw readout (optional).
- llama.cpp binaries (`llama-imatrix`, `llama-quantize`, `llama-perplexity`, `llama-cli`).
  Prebuilt CPU binaries: https://github.com/ggml-org/llama.cpp/releases — no build step.
  Point the pipeline at them with `--llama-bin <dir>`, `$LLAMA_BIN`, or PATH.
- A base **fp16 GGUF** of the model (convert HF → GGUF with llama.cpp's `convert_hf_to_gguf.py`).

## The three commands
```bash
# 1. quantize: base fp16 gguf + calib text -> sub-4-bit gguf + manifest.json
python pbr_pipeline.py quantize \
    --base qwen2.5-0.5b-instruct-fp16.gguf \
    --calib pipeline_data/calib.txt \
    --recipe ship-3bit --out dist/qwen05b-pbr-q3.gguf

# 2. validate: perplexity + size (folds result into the manifest)
python pbr_pipeline.py validate \
    --model dist/qwen05b-pbr-q3.gguf --eval pipeline_data/wiki_eval.txt

# 3. run: generate text to prove the handed-over file works
python pbr_pipeline.py run --model dist/qwen05b-pbr-q3.gguf \
    --prompt "Explain quantization in one sentence."
```
`python pbr_pipeline.py recipes` lists the presets and which model sizes each suits.

For iterating on a big model, build the importance matrix once and reuse it (skips the slow
step): `--reuse-imatrix dist/qwen05b-pbr-q3.imatrix.dat`.

## Recipes (encode our validated findings)
| recipe | type | for | note |
|---|---|---|---|
| `ship-3bit` | Q3_K_M + imatrix | **default, all sizes** | honest winner at 0.5B; beats IQ3 at equal size |
| `ship-3bit-embed` | Q3_K_M + imatrix + q8 embed | when base over-spends embeddings | measure first; often not worth the size |
| `ship-3bit-iq` | IQ3_M + imatrix | ≥1.5B | codebook only pays at scale |
| `ship-2bit-iq` | IQ2_M + imatrix + q8 embed | ≥3B | the sub-4-bit-@-≥98% regime; needs task-accuracy proof |

## Reproduced result (0.5B, laptop CPU, 2026-09-23)
Ran on Qwen2.5-0.5B-Instruct-fp16 with `ship-3bit`:
- quantize + imatrix (32 chunks): **145 s** (while sharing the CPU with other jobs).
- output: **432.0 MB**, **5.485 eff bpw** (stored tensor elements).
- generation: coherent, **75.1 tok/s** on CPU.
- PPL (prior validated run, same recipe/artifact): **16.509** on the eval slice
  (see [`gguf_validation_results.md`](gguf_validation_results.md)); beats stock Q3_K_M (16.549)
  and beats IQ3_M+imatrix (16.933) at equal size.

The manifest for each model records the base sha256, the exact quantize command, output
sha256, size, eff bpw, and (after validate) PPL — so any result can be re-verified.

## What the pipeline does NOT claim
- No proven quant-quality edge over a *well-tuned* stock GGUF at 0.5B beyond `imatrix`
  (which is llama.cpp's). The sub-4-bit-@-≥98% story is a **3B–7B** deliverable, still untested.
- Rotation (QuaRot/Hadamard) is not yet expressed through the GGUF path; it is the one
  remaining offline lever and matters most at 2-bit / at scale.

## Next
- Run `ship-3bit` (and `ship-2bit-iq`) on a **3B** model — same laptop, ~1 h, free. Decisive test.
- Add task-accuracy (`lm-eval-harness`) to state the real "≥98%".
- On-phone tok/s via a prebuilt llama.cpp Android app.
