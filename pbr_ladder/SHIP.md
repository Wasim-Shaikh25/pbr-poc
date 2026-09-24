# SHIP — how to ship the PBR sub-4-bit model + pipeline

Date: 2026-09-24. What we ship, the exact commands, and the demo that makes a buyer react.
Engine decision: **IQ3_M + imatrix** (`ship-sub4`) — 99.1% quality @ 3.86 bpw, proven near-optimal
([`SUB3BIT_FINDINGS.md`](SUB3BIT_FINDINGS.md)). Training-free, CPU-only, standard GGUF.

## What "the product" actually is (3 pieces)
1. **The pipeline** — [`pbr_pipeline.py`](pbr_pipeline.py): one CPU command turns any base model into a
   sub-4-bit GGUF + an auditable provenance manifest.
2. **The model(s)** — reference IQ3 GGUFs (e.g. Qwen2.5-3B @ 1.49 GB) that run on the existing
   llama.cpp/Ollama ecosystem (CPU / Android / iOS / Metal) unchanged.
3. **The proof** — [`results_3b_taskacc.md`](results_3b_taskacc.md) (99.1% HellaSwag) + the manifest
   (base hash, exact quantize command, output hash, size, eff bpw) = reproducible, verifiable.

## Step 1 — produce a shippable model (one command)
```bash
export LLAMA_BIN="/c/python_project/bonsai test/bin/llama.cpp"
python pbr_ladder/pbr_pipeline.py quantize \
  --base <path-to-f16.gguf> --calib pbr_ladder/pipeline_data/calib.txt \
  --recipe ship-sub4 --out D:/pbr_work/out/mymodel-iq3.gguf
python pbr_ladder/pbr_pipeline.py validate --model D:/pbr_work/out/mymodel-iq3.gguf \
  --eval pbr_ladder/pipeline_data/wiki_eval.txt
```
Output: the GGUF + `mymodel-iq3.manifest.json` (provenance). Already dogfooded on 0.5B and 3B.

## Step 2 — the demo that sells it (on-phone, recommended next)
The claim "runs on a phone at real speed with ~99% quality" becomes a screen recording:
1. Install a prebuilt llama.cpp Android app (e.g. **PocketPal** or **ChatterUI**) on the owner's phone.
2. Copy the IQ3 GGUF (`qwen3b-iq3-imat.gguf`, ~1.49 GB) to the device.
3. Load it, run a prompt, **record tok/s + RAM**. (Decode is memory-bandwidth bound; expect a real
   phone number, not cloud-speed — honesty in the caption.)
4. Screen-record: model size (1.49 GB vs Q4's 1.84 GB), the answer quality, the tok/s.
That recording + the 99.1% proof + the one-command pipeline = the shareable pitch.

## Step 3 — package the repo for handoff
- README pointing to: PIPELINE.md (how-to), results_3b_taskacc.md (proof), POSITIONING.md (the wedge),
  SUB3BIT_FINDINGS.md (why IQ3 is the honest ceiling — credibility).
- `.gitignore` already excludes `*.gguf` / `*.imatrix.dat` (models are reproducible from the runners).

## The honest product claim (what we can say without lying)
> "Bring your own model. One CPU-only command turns it into a **sub-4-bit build (~3.86 bpw, 23%
> smaller than Q4) that keeps 99% of quality on a real downstream task**, runs on a phone via
> standard llama.cpp, and ships with a manifest proving exactly how it was made and how to re-verify."

What we do NOT claim: sub-3-bit at 99% (impossible training-free), beating IQ3 (it's the ceiling),
or specific on-phone tok/s until measured on device.

## Next action
Set up Step 2 (on-phone demo) — it's the artifact a partner reacts to, and the one remaining piece
with real upside. Everything else (engine, pipeline, proof) is done.
