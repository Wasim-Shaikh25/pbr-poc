#!/usr/bin/env python3
"""PBR-Ladder deployment pipeline: base model -> ready-to-run, sub-4-bit GGUF.

This is the *defensible* pipeline. It takes a base fp16 GGUF plus a calibration
slice, applies our recipe on top of llama.cpp's existing quantizer (importance
matrix + per-tensor bit allocation + embedding lever), and hands back a single
standard `.gguf` that runs on every llama.cpp backend TODAY (CPU / Android / iOS
/ Metal) with zero new kernels. Under mmap the file stays packed in RAM -- it
never expands back to fp16 (see artifacts/disk_ram_tunnel.md: 0.078x peak RSS,
bit-exact logits). That mmap behaviour IS the shippable "tunnel".

Every produced model is accompanied by a manifest.json recording the exact
recipe, flags, command, file size, effective bits-per-weight and (after
validate) perplexity -- so anyone can reproduce and audit the result. That
manifest is what makes the handoff defensible: "here is the model, here is
exactly how it was made, here is how to re-verify it."

Design constraints (owner):
  * No custom kernel, no industry infra. CPU-only, consumer/free hardware.
  * Ship only sub-4-bit models; the >=98% quality promise is a 3B-7B deliverable
    (0.5B is embedding-dominated -- see gguf_validation_results.md).

Requires llama.cpp binaries (llama-imatrix, llama-quantize, llama-perplexity,
llama-cli). Locate them via --llama-bin, or $LLAMA_BIN, or PATH. Get prebuilt
CPU binaries from https://github.com/ggml-org/llama.cpp/releases (no build step).

Usage:
  # 1. make a shippable model + manifest
  python pbr_pipeline.py quantize \
      --base qwen2.5-0.5b-instruct-fp16.gguf \
      --calib pipeline_data/calib.txt \
      --recipe ship-3bit --out dist/qwen05b-pbr-q3.gguf

  # 2. verify it runs and measure quality/size
  python pbr_pipeline.py validate \
      --model dist/qwen05b-pbr-q3.gguf \
      --eval pipeline_data/wiki_eval.txt

  # 3. prove the handed-over file actually generates text
  python pbr_pipeline.py run --model dist/qwen05b-pbr-q3.gguf \
      --prompt "Explain what quantization is in one sentence."

  # list the built-in recipes and what each is for
  python pbr_pipeline.py recipes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Recipes. Each recipe is a named preset that encodes a validated finding.
# `type` is the llama-quantize target type. `flags` are extra CLI flags.
# `imatrix` = build/use an importance matrix (always a free win, keep it on).
# `min_params` = the model size below which this recipe is NOT recommended
#   (documented, not enforced) -- our results show IQ/below-3-bit only pay off
#   at scale; on 0.5B, scalar Q3_K_M + imatrix is the honest winner.
# ---------------------------------------------------------------------------
RECIPES: dict[str, dict] = {
    "ship-3bit": {
        "type": "Q3_K_M",
        "imatrix": True,
        "flags": [],
        "target_bpw": "~3.5 (weights); embeddings dominate on small models",
        "min_params": 0,
        "note": (
            "Scalar k-quant + importance matrix. The honest winner at 0.5B "
            "(beats IQ3 at equal size). Safest default across all sizes."
        ),
    },
    "ship-3bit-embed": {
        "type": "Q3_K_M",
        "imatrix": True,
        "flags": ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
        "target_bpw": "~3.5 weights, 8-bit embed/output",
        "min_params": 0,
        "note": (
            "Adds the 8-bit embedding lever. Measured marginal on Qwen2.5-3B "
            "(<1% PPL for +10% size); generally NOT worth it. Prefer ship-sub4."
        ),
    },
    "ship-sub4": {
        "type": "IQ3_M",
        "imatrix": True,
        "flags": [],
        "target_bpw": "~3.86 (codebook)",
        "min_params": 1_500_000_000,
        "note": (
            "THE SWEET SPOT (validated on Qwen2.5-3B, 2026-09-24): IQ3_M + imatrix "
            "= near-Q4 quality (+5.4% PPL vs Q4_K_M) at ~3.86 bpw, 23% smaller and "
            "genuinely sub-4-bit. IQ codebooks WIN at >=~3B (they lose at 0.5B). "
            "No q8-embed: the embedding lever adds <1% PPL for +10% size. "
            "Recommended default for shipping a sub-4-bit model at >=1.5B."
        ),
    },
    "ship-3bit-iq": {  # alias kept for back-compat; identical to ship-sub4
        "type": "IQ3_M",
        "imatrix": True,
        "flags": [],
        "target_bpw": "~3.86 (codebook)",
        "min_params": 1_500_000_000,
        "note": "Alias of ship-sub4 (IQ3_M + imatrix). See ship-sub4.",
    },
    "ship-2bit-iq": {
        "type": "IQ2_M",
        "imatrix": True,
        "flags": [],
        "target_bpw": "~2.96 (codebook)",
        "min_params": 3_000_000_000,
        "note": (
            "EXTREME SIZE ONLY. IQ2_M+imatrix on Qwen2.5-3B = 2.96 bpw / smallest, "
            "but +33% PPL vs Q4 -> below a >=98% quality bar. Use only when size "
            "trumps quality. Prefer ship-sub4. (q8-embed dropped: marginal gain.)"
        ),
    },
}


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------
def _exe(name: str) -> str:
    """Return platform executable name (Windows appends .exe)."""
    if os.name == "nt" and not name.endswith(".exe"):
        return name + ".exe"
    return name


def find_bin(tool: str, llama_bin: str | None) -> str:
    """Locate a llama.cpp tool via --llama-bin dir, $LLAMA_BIN, or PATH."""
    candidates = []
    if llama_bin:
        candidates.append(Path(llama_bin) / _exe(tool))
    env = os.environ.get("LLAMA_BIN")
    if env:
        candidates.append(Path(env) / _exe(tool))
    for c in candidates:
        if c.is_file():
            return str(c)
    onpath = shutil.which(tool) or shutil.which(_exe(tool))
    if onpath:
        return onpath
    raise FileNotFoundError(
        f"Could not find '{tool}'. Pass --llama-bin <dir>, set $LLAMA_BIN, or put "
        f"llama.cpp binaries on PATH. Prebuilt CPU binaries: "
        f"https://github.com/ggml-org/llama.cpp/releases"
    )


def _run(cmd: list[str], quiet: bool = False) -> str:
    """Run a subprocess, streaming to a captured buffer, raise on failure."""
    if not quiet:
        print("  $", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        sys.stderr.write(out[-4000:])
        raise SystemExit(f"command failed (exit {proc.returncode}): {cmd[0]}")
    return out


def _sha256(path: Path, limit_mb: int = 0) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_mb and read >= limit_mb << 20:
                break
    return h.hexdigest()


def _param_count(path: Path) -> int | None:
    """Stored tensor-element count read straight from the GGUF file.

    This is the number of weights the file actually encodes, so
    size_bytes*8/param_count is a self-consistent effective bits-per-weight.
    Note: it can exceed the model's nominal HF param count when lm_head is
    stored untied from the embeddings (e.g. Qwen 0.5B: ~630M stored vs ~494M
    nominal). Requires the `gguf` package (ships with llama.cpp; `pip install
    gguf`); returns None if unavailable so the pipeline still works.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    try:
        reader = GGUFReader(str(path))
        return sum(int(t.n_elements) for t in reader.tensors)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_recipes(_args) -> int:
    print("Built-in recipes (see gguf_validation_results.md for the evidence):\n")
    for name, r in RECIPES.items():
        rec = ""
        if r["min_params"]:
            rec = f"  [recommended only >= {r['min_params']/1e9:.1f}B params]"
        print(f"* {name}  -> {r['type']}  (imatrix={'on' if r['imatrix'] else 'off'}){rec}")
        print(f"    target: {r['target_bpw']}")
        print(f"    {r['note']}\n")
    return 0


def cmd_quantize(args) -> int:
    base = Path(args.base).resolve()
    if not base.is_file():
        raise SystemExit(f"base model not found: {base}")
    if args.recipe not in RECIPES:
        raise SystemExit(f"unknown recipe '{args.recipe}'. Options: {', '.join(RECIPES)}")
    recipe = RECIPES[args.recipe]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    imatrix_bin = find_bin("llama-imatrix", args.llama_bin)
    quantize_bin = find_bin("llama-quantize", args.llama_bin)

    t0 = time.time()
    imatrix_path = None
    if recipe["imatrix"]:
        if args.reuse_imatrix:
            imatrix_path = Path(args.reuse_imatrix).resolve()
            if not imatrix_path.is_file():
                raise SystemExit(f"--reuse-imatrix file not found: {imatrix_path}")
            print(f"[1/2] reusing importance matrix: {imatrix_path.name} (skipped rebuild)")
        else:
            if not args.calib:
                raise SystemExit(f"recipe '{args.recipe}' needs --calib <text file> for the imatrix "
                                 f"(or --reuse-imatrix <file> to reuse one)")
            calib = Path(args.calib).resolve()
            if not calib.is_file():
                raise SystemExit(f"calib file not found: {calib}")
            imatrix_path = out.with_suffix(".imatrix.dat")
            print(f"[1/2] importance matrix  ({args.chunks} chunks of calib)")
            _run([imatrix_bin, "-m", str(base), "-f", str(calib),
                  "-o", str(imatrix_path), "--chunks", str(args.chunks)])

    print(f"[2/2] quantize -> {recipe['type']}")
    q_cmd = [quantize_bin]
    if imatrix_path:
        q_cmd += ["--imatrix", str(imatrix_path)]
    q_cmd += list(recipe["flags"])
    q_cmd += [str(base), str(out), recipe["type"]]
    _run(q_cmd)

    size = out.stat().st_size
    params = _param_count(out)
    manifest = {
        "produced_by": "pbr_pipeline.py quantize",
        "produced_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": base.name,
        "base_sha256_head64mb": _sha256(base, limit_mb=64),
        "recipe": args.recipe,
        "recipe_detail": recipe,
        "quantize_command": " ".join(q_cmd),
        "output": out.name,
        "output_sha256": _sha256(out),
        "size_bytes": size,
        "size_mb": round(size / 1e6, 1),
        "param_count_stored": params,
        "param_count_note": "GGUF stored tensor elements (may exceed nominal HF count if lm_head untied)",
        "effective_bpw": round(size * 8 / params, 3) if params else None,
        "imatrix": imatrix_path.name if imatrix_path else None,
        "elapsed_s": round(time.time() - t0, 1),
        "note": recipe["note"],
        "runtime": "standard llama.cpp; mmap keeps it packed in RAM (never fp16)",
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n=== quantize done ===")
    print(f"model    : {out}   ({manifest['size_mb']} MB"
          + (f", {manifest['effective_bpw']} eff bpw" if manifest["effective_bpw"] else "") + ")")
    print(f"manifest : {manifest_path}")
    print("next     : python pbr_pipeline.py validate --model "
          f'"{out}" --eval <eval.txt>')
    return 0


def cmd_validate(args) -> int:
    model = Path(args.model).resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    eval_txt = Path(args.eval).resolve()
    if not eval_txt.is_file():
        raise SystemExit(f"eval text not found: {eval_txt}")
    ppl_bin = find_bin("llama-perplexity", args.llama_bin)

    print(f"perplexity  (ctx={args.ctx})")
    out = _run([ppl_bin, "-m", str(model), "-f", str(eval_txt), "-c", str(args.ctx)])
    ppl = None
    for line in out.splitlines():
        if "Final estimate: PPL" in line:
            try:
                ppl = float(line.split("PPL =")[1].split()[0])
            except (IndexError, ValueError):
                pass
    size = model.stat().st_size

    result = {
        "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model.name,
        "eval_file": eval_txt.name,
        "ctx": args.ctx,
        "ppl": ppl,
        "size_mb": round(size / 1e6, 1),
    }
    # fold PPL into the sibling manifest if present
    manifest_path = model.with_suffix(".manifest.json")
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text())
        m["validation"] = result
        manifest_path.write_text(json.dumps(m, indent=2))
        print(f"  (updated {manifest_path.name})")

    print("\n=== validate done ===")
    print(f"PPL   : {ppl}   (eval slice; compare only within the same slice/ctx)")
    print(f"size  : {result['size_mb']} MB")
    if args.baseline:
        b = Path(args.baseline).resolve()
        if b.is_file():
            bout = _run([ppl_bin, "-m", str(b), "-f", str(eval_txt), "-c", str(args.ctx)],
                        quiet=True)
            bppl = None
            for line in bout.splitlines():
                if "Final estimate: PPL" in line:
                    bppl = float(line.split("PPL =")[1].split()[0])
            print(f"baseline PPL: {bppl}  ({b.name}, {b.stat().st_size/1e6:.1f} MB)")
            if bppl and ppl:
                print(f"delta : {ppl - bppl:+.3f} PPL vs baseline")
    return 0


def cmd_run(args) -> int:
    model = Path(args.model).resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    cli_bin = find_bin("llama-cli", args.llama_bin)
    print("generating (proves the handed-over file runs)...\n")
    # Cap context: long-context models (e.g. phi-3.5 = 128K) otherwise try to
    # allocate a huge KV cache and OOM. -c bounds it to what we actually need.
    out = _run([cli_bin, "-m", str(model), "-p", args.prompt,
                "-n", str(args.n_predict), "-c", str(args.ctx), "-st"], quiet=True)
    # print only the generated tail, not the load spam
    print(out[-1500:])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llama-bin", help="directory containing llama.cpp binaries "
                                       "(else $LLAMA_BIN, else PATH)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("recipes", help="list built-in recipes")
    sp.set_defaults(func=cmd_recipes)

    sq = sub.add_parser("quantize", help="base gguf -> shippable sub-4-bit gguf + manifest")
    sq.add_argument("--base", required=True, help="base fp16 .gguf")
    sq.add_argument("--calib", help="calibration text (for the importance matrix)")
    sq.add_argument("--recipe", default="ship-3bit", help="recipe name (see `recipes`)")
    sq.add_argument("--out", required=True, help="output .gguf path")
    sq.add_argument("--chunks", type=int, default=32, help="imatrix chunks (default 32)")
    sq.add_argument("--reuse-imatrix", help="reuse an existing .imatrix.dat instead of "
                                            "rebuilding (skips the slow step for 3B iteration)")
    sq.set_defaults(func=cmd_quantize)

    sv = sub.add_parser("validate", help="measure PPL + size of a produced model")
    sv.add_argument("--model", required=True)
    sv.add_argument("--eval", required=True, help="held-out eval text")
    sv.add_argument("--ctx", type=int, default=512)
    sv.add_argument("--baseline", help="optional stock .gguf to compare against")
    sv.set_defaults(func=cmd_validate)

    sr = sub.add_parser("run", help="generate a bit of text to prove the model runs")
    sr.add_argument("--model", required=True)
    sr.add_argument("--prompt", default="Hello, my name is")
    sr.add_argument("--n-predict", type=int, default=64)
    sr.add_argument("--ctx", type=int, default=2048, help="context cap (bounds KV cache; "
                                                          "raise only if you need long prompts)")
    sr.set_defaults(func=cmd_run)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
