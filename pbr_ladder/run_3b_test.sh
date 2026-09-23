#!/usr/bin/env bash
# End-to-end 3B recipe test: does our sub-4-bit recipe beat a well-tuned stock
# GGUF at ~3B, where 0.5B was too embedding-dominated to show it?
#
# Base: Qwen2.5-3B-Instruct f16 (same family as our 0.5B work -> clean scaling
# comparison). Compares, on the SAME base + SAME eval slice:
#   stock Q3_K_M (no imatrix)     <- the honest baseline to beat at 3-bit
#   ship-3bit    (Q3_K_M+imatrix) <- our recipe at 3-bit
#   stock IQ2_M  (no imatrix)     <- baseline below-3-bit
#   ship-2bit-iq (IQ2_M+imatrix+q8embed) <- our recipe below-3-bit
#   stock Q4_K_M (reference: "what you'd normally download")
#
# Dogfoods pbr_pipeline.py for the recipe configs (proves it works at 3B,
# emits manifests). Writes a results table. Designed to run in the background.
#
# Env: LLAMA_BIN (llama.cpp bin dir), REPO (pbr-poc root), WORK (D:/pbr_work).
set -u
LLAMA_BIN="${LLAMA_BIN:?set LLAMA_BIN}"
REPO="${REPO:-/c/python_project/pbr-poc}"
WORK="${WORK:-/d/pbr_work}"
BASE="$WORK/models/Qwen2.5-3B-Instruct-f16.gguf"
OUT="$WORK/out"; mkdir -p "$OUT"
CALIB="$REPO/pbr_ladder/pipeline_data/calib.txt"
EVAL="$WORK/wiki_eval_3b.txt"          # trimmed slice (see below) for tractable CPU eval
CTX=512
PIPE="$REPO/pbr_ladder/pbr_pipeline.py"
RES="$REPO/pbr_ladder/results_3b.md"
EXPECT=6178317216                       # f16 byte size, to confirm a complete download

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 0. wait for the download to finish (complete = exact expected size, stable)
log "waiting for base download to complete ($BASE)"
while :; do
  [ -f "$BASE" ] || { sleep 30; continue; }
  sz=$(stat -c%s "$BASE" 2>/dev/null || echo 0)
  [ "$sz" = "$EXPECT" ] && { log "base present, $sz bytes"; break; }
  sleep 30
done

# make a trimmed eval slice so 3B CPU perplexity stays ~10-20 min/config
head -c 150000 "$REPO/pbr_ladder/pipeline_data/wiki_eval.txt" > "$EVAL"

ppl(){ # model.gguf -> PPL number
  "$LLAMA_BIN/llama-perplexity.exe" -m "$1" -f "$EVAL" -c $CTX 2>&1 \
    | grep -oE "Final estimate: PPL = [0-9.]+" | tail -1 | grep -oE "[0-9.]+$"; }
mb(){ python -c "import os;print(round(os.path.getsize(r'$1')/1e6,1))"; }

log "=== quantize: our recipes (via pipeline) ==="
python "$PIPE" quantize --base "$BASE" --calib "$CALIB" --recipe ship-3bit \
  --out "$OUT/qwen3b-ship3.gguf" 2>&1 | tail -2
IMAT="$OUT/qwen3b-ship3.imatrix.dat"
python "$PIPE" quantize --base "$BASE" --recipe ship-2bit-iq --reuse-imatrix "$IMAT" \
  --out "$OUT/qwen3b-ship2iq.gguf" 2>&1 | tail -2

log "=== quantize: stock baselines (direct, no imatrix) ==="
"$LLAMA_BIN/llama-quantize.exe" "$BASE" "$OUT/qwen3b-stock-q3km.gguf" Q3_K_M >/dev/null 2>&1
"$LLAMA_BIN/llama-quantize.exe" "$BASE" "$OUT/qwen3b-stock-iq2m.gguf" IQ2_M >/dev/null 2>&1
"$LLAMA_BIN/llama-quantize.exe" "$BASE" "$OUT/qwen3b-stock-q4km.gguf" Q4_K_M >/dev/null 2>&1

# free the big f16 before the slow evals (perplexity only needs the quantized files)
log "quantize done; removing f16 base to free disk"
rm -f "$BASE"

log "=== perplexity (trimmed slice, ctx=$CTX) ==="
declare -A P S
for tag in stock-q3km ship3 stock-iq2m ship2iq stock-q4km; do
  case $tag in
    ship3)      f="$OUT/qwen3b-ship3.gguf";;
    ship2iq)    f="$OUT/qwen3b-ship2iq.gguf";;
    *)          f="$OUT/qwen3b-$tag.gguf";;
  esac
  P[$tag]=$(ppl "$f"); S[$tag]=$(mb "$f")
  log "  $tag : PPL=${P[$tag]}  ${S[$tag]}MB"
done

log "=== writing $RES ==="
{
  echo "# 3B recipe test — Qwen2.5-3B-Instruct (f16 base), $(date +%Y-%m-%d)"
  echo
  echo "Same base + same trimmed eval slice (150 KB, ctx=$CTX). Question: does our"
  echo "recipe beat a well-tuned stock GGUF at ~3B where 0.5B could not?"
  echo
  echo "| config | PPL | size (MB) |"
  echo "|---|---|---|"
  echo "| stock Q3_K_M (no imatrix) | ${P[stock-q3km]} | ${S[stock-q3km]} |"
  echo "| **ship-3bit (Q3_K_M+imatrix)** | **${P[ship3]}** | ${S[ship3]} |"
  echo "| stock IQ2_M (no imatrix) | ${P[stock-iq2m]} | ${S[stock-iq2m]} |"
  echo "| **ship-2bit-iq (IQ2_M+imatrix+q8embed)** | **${P[ship2iq]}** | ${S[ship2iq]} |"
  echo "| stock Q4_K_M (reference) | ${P[stock-q4km]} | ${S[stock-q4km]} |"
  echo
  echo "Lower PPL = better. Compare ship-3bit vs stock-q3km at ~equal size, and"
  echo "ship-2bit-iq vs stock-iq2m. If ship-* does NOT win: diagnose (rotation"
  echo "dropped? calib mismatch? allocation? arch-specific?) and iterate."
} > "$RES"
log "DONE. results -> $RES"
cat "$RES"
