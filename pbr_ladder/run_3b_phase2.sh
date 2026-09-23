#!/usr/bin/env bash
# Phase 2 of the 3B test: fill the matched baselines that phase 1 could not,
# to isolate WHAT is winning. Reuses the saved imatrix (no rebuild). Re-downloads
# the f16 base (robust resume-retry) since phase 1 deleted it to save disk.
#
# Adds, all vs the same base + same imatrix + same eval slice:
#   stock-iq2m-imat  : IQ2_M + imatrix, NO q8 embed  <- matched baseline for ship-2bit-iq
#                      (delta to ship-2bit-iq isolates the 8-bit embedding lever)
#   iq3-imat         : IQ3_M + imatrix               <- 3-bit codebook frontier
#   iq3-imat-q8e     : IQ3_M + imatrix + q8 embed    <- 3-bit codebook + embed lever
set -u
LLAMA_BIN="${LLAMA_BIN:?}"; REPO="${REPO:-/c/python_project/pbr-poc}"; WORK="${WORK:-/d/pbr_work}"
BASE="$WORK/models/Qwen2.5-3B-Instruct-f16.gguf"; PART="$BASE.part"
OUT="$WORK/out"; IMAT="$OUT/qwen3b-ship3.imatrix.dat"
EVAL="$WORK/wiki_eval_3b.txt"; CTX=512; EXPECT=6178317216
URL="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-f16.gguf"
RES="$REPO/pbr_ladder/results_3b.md"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
[ -f "$IMAT" ] || { log "FATAL: saved imatrix missing at $IMAT"; exit 1; }
[ -f "$EVAL" ] || head -c 150000 "$REPO/pbr_ladder/pipeline_data/wiki_eval.txt" > "$EVAL"

log "re-downloading f16 base (robust resume-retry)"
for a in $(seq 1 80); do
  sz=$(stat -c%s "$PART" 2>/dev/null || echo 0); [ "$sz" -ge "$EXPECT" ] && break
  [ -f "$BASE" ] && { sz=$(stat -c%s "$BASE"); [ "$sz" -eq "$EXPECT" ] && break; }
  curl -L -C - --retry 100 --retry-delay 3 --retry-all-errors --connect-timeout 30 -o "$PART" "$URL"
  sleep 3
done
[ -f "$BASE" ] || { [ "$(stat -c%s "$PART" 2>/dev/null||echo 0)" -eq "$EXPECT" ] && mv "$PART" "$BASE"; }
[ -f "$BASE" ] && [ "$(stat -c%s "$BASE")" -eq "$EXPECT" ] || { log "FATAL: base incomplete"; exit 1; }
log "base ready"

ppl(){ "$LLAMA_BIN/llama-perplexity.exe" -m "$1" -f "$EVAL" -c $CTX 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | tail -1 | grep -oE "[0-9.]+$"; }
mb(){ local b=$(stat -c%s "$1" 2>/dev/null||echo 0); awk "BEGIN{printf \"%.1f\", $b/1e6}"; }
bpw(){ python - "$1" <<'PY'
import sys,os
from gguf import GGUFReader
f=sys.argv[1]; r=GGUFReader(f); p=sum(int(t.n_elements) for t in r.tensors)
print(f"{os.path.getsize(f)*8/p:.3f}")
PY
}

log "quantize matched baselines (reuse imatrix)"
"$LLAMA_BIN/llama-quantize.exe" --imatrix "$IMAT" "$BASE" "$OUT/qwen3b-stock-iq2m-imat.gguf" IQ2_M >/dev/null 2>&1
"$LLAMA_BIN/llama-quantize.exe" --imatrix "$IMAT" "$BASE" "$OUT/qwen3b-iq3-imat.gguf" IQ3_M >/dev/null 2>&1
"$LLAMA_BIN/llama-quantize.exe" --imatrix "$IMAT" --token-embedding-type q8_0 --output-tensor-type q8_0 "$BASE" "$OUT/qwen3b-iq3-imat-q8e.gguf" IQ3_M >/dev/null 2>&1
log "quantize done; removing f16 to free disk"; rm -f "$BASE" "$PART"

log "perplexity"
{
  echo; echo "## Phase 2 — matched baselines (2026-09-24, reuses phase-1 imatrix)"
  echo "| config | PPL | size | eff bpw |"
  echo "|---|---|---|---|"
  for tag in stock-iq2m-imat iq3-imat iq3-imat-q8e; do
    f="$OUT/qwen3b-$tag.gguf"; P=$(ppl "$f"); S=$(mb "$f"); B=$(bpw "$f")
    echo "$tag PPL=$P ${S}MB ${B}bpw" >&2   # progress to stderr, not into $RES
    echo "| $tag | $P | ${S} MB | ${B} |"
  done
  echo
  echo "Compare **stock-iq2m-imat** (IQ2+imatrix, no q8 embed) vs **ship-2bit-iq**"
  echo "(12.285, IQ2+imatrix+q8 embed): the gap = the 8-bit embedding lever alone."
} >> "$RES"
log "DONE. appended to $RES"
tail -20 "$RES"
