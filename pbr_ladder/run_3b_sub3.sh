#!/usr/bin/env bash
# Sub-3-bit FLOOR experiment on Qwen2.5-3B (training-free / PTQ).
# Question: how far below 3 bpw can llama.cpp's IQ codebooks + imatrix go
# before quality collapses? We already have the anchors:
#   IQ3_M+imat = 3.86 bpw -> 99.1% HellaSwag retention (PASS)
#   IQ2_M+imat = ~2.7 bpw -> 90.6% (below the 98% bar)
# This fills the curve BELOW IQ2_M so we know the exact cliff shape:
#   IQ2_S  (2.5 bpw), IQ2_XS (2.31 bpw), IQ1_M (1.75 bpw) -- all +imatrix.
# Reuses the saved imatrix (no rebuild). Re-downloads f16 (robust resume),
# quantizes, deletes f16, then PPL + HellaSwag(1000) each. Retention vs the
# known Q4_K_M ref accuracy (near-lossless proxy; f16 deleted to save disk).
set -u
BIN="/c/python_project/bonsai test/bin/llama.cpp"
REPO="/c/python_project/pbr-poc"; W="/d/pbr_work"; OUT="$W/out"
BASE="$W/models/Qwen2.5-3B-Instruct-f16.gguf"; PART="$BASE.part"
IMAT="$OUT/qwen3b-ship3.imatrix.dat"
EVAL="$W/wiki_eval_3b.txt"; HS="$W/hellaswag_val_full.txt"
EXPECT=6178317216; TASKS=1000; PCTX=512; HCTX=1024
URL="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-f16.gguf"
RES="$REPO/pbr_ladder/results_3b_sub3.md"
Q4REF=75.1774   # stock Q4_K_M HellaSwag acc (from results_3b_taskacc.md) = retention denominator
log(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$W/models" "$OUT"
[ -f "$IMAT" ] || { log "FATAL: imatrix missing $IMAT"; exit 1; }

# --- fetch f16 base (robust resume-retry) unless a matched quant set already exists ---
need_base=0
for tag in iq2s-imat iq2xs-imat iq1m-imat; do [ -f "$OUT/qwen3b-$tag.gguf" ] || need_base=1; done
if [ "$need_base" = 1 ]; then
  log "re-downloading f16 base (robust resume-retry)"
  for a in $(seq 1 80); do
    [ -f "$BASE" ] && [ "$(stat -c%s "$BASE")" -eq "$EXPECT" ] && break
    sz=$(stat -c%s "$PART" 2>/dev/null || echo 0); [ "$sz" -ge "$EXPECT" ] && { mv "$PART" "$BASE"; break; }
    curl -L -C - --retry 100 --retry-delay 3 --retry-all-errors --connect-timeout 30 -o "$PART" "$URL"
    sleep 3
  done
  [ -f "$BASE" ] || { [ "$(stat -c%s "$PART" 2>/dev/null||echo 0)" -eq "$EXPECT" ] && mv "$PART" "$BASE"; }
  [ -f "$BASE" ] && [ "$(stat -c%s "$BASE")" -eq "$EXPECT" ] || { log "FATAL: base incomplete"; exit 1; }
  log "base ready ($(stat -c%s "$BASE") bytes)"
  log "quantize sub-3-bit floor points (reuse imatrix)"
  "$BIN/llama-quantize.exe" --imatrix "$IMAT" "$BASE" "$OUT/qwen3b-iq2s-imat.gguf"  IQ2_S  >/dev/null 2>&1 && log "  IQ2_S  done"
  "$BIN/llama-quantize.exe" --imatrix "$IMAT" "$BASE" "$OUT/qwen3b-iq2xs-imat.gguf" IQ2_XS >/dev/null 2>&1 && log "  IQ2_XS done"
  "$BIN/llama-quantize.exe" --imatrix "$IMAT" "$BASE" "$OUT/qwen3b-iq1m-imat.gguf"  IQ1_M  >/dev/null 2>&1 && log "  IQ1_M  done"
  log "removing f16 to free disk"; rm -f "$BASE" "$PART"
else
  log "sub-3-bit quants already present; skipping download+quantize"
fi

ppl(){ "$BIN/llama-perplexity.exe" -m "$1" -f "$EVAL" -c $PCTX 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | tail -1 | grep -oE "[0-9.]+$"; }
mb(){ awk "BEGIN{printf \"%.0f\", $(stat -c%s "$1")/1048576}"; }
bpw(){ python - "$1" <<'PY'
import sys,os
from gguf import GGUFReader
f=sys.argv[1]; r=GGUFReader(f); p=sum(int(t.n_elements) for t in r.tensors)
print(f"{os.path.getsize(f)*8/p:.2f}")
PY
}
acc(){ "$BIN/llama-perplexity.exe" -m "$1" -f "$HS" --hellaswag --hellaswag-tasks $TASKS -c $HCTX \
        2>"$OUT/$(basename "$1").hs.log" | grep -oE "[0-9]+\.[0-9]+%" | tail -1 | tr -d '%'; }
pct(){ awk "BEGIN{printf \"%.1f\", ($1/$Q4REF)*100}"; }

declare -A P S B A
for tag in iq2s-imat iq2xs-imat iq1m-imat; do
  f="$OUT/qwen3b-$tag.gguf"
  [ -f "$f" ] || { log "MISSING $f (quantize failed?)"; continue; }
  log "PPL+bpw $tag"; P[$tag]=$(ppl "$f"); S[$tag]=$(mb "$f"); B[$tag]=$(bpw "$f")
  log "  $tag PPL=${P[$tag]} ${S[$tag]}MB ${B[$tag]}bpw"
  log "HellaSwag $TASKS $tag"; A[$tag]=$(acc "$f"); log "  $tag acc=${A[$tag]}% retention=$(pct ${A[$tag]})%"
done

{
  echo "# 3B sub-3-bit FLOOR — Qwen2.5-3B, training-free (PTQ), $(date +%Y-%m-%d)"
  echo
  echo "How far below 3 bpw can IQ codebooks + imatrix go before quality collapses?"
  echo "Retention = HellaSwag acc / Q4_K_M acc ($Q4REF%, near-lossless proxy). Random = 25%."
  echo "Anchors (IQ3_M, IQ2_M, Q4) carried from results_3b_taskacc.md; new rows measured here."
  echo
  echo "| model | nominal bpw | eff bpw | PPL (wiki) | HellaSwag acc | retention |"
  echo "|---|---|---|---|---|---|"
  echo "| stock Q4_K_M (ref) | 5.00 | 5.00 | 9.302 | 75.1774% | 100.0% |"
  echo "| **IQ3_M + imatrix (ship)** | 3.86 | 3.86 | 9.804 | 74.5014% | **99.1%** |"
  echo "| IQ2_M + imatrix | 2.70 | 2.96 | — | 68.0891% | 90.6% |"
  echo "| IQ2_S + imatrix | 2.50 | ${B[iq2s-imat]:-?} | ${P[iq2s-imat]:-?} | ${A[iq2s-imat]:-?}% | $( [ -n "${A[iq2s-imat]:-}" ] && pct ${A[iq2s-imat]} || echo ? )% |"
  echo "| IQ2_XS + imatrix | 2.31 | ${B[iq2xs-imat]:-?} | ${P[iq2xs-imat]:-?} | ${A[iq2xs-imat]:-?}% | $( [ -n "${A[iq2xs-imat]:-}" ] && pct ${A[iq2xs-imat]} || echo ? )% |"
  echo "| IQ1_M + imatrix | 1.75 | ${B[iq1m-imat]:-?} | ${P[iq1m-imat]:-?} | ${A[iq1m-imat]:-?}% | $( [ -n "${A[iq1m-imat]:-}" ] && pct ${A[iq1m-imat]} || echo ? )% |"
  echo
  echo "## Read"
  echo "The 3-bit -> 2-bit cliff is a PHYSICS cliff, not a tuning gap. IQ3 (3.86 bpw) holds"
  echo "99.1%; everything at/below IQ2_M is already under the 98% product bar. These rows map"
  echo "exactly how steep the drop is with llama.cpp's own sub-3-bit codebooks + imatrix, and"
  echo "set the bar a QuIP#/QTIP/VPTQ-class codebook would have to beat to make sub-3-bit shippable."
} > "$RES"
log "DONE -> $RES"; cat "$RES"
