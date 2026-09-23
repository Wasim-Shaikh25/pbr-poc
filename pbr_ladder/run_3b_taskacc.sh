#!/usr/bin/env bash
# Task-accuracy for the 3B models: HellaSwag via llama.cpp (no lm-eval install).
# Turns "+5.4% PPL" into a real "retains X% of accuracy" number. Uses Q4_K_M as
# the near-lossless full-quality proxy (f16 base was deleted to save disk), so
# retention = model_acc / q4km_acc.
set -u
LLAMA_BIN="${LLAMA_BIN:?}"; REPO="${REPO:-/c/python_project/pbr-poc}"; WORK="${WORK:-/d/pbr_work}"
OUT="$WORK/out"; DATA="$WORK/hellaswag_val_full.txt"; TASKS="${TASKS:-1000}"; CTX=1024
RES="$REPO/pbr_ladder/results_3b_taskacc.md"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
acc(){ # model.gguf -> final HellaSwag accuracy (percent, no % sign)
  "$LLAMA_BIN/llama-perplexity.exe" -m "$1" -f "$DATA" --hellaswag \
    --hellaswag-tasks "$TASKS" -c $CTX 2>"$OUT/$(basename "$1").hs.log" \
    | grep -oE "[0-9]+\.[0-9]+%" | tail -1 | tr -d '%'
}

declare -A A
# reference first (the retention denominator)
for tag in stock-q4km iq3-imat stock-q3km stock-iq2m-imat; do
  f="$OUT/qwen3b-$tag.gguf"
  log "hellaswag $TASKS tasks: $tag"
  A[$tag]=$(acc "$f")
  log "  $tag acc = ${A[$tag]}%"
done

ref="${A[stock-q4km]}"
pct(){ awk "BEGIN{printf \"%.1f\", ($1/$ref)*100}"; }
{
  echo "# 3B task-accuracy — HellaSwag ($TASKS tasks), Qwen2.5-3B, $(date +%Y-%m-%d)"
  echo
  echo "Retention = model_acc / Q4_K_M_acc (Q4_K_M = near-lossless full-quality proxy;"
  echo "f16 base was deleted to save disk). HellaSwag random baseline = 25%."
  echo
  echo "| model | bpw | HellaSwag acc | retention vs Q4 |"
  echo "|---|---|---|---|"
  echo "| stock Q4_K_M (ref) | 5.00 | ${A[stock-q4km]}% | 100.0% |"
  echo "| **IQ3_M + imatrix (ship)** | 3.86 | **${A[iq3-imat]}%** | **$(pct ${A[iq3-imat]})%** |"
  echo "| stock Q3_K_M | 4.12 | ${A[stock-q3km]}% | $(pct ${A[stock-q3km]})% |"
  echo "| IQ2_M + imatrix | 2.96 | ${A[stock-iq2m-imat]}% | $(pct ${A[stock-iq2m-imat]})% |"
  echo
  echo "If IQ3+imatrix retention >= 98%, the sub-4-bit-at-quality claim is REAL on a"
  echo "downstream task (not just PPL). If below, IQ3 still holds quality better than"
  echo "the alternatives; report honestly and consider a higher-bit or per-tensor tune."
} > "$RES"
log "DONE -> $RES"; cat "$RES"
