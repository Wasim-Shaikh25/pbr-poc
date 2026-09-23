#!/usr/bin/env bash
# Small end-to-end validation of the recipe-on-GGUF pipeline (no kernel, no infra).
# Proves: base f16 -> imatrix -> quantize (stock vs stock+imatrix vs our recipe) ->
# perplexity + file size (= packed in-RAM size via mmap). Fast slice for speed.
# Usage: bash run_validation.sh <path-to-base-fp16.gguf>
set -u
SP="C:/Users/SHAIKH~1/AppData/Local/Temp/claude/C--python-project-pbr-poc/b6937007-7a3a-400e-aa4c-732d242b3438/scratchpad"
T="$SP/tools/llama"
BASE="$1"
OUT="$SP/ggufval"; mkdir -p "$OUT"
CTX=512

echo "### 1. imatrix (importance matrix from our calib slice) ###"
"$T/llama-imatrix.exe" -m "$BASE" -f "$SP/calib.txt" -o "$OUT/imatrix.dat" --chunks 32 2>&1 | tail -3

echo "### 2. quantize three ways at Q3_K_M ###"
echo "-- 2a stock (no imatrix) --"
"$T/llama-quantize.exe" "$BASE" "$OUT/stock_q3km.gguf" Q3_K_M 2>&1 | tail -1
echo "-- 2b stock + imatrix --"
"$T/llama-quantize.exe" --imatrix "$OUT/imatrix.dat" "$BASE" "$OUT/imat_q3km.gguf" Q3_K_M 2>&1 | tail -1
echo "-- 2c OUR RECIPE: imatrix + q8_0 embeddings + q8_0 output --"
"$T/llama-quantize.exe" --imatrix "$OUT/imatrix.dat" --token-embedding-type q8_0 --output-tensor-type q8_0 "$BASE" "$OUT/recipe_q3km.gguf" Q3_K_M 2>&1 | tail -1

echo "### 3. perplexity (same eval slice, ctx=$CTX) ###"
for f in stock_q3km imat_q3km recipe_q3km; do
  echo "-- PPL $f --"
  "$T/llama-perplexity.exe" -m "$OUT/$f.gguf" -f "$SP/wiki_test.txt" -c $CTX 2>&1 | grep -iE "Final estimate|PPL|perplexity" | tail -2
done

echo "### 4. packed sizes (= in-RAM footprint under mmap) ###"
ls -la "$OUT"/*.gguf
echo "### base f16 size ###"; ls -la "$BASE"
