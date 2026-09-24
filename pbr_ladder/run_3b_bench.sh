#!/usr/bin/env bash
# Decode-speed + effective-bandwidth bench on THIS machine.
# Decode is memory-bandwidth-bound: tok/s ~= mem_BW / model_bytes. So for each
# quant we measure generation tok/s (tg128) at several thread counts, then back
# out the ACHIEVED bandwidth = tok/s * model_bytes. That reveals (a) the real
# GB/s ceiling of this laptop, (b) the tok/s-vs-bits curve, (c) the thread count
# where bandwidth saturates and more cores stop helping.
# Waits for the sub-3-bit job to finish first (shares CPU), then runs.
set -u
BIN="/c/python_project/bonsai test/bin/llama.cpp"
REPO="/c/python_project/pbr-poc"; W="/d/pbr_work"; OUT="$W/out"
RES="$REPO/pbr_ladder/results_3b_bench.md"; JSON="$W/bench.json"
THREADS="4,8,16"; NGEN=128
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for sub-3-bit job to finish (grep DONE in sub3.out)"
until grep -q "DONE ->" "$W/sub3.out" 2>/dev/null; do sleep 60; done
log "sub-3 done; starting bench"

# bench each quant that exists, largest->smallest bpw; JSON for clean parse
models=(qwen3b-stock-q4km qwen3b-iq3-imat qwen3b-stock-iq2m-imat qwen3b-iq2s-imat qwen3b-iq2xs-imat qwen3b-iq1m-imat)
echo "[" > "$JSON"; first=1
for m in "${models[@]}"; do
  f="$OUT/$m.gguf"; [ -f "$f" ] || { log "skip (missing) $m"; continue; }
  log "bench $m (tg$NGEN, threads=$THREADS)"
  out=$("$BIN/llama-bench.exe" -m "$f" -p 0 -n $NGEN -t $THREADS -o json 2>>"$W/bench.err")
  # strip surrounding [ ] and append rows
  rows=$(echo "$out" | sed '1d;$d')
  [ -n "$rows" ] && { [ $first -eq 0 ] && echo "," >> "$JSON"; echo "$rows" >> "$JSON"; first=0; }
done
echo "]" >> "$JSON"

python - "$JSON" "$RES" <<'PY'
import json,sys,os
data=json.load(open(sys.argv[1]))
# group by model file
rows=[]
for r in data:
    fn=os.path.basename(r.get("model_filename",""))
    sz=int(r.get("model_size",0))            # bytes
    th=r.get("n_threads")
    tps=float(r.get("avg_ts",0))             # tokens/sec (generation)
    gbps=tps*sz/1e9                           # achieved bandwidth (weights read once/token)
    rows.append((fn,sz,th,tps,gbps))
# order for display
order=["qwen3b-stock-q4km","qwen3b-iq3-imat","qwen3b-stock-iq2m-imat",
       "qwen3b-iq2s-imat","qwen3b-iq2xs-imat","qwen3b-iq1m-imat"]
def key(fn):
    b=fn.replace(".gguf","")
    return order.index(b) if b in order else 99
rows.sort(key=lambda x:(key(x[0]), x[2]))
peak=max((r[4] for r in rows), default=0)
with open(sys.argv[2],"w") as o:
    o.write("# 3B decode speed + effective bandwidth — this laptop, %s\n\n"%__import__("datetime").date.today())
    o.write("Generation (tg%d) tok/s per quant per thread count. Achieved GB/s = tok/s x model_bytes\n"%128)
    o.write("(decode reads every weight once per token). Peak achieved BW ~= this machine's usable\n")
    o.write("memory bandwidth ceiling. Where tok/s stops rising with threads = bandwidth saturation.\n\n")
    o.write("| model | MB | threads | tok/s | achieved GB/s |\n|---|---|---|---|---|\n")
    for fn,sz,th,tps,gbps in rows:
        o.write("| %s | %.0f | %s | %.1f | %.1f |\n"%(fn.replace('.gguf',''),sz/1048576,th,tps,gbps))
    o.write("\n**Peak achieved bandwidth: %.1f GB/s** (this machine's practical ceiling).\n"%peak)
    o.write("\nTo hit 70-90 tok/s you need model_bytes <= peak_GBps/target_tps. ")
    if peak>0:
        o.write("At %.0f GB/s: 70 tok/s => <= %.2f GB model; 90 tok/s => <= %.2f GB.\n"%(peak,peak/70,peak/90))
print(open(sys.argv[2]).read())
PY
log "DONE -> $RES"
