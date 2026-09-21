"""Re-run push_below4 (and push_below3) on the 9 Phase 0 tensors with real acts.

Logs raw stdout. Compares push_below4 lines to pbr_ladder/real_tensor_results.txt.
"""
import os
import re
import subprocess
import sys
import time

ROOT = "/workspace/pbr_ladder"
LOG = "/workspace/artifacts/pbr_ladder/phase0_push_below4.log"
PRIOR = os.path.join(ROOT, "real_tensor_results.txt")
M = os.path.join(ROOT, "qwen05b/model.safetensors")

TENSORS = []
for layer in (2, 12, 21):
    for module, short in (
        ("mlp.gate_proj", "gate"),
        ("mlp.down_proj", "down"),
        ("self_attn.q_proj", "q"),
    ):
        TENSORS.append(
            (
                f"L{layer} {module}",
                f"model.layers.{layer}.{module}.weight",
                os.path.join(ROOT, f"acts_L{layer}_{short}.npy"),
            )
        )


def run(script, tensor, acts, logf):
    cmd = [sys.executable, script, M, tensor, acts]
    env = os.environ.copy()
    env["PBR_CHUNK"] = env.get("PBR_CHUNK", "50000")
    env["PYTHONUNBUFFERED"] = "1"
    logf.write(f"\n$ {' '.join(cmd)}\n")
    logf.flush()
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logf.write(p.stdout)
    if not p.stdout.endswith("\n"):
        logf.write("\n")
    logf.write(f"[exit {p.returncode} elapsed {time.time()-t0:.1f}s]\n")
    logf.flush()
    return p.returncode, p.stdout


def parse_push4_blocks(text):
    """Map header '########## NAME ##########' -> list of push_below4 data lines."""
    blocks = {}
    parts = re.split(r"########## (.+?) ##########\n", text)
    # parts[0] preamble, then name, body, name, body...
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1]
        m = re.search(r"--- push_below4 ---\n(.*?)(?:--- |\Z)", body, re.S)
        if not m:
            blocks[name] = None
            continue
        rows = []
        for line in m.group(1).splitlines():
            line = line.rstrip()
            if not line or line.startswith("method") or line.startswith("Traceback") or line.startswith(" "):
                continue
            # name may contain spaces; last 3 fields are numbers
            mm = re.match(r"(.+?)\s{2,}(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", line)
            if mm:
                rows.append((mm.group(1).strip(), float(mm.group(2)), float(mm.group(3)), float(mm.group(4))))
        blocks[name] = rows
    return blocks


def parse_fresh(stdout):
    rows = []
    for line in stdout.splitlines():
        mm = re.match(r"(.+?)\s{2,}(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", line.rstrip())
        if mm and not line.startswith("method"):
            rows.append((mm.group(1).strip(), float(mm.group(2)), float(mm.group(3)), float(mm.group(4))))
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "below4"
    scripts = []
    if which in ("below4", "both"):
        scripts.append("push_below4.py")
    if which in ("below3", "both"):
        scripts.append("push_below3.py")
    prior_text = open(PRIOR, encoding="utf-8").read()
    prior = parse_push4_blocks(prior_text)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    summary = []
    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n===== sweep {which} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for label, tensor, acts in TENSORS:
            if not os.path.exists(acts):
                logf.write(f"MISSING ACTS {acts}\n")
                summary.append((label, "MISSING_ACTS", None))
                continue
            for script in scripts:
                logf.write(f"\n########## {label} ##########\n--- {script} ---\n")
                code, out = run(script, tensor, acts, logf)
                if script != "push_below4.py":
                    summary.append((label, script, "ok" if code == 0 else f"exit {code}"))
                    continue
                fresh = parse_fresh(out)
                old = prior.get(label)
                if old is None:
                    status = "NO_PRIOR_PUSH4 (prior crashed or absent)"
                    summary.append((label, status, fresh))
                    logf.write(f"COMPARE {label}: {status}\n")
                    continue
                if code != 0:
                    summary.append((label, f"CRASH exit {code}", None))
                    logf.write(f"COMPARE {label}: CRASH\n")
                    continue
                mismatches = []
                if len(fresh) != len(old):
                    mismatches.append(f"row count {len(fresh)} vs {len(old)}")
                for a, b in zip(fresh, old):
                    if a[0] != b[0] or abs(a[1] - b[1]) > 0.001 or abs(a[2] - b[2]) > 0.02 or abs(a[3] - b[3]) > 0.02:
                        mismatches.append(f"{a[0]}: new {a[1]:.3f} {a[2]:.2f} {a[3]:.2f} vs old {b[1]:.3f} {b[2]:.2f} {b[3]:.2f}")
                status = "MATCH" if not mismatches else "DIFFER"
                summary.append((label, status, mismatches))
                logf.write(f"COMPARE {label}: {status}\n")
                for m in mismatches:
                    logf.write(f"  {m}\n")
    print("SUMMARY")
    for row in summary:
        print(row[0], row[1])
        if row[1] == "DIFFER":
            for m in row[2]:
                print("   ", m)


if __name__ == "__main__":
    main()
