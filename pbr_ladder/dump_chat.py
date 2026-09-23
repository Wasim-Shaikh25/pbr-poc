# Convert this session's transcript JSONL -> readable markdown (user + assistant text,
# with tool calls summarized compactly). Does not load the whole thing into anyone's
# context; streams line by line.
import json, glob, os, datetime
DIR = r"C:\Users\Shaikh Wasim\.claude\projects\C--python-project-pbr-poc"
import sys
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
files = sorted(glob.glob(os.path.join(DIR, f"{pat}.jsonl")), key=os.path.getmtime)
print("found:", [os.path.basename(f) for f in files])

def text_of(content):
    """Extract readable text + tool markers from a message content field."""
    if isinstance(content, str):
        return content.strip()
    out = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                out.append(str(b)); continue
            t = b.get("type")
            if t == "text":
                out.append(b.get("text", "").strip())
            elif t == "thinking":
                pass  # skip internal thinking
            elif t == "tool_use":
                inp = b.get("input", {})
                desc = inp.get("description") or inp.get("command") or inp.get("file_path") or ""
                desc = str(desc).replace("\n", " ")[:120]
                out.append(f"> 🔧 **{b.get('name')}** — {desc}")
            elif t == "tool_result":
                c = b.get("content", "")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                c = str(c).replace("\n", " ")
                if c.strip():
                    out.append(f"> _result:_ {c[:200]}")
    return "\n\n".join(s for s in out if s)

lines_md = ["# PBR-Ladder Session Transcript",
            f"_exported {datetime.datetime.now():%Y-%m-%d %H:%M}_", ""]
seen = 0
for f in files:
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            try: o = json.loads(line)
            except Exception: continue
            typ = o.get("type", "")
            msg = o.get("message", {})
            role = msg.get("role") if isinstance(msg, dict) else None
            if typ == "summary":
                lines_md.append(f"\n## [context summary]\n\n{o.get('summary','')}\n"); seen += 1; continue
            if role not in ("user", "assistant"):
                continue
            body = text_of(msg.get("content"))
            # skip pure system-reminder / local-command noise and empty
            if not body: continue
            if body.startswith("<") and "system-reminder" in body[:60]: continue
            label = "🧑 User" if role == "user" else "🤖 Claude"
            lines_md.append(f"\n### {label}\n\n{body}\n"); seen += 1

tag = "THIS" if pat != "*" else "ALL"
out_path = os.path.join(r"C:\python_project\pbr-poc\pbr_ladder", f"SESSION_CHAT_{tag}.md")
with open(out_path, "w", encoding="utf-8") as w:
    w.write("\n".join(lines_md))
print(f"wrote {out_path}  ({seen} entries, {os.path.getsize(out_path)/1024:.0f} KB)")
