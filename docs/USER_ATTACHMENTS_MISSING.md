# User Matrix Mantissa attachments — not on this VM

The follow-up asked to materialize files from:

`/home/box/agent-data/agents/48a5af41-48b7-41f8-bc13-38db03ae9fd5/attachments/`

That path **does not exist** on this cloud VM (`/home/box` is absent).
Directory listing returned empty. This agent is `bc-7ab84e2f-…`, not
`48a5af41-…`.

## Expected mapping (from the user message)

| attachment (hash prefix) | intended repo path |
| --- | --- |
| `8e51207f…bcbccc79.md` | README fragment / quick start |
| `5ea6069dc995…ccc79.py` | `pbr_poc.py` (codec core) |
| `a6f16719…89bff7.py` | `pbr_edge_test.py` |
| `710d9515…73fe.json` | reference `pbr_edge_results.json` |
| `8d300ecd…d02a7.txt` | `docs/EDGE_TEST_EXECUTION.txt` |
| `8699af8e…3efa.docx` | `docs/` guide (convert to md if possible) |

## Searched (no matches)

- `/home/box/agent-data/.../attachments/`
- `/home/ubuntu/.cursor/projects/workspace/uploads/` (only three older Stage-1A PDFs)
- `/tmp/cursor/attachments`, `/opt/cursor/attachments`
- this run’s transcript (paths/hashes only; **no file bodies**)
- GitHub `Wasim-Shaikh25/pbr-poc` and code search for `pbr_poc.py`

## What this PR does *not* contain

- No reinvented `pbr_poc.py`
- No Qwen benchmark
- No Gates A–C results from the **user** suite (`python pbr_edge_test.py` cannot be run without that file)
- **Gate D still required** once the package lands

A separate reconstruction (not the user package) is in
https://github.com/Wasim-Shaikh25/pbr-poc/pull/2 — do not treat that as a
substitute for these attachments.

Re-attach the six files onto this cloud agent (or drop them in the repo)
and a follow-up can copy them onto this branch unchanged, fix imports, and
re-run the edge suite.
