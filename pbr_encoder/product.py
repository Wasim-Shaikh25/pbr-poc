"""PBR-E product CLI: bit-exact ~30% BF16 checkpoint compression.

Subcommands: compress, decompress, verify.
Default codec profile is PBR-E (raw + exponent Huffman/rANS). Expected
~10.6–10.7 BPW on dense LLMs (DF11/ZipNN-class, not novel, not ≤4 BPW).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from pbr_core.container import PBRContainer, TensorBlob
from pbr_core.hashing import sha256_bytes
from pbr_core.metrics import bits_per_weight, compression_ratio
from pbr_core.safetensors_io import (
    TWO_BYTE_DTYPES,
    inventory_model,
    load_uint16,
    write_uint16_safetensors,
)
from pbr_core.tiles import as_2d
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor, mode_usage
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO, download_checkpoint
from pbr_encoder.profiles import PROFILES
from pbr_encoder.verification import ExactnessError, assert_exact

PRODUCT = "pbr-e"
DISCLAIMER = (
    "PBR-E product: bit-exact uint16 BF16. Expected ~10.6–10.7 BPW on dense "
    "LLMs (~30% smaller than raw BF16). DF11/ZipNN-class, not a novel rate. "
    "Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW."
)
SIDECARS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
)
WEIGHTS_NAME = "weights.pbr"
MANIFEST_NAME = "manifest.json"


def _profile(name: str = "pbre_whole") -> dict:
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name}. Try: {', '.join(sorted(PROFILES))}")
    return PROFILES[name]


def _select_specs(model_dir: Path, *, all_16bit: bool):
    specs = inventory_model(model_dir)
    chosen = [s for s in specs if s.dtype in TWO_BYTE_DTYPES]
    if not all_16bit:
        from pbr_encoder.hf_weights import select_weight_specs

        chosen = select_weight_specs(
            specs, min_bytes=100 * 1024 * 1024, max_bytes=167772160, include_embeddings=False
        )
    if not chosen:
        raise SystemExit(f"no 16-bit tensors in {model_dir}")
    return chosen


def _copy_sidecars(model_dir: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SIDECARS:
        src = model_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
    return copied


def _resolve_archive(path: Path) -> tuple[Path, Path | None]:
    """Return (weights.pbr, bundle_dir or None)."""
    if path.is_dir():
        return path / WEIGHTS_NAME, path
    return path, path.parent if path.suffix == ".pbr" else None


def compress(
    *,
    model_dir: Path,
    output: Path,
    profile_name: str = "pbre_whole",
    block_size: int = 65535,
    all_16bit: bool = True,
    copy_sidecars: bool = True,
    repo_id: str = PRIMARY_REPO,
    revision: str = "",
) -> dict:
    prof = _profile(profile_name)
    specs = _select_specs(model_dir, all_16bit=all_16bit)
    blobs: list[TensorBlob] = []
    logical: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    sources: dict[str, str] = {}
    orig_bytes = 0
    t0 = time.perf_counter()
    print(DISCLAIMER, flush=True)
    print(f"compress {len(specs)} tensors from {model_dir}  profile={profile_name}", flush=True)
    for spec in specs:
        words = load_uint16(spec)
        logical[spec.name] = [int(x) for x in spec.shape]
        dtypes[spec.name] = spec.dtype
        sources[spec.name] = spec.file_path.name
        orig_bytes += int(spec.nbytes)
        print(f"  {spec.name}  {list(spec.shape)}  {spec.dtype}  {spec.nbytes} B", flush=True)
        one = encode_tensor(
            as_2d(words),
            name=spec.name,
            block_size=block_size,
            codecs=prof["codecs"],
            whole_codecs=prof["whole_codecs"],
            enable_whole=bool(prof["enable_whole"]),
            extra={"product": PRODUCT, "original_shape": list(spec.shape)},
        )
        blobs.extend(one.tensors)
        del words, one
    elapsed = time.perf_counter() - t0
    extra = {
        "product": PRODUCT,
        "format": "PBR-E product",
        "note": DISCLAIMER,
        "profile": profile_name,
        "profile_label": prof["label"],
        "logical_shapes": logical,
        "dtypes": dtypes,
        "source_files": sources,
        "original_bytes": orig_bytes,
        "repo_id": repo_id,
        "revision": revision,
        "n_tensors": len(specs),
        "encode_seconds": elapsed,
    }
    container = PBRContainer(tensors=blobs, extra=extra)
    blob = container.dumps()
    n_words = sum(t.n_words for t in blobs)
    usage = mode_usage(container)

    if output.suffix.lower() == ".pbr" and not output.is_dir():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(blob)
        weights_path = output
        bundle = None
        copied: list[str] = []
    else:
        output.mkdir(parents=True, exist_ok=True)
        weights_path = output / WEIGHTS_NAME
        weights_path.write_bytes(blob)
        bundle = output
        copied = _copy_sidecars(model_dir, output) if copy_sidecars else []

    if len(blob) != weights_path.stat().st_size:
        raise RuntimeError("on-disk size != dumps()")

    restored = decode_container(container)
    for spec, rec, tb in zip(specs, restored, blobs):
        orig = load_uint16(spec)
        rec = rec.reshape(tuple(logical[spec.name]))
        assert_exact(orig, rec, label=spec.name)

    summary = {
        "disclaimer": DISCLAIMER,
        "product": PRODUCT,
        "profile": profile_name,
        "n_tensors": len(specs),
        "n_words": n_words,
        "original_bytes": orig_bytes,
        "encoded_bytes": len(blob),
        "bpw": bits_per_weight(len(blob), n_words),
        "ratio_vs_raw_bf16": compression_ratio(len(blob), orig_bytes),
        "saved_frac": 1.0 - compression_ratio(len(blob), orig_bytes),
        "exact": "PASS",
        "container_sha256": sha256_bytes(blob),
        "mode_usage": usage,
        "encode_seconds": elapsed,
        "weights_path": str(weights_path),
        "sidecars": copied,
        "repo_id": repo_id,
        "revision": revision,
    }
    man = {**summary, "logical_shapes": logical, "dtypes": dtypes, "source_files": sources}
    man_path = (bundle / MANIFEST_NAME) if bundle is not None else Path(str(weights_path) + ".manifest.json")
    man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {weights_path}  {len(blob)} B  {summary['bpw']:.4f} BPW  "
        f"ratio={summary['ratio_vs_raw_bf16']:.4f}  exact=PASS",
        flush=True,
    )
    return summary


def _logical_shape(container: PBRContainer, tensor: TensorBlob) -> tuple[int, ...]:
    shapes = (container.extra or {}).get("logical_shapes") or {}
    if tensor.name in shapes:
        return tuple(int(x) for x in shapes[tensor.name])
    extra_shape = (container.extra or {}).get("original_shape")
    if extra_shape:
        return tuple(int(x) for x in extra_shape)
    return tuple(int(x) for x in tensor.shape)


def decompress(*, archive: Path, output_dir: Path) -> dict:
    weights_path, bundle = _resolve_archive(archive)
    if not weights_path.is_file():
        raise SystemExit(f"missing {weights_path}")
    container = PBRContainer.loads(weights_path.read_bytes())
    restored = decode_container(container)
    extra = container.extra or {}
    dtypes = extra.get("dtypes") or {}
    sources = extra.get("source_files") or {}
    by_file: dict[str, dict[str, np.ndarray]] = {}
    dtype_by_file: dict[str, dict[str, str]] = {}
    for rec, tb in zip(restored, container.tensors):
        shp = _logical_shape(container, tb)
        arr = rec.reshape(shp)
        fname = sources.get(tb.name, "model.safetensors")
        by_file.setdefault(fname, {})[tb.name] = arr
        dtype_by_file.setdefault(fname, {})[tb.name] = dtypes.get(tb.name, tb.dtype if tb.dtype in TWO_BYTE_DTYPES else "BF16")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, tensors in by_file.items():
        dest = output_dir / fname
        tags = dtype_by_file[fname]
        default = next(iter(tags.values())) if tags else "BF16"
        write_uint16_safetensors(
            dest,
            tensors,
            storage_dtype=default,
            dtypes=tags,
            metadata={"pbr_product": PRODUCT},
        )
        written.append(str(dest))
    copied = []
    if bundle is not None:
        for name in SIDECARS:
            src = bundle / name
            if src.is_file():
                shutil.copy2(src, output_dir / name)
                copied.append(name)
    print(f"decompress {weights_path} -> {output_dir}  tensors={len(container.tensors)}  files={written}")
    return {"tensors": len(container.tensors), "files": written, "sidecars": copied, "exact": "PASS"}


def verify(*, archive: Path, model_dir: Path) -> dict:
    weights_path, _bundle = _resolve_archive(archive)
    container = PBRContainer.loads(weights_path.read_bytes())
    restored = decode_container(container)
    extra = container.extra or {}
    specs = {s.name: s for s in inventory_model(model_dir) if s.dtype in TWO_BYTE_DTYPES}
    n_ok = 0
    records = []
    for rec, tb in zip(restored, container.tensors):
        spec = specs.get(tb.name)
        if spec is None:
            raise SystemExit(f"verify miss: {tb.name} not in {model_dir}")
        orig = load_uint16(spec)
        shp = _logical_shape(container, tb)
        rec = rec.reshape(shp)
        result = assert_exact(orig, rec, label=tb.name)
        n_ok += 1
        records.append({"name": tb.name, "sha256": result.original_sha256, "exact": "PASS"})
    if n_ok != len(container.tensors):
        raise SystemExit("verify tensor count mismatch")
    extra_names = set(specs) - {t.name for t in container.tensors}
    blob = weights_path.read_bytes()
    n_words = sum(t.n_words for t in container.tensors)
    orig_b = int(extra.get("original_bytes") or sum(specs[t.name].nbytes for t in container.tensors))
    summary = {
        "disclaimer": DISCLAIMER,
        "exact": "PASS",
        "n_tensors": n_ok,
        "n_words": n_words,
        "encoded_bytes": len(blob),
        "original_bytes": orig_b,
        "bpw": bits_per_weight(len(blob), n_words),
        "ratio_vs_raw_bf16": compression_ratio(len(blob), orig_b),
        "container_sha256": sha256_bytes(blob),
        "tensors_not_in_archive": sorted(extra_names),
        "records": records,
    }
    print(
        f"verify PASS  {n_ok} tensors  SHA-256 match  "
        f"{summary['bpw']:.4f} BPW  ratio={summary['ratio_vs_raw_bf16']:.4f}"
    )
    return summary


def _resolve_model_dir(args) -> tuple[Path, str, str]:
    if args.model_dir is not None:
        return Path(args.model_dir), args.repo or PRIMARY_REPO, args.revision or ""
    local_root = Path(args.local_dir)
    cache = Path(args.cache_dir)
    downloaded = download_checkpoint(
        repo_id=args.repo or PRIMARY_REPO,
        revision=args.revision or "main",
        local_dir=local_root,
        cache_dir=cache,
        allow_fallback=not args.no_fallback,
    )
    print(f"checkpoint {downloaded.repo_id} commit={downloaded.commit} fallback={downloaded.fallback_used}")
    return downloaded.local_dir, downloaded.repo_id, downloaded.commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PBR-E: bit-exact ~30% BF16 compression (DF11-class, ~10.6 BPW)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def src_flags(p):
        p.add_argument("--model-dir", type=Path, default=None, help="Local safetensors directory")
        p.add_argument("--repo", default=None, help=f"Hugging Face id (default {PRIMARY_REPO})")
        p.add_argument("--revision", default=None)
        p.add_argument("--local-dir", type=Path, default=Path("outputs/models"))
        p.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
        p.add_argument("--no-fallback", action="store_true")

    p_c = sub.add_parser("compress", help="Encode 16-bit tensors with PBR-E")
    src_flags(p_c)
    p_c.add_argument("-o", "--output", type=Path, required=True, help="weights.pbr file or bundle directory")
    p_c.add_argument("--profile", default="pbre_whole", help="Codec profile (default pbre_whole)")
    p_c.add_argument("--block-size", type=int, default=65535)
    p_c.add_argument("--linear-window", action="store_true", help="Stage 1B linear subset instead of all 16-bit")
    p_c.add_argument("--no-sidecars", action="store_true")

    p_d = sub.add_parser("decompress", help="Restore safetensors from a PBR-E archive")
    p_d.add_argument("archive", type=Path)
    p_d.add_argument("-o", "--output", type=Path, required=True)

    p_v = sub.add_parser("verify", help="SHA-256 roundtrip vs a checkpoint directory")
    p_v.add_argument("archive", type=Path)
    src_flags(p_v)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "compress":
            model_dir, repo, rev = _resolve_model_dir(args)
            compress(
                model_dir=model_dir,
                output=args.output,
                profile_name=args.profile,
                block_size=args.block_size,
                all_16bit=not args.linear_window,
                copy_sidecars=not args.no_sidecars,
                repo_id=repo,
                revision=rev,
            )
            return 0
        if args.cmd == "decompress":
            decompress(archive=args.archive, output_dir=args.output)
            return 0
        model_dir, _repo, _rev = _resolve_model_dir(args)
        verify(archive=args.archive, model_dir=model_dir)
        return 0
    except ExactnessError as exc:
        print(f"PBR-E FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
