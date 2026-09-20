"""Adaptive exact exponent codecs for H95Q container v2.

Modes (compete by complete physical section bytes per tensor):
  EXP_RAW8       — raw u8 fallback (8 BPW)
  EXP_RANS       — frequency table + rANS
  EXP_HUFFMAN    — canonical Huffman
  EXP_DELTA_RANS — delta along traversal (uint8 wrap) + rANS
  EXP_RUN_RANS   — run-length then rANS on values + lengths

Complete section layout (all modes)::

    mode:u8
    + mode-specific payload (tables, lengths, bitstream / raw)

Reported section bytes must equal physical ``exp_len``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from pbr_core.huffman import dump_table, load_table, table_from_symbols as huff_table_from_symbols
from pbr_core.rans import (
    dump_freq_table,
    load_freq_table,
    rans_decode,
    rans_encode,
    table_from_symbols as rans_table_from_symbols,
)
from pbr_h95.bitpack import pack_exp_u8, unpack_exp_u8

EXP_RAW8 = 0
EXP_RANS = 1
EXP_HUFFMAN = 2
EXP_DELTA_RANS = 3
EXP_RUN_RANS = 4

MODE_NAMES = {
    EXP_RAW8: "EXP_RAW8",
    EXP_RANS: "EXP_RANS",
    EXP_HUFFMAN: "EXP_HUFFMAN",
    EXP_DELTA_RANS: "EXP_DELTA_RANS",
    EXP_RUN_RANS: "EXP_RUN_RANS",
}

# Tiny tensors: skip expensive modes that cannot beat RAW8 after overhead.
_MIN_N_FOR_ENTROPY = 64
_MAX_RUN_LEN = 255


@dataclass(frozen=True)
class ExpEncodeResult:
    mode: int
    blob: bytes
    n_weights: int
    table_bytes: int
    stream_bytes: int
    mode_id_bytes: int
    length_field_bytes: int
    complete_bytes: int

    @property
    def mode_name(self) -> str:
        return MODE_NAMES.get(self.mode, f"MODE_{self.mode}")

    @property
    def complete_bpw(self) -> float:
        if self.n_weights <= 0:
            return 0.0
        return self.complete_bytes * 8.0 / self.n_weights

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mode_name": self.mode_name,
            "n_weights": self.n_weights,
            "complete_bytes": self.complete_bytes,
            "complete_bpw": round(self.complete_bpw, 6),
            "table_bytes": self.table_bytes,
            "stream_bytes": self.stream_bytes,
            "mode_id_bytes": self.mode_id_bytes,
            "length_field_bytes": self.length_field_bytes,
        }


def _as_u8(exp: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(exp, dtype=np.uint8).ravel()


def delta_u8(exp: np.ndarray) -> np.ndarray:
    """Signed wrap delta: d[0]=e[0], d[i]=(e[i]-e[i-1]) & 0xFF."""
    flat = _as_u8(exp)
    if flat.size == 0:
        return flat
    out = np.empty_like(flat)
    out[0] = flat[0]
    if flat.size > 1:
        out[1:] = (flat[1:].astype(np.int16) - flat[:-1].astype(np.int16)) & np.int16(0xFF)
        out[1:] = out[1:].astype(np.uint8)
    return out


def undelta_u8(deltas: np.ndarray) -> np.ndarray:
    flat = _as_u8(deltas)
    if flat.size == 0:
        return flat
    acc = np.cumsum(flat.astype(np.int32), dtype=np.int32) & np.int32(0xFF)
    return acc.astype(np.uint8)


def run_length_encode_u8(exp: np.ndarray, max_run: int = _MAX_RUN_LEN) -> tuple[np.ndarray, np.ndarray]:
    flat = _as_u8(exp)
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8)
    values: list[int] = []
    lengths: list[int] = []
    i = 0
    n = int(flat.size)
    while i < n:
        v = int(flat[i])
        j = i + 1
        lim = min(n, i + max_run)
        while j < lim and int(flat[j]) == v:
            j += 1
        values.append(v)
        lengths.append(j - i)
        i = j
    return np.asarray(values, dtype=np.uint8), np.asarray(lengths, dtype=np.uint8)


def run_length_decode_u8(values: np.ndarray, lengths: np.ndarray, n: int) -> np.ndarray:
    vals = _as_u8(values)
    lens = _as_u8(lengths)
    if vals.size != lens.size:
        raise ValueError("run value/length mismatch")
    out = np.empty(n, dtype=np.uint8)
    pos = 0
    for v, L in zip(vals.tolist(), lens.tolist()):
        L = int(L)
        if L <= 0 or pos + L > n:
            raise ValueError("bad run length")
        out[pos : pos + L] = np.uint8(v)
        pos += L
    if pos != n:
        raise ValueError(f"run decode size {pos} != {n}")
    return out


def _pack_rans_payload(stream: np.ndarray) -> tuple[bytes, int, int, int]:
    """Return (payload_without_mode, table_bytes, stream_bytes, length_field_bytes)."""
    if stream.size == 0:
        # mode handled outside; empty: table_len=0 stream_len=0
        return struct.pack("<II", 0, 0), 0, 0, 8
    freq = rans_table_from_symbols(stream)
    table = dump_freq_table(freq)
    bitstream = rans_encode(stream, freq)
    payload = struct.pack("<II", len(table), len(bitstream)) + table + bitstream
    return payload, len(table), len(bitstream), 8


def _unpack_rans_payload(data: bytes, offset: int, n: int) -> tuple[np.ndarray, int]:
    table_len, stream_len = struct.unpack_from("<II", data, offset)
    offset += 8
    if table_len == 0 and n == 0:
        return np.zeros(0, dtype=np.uint8), offset
    freq, end = load_freq_table(data, offset)
    if end - offset != table_len:
        raise ValueError("rANS table length mismatch")
    bitstream = data[end : end + stream_len]
    if len(bitstream) != stream_len:
        raise ValueError("rANS stream truncated")
    return rans_decode(bitstream, n, freq), end + stream_len


def _pack_huffman_payload(stream: np.ndarray) -> tuple[bytes, int, int, int]:
    if stream.size == 0:
        return struct.pack("<II", 0, 0), 0, 0, 8
    table = huff_table_from_symbols(stream)
    codebook = dump_table(table)
    bitstream = table.encode_symbols(stream)
    payload = struct.pack("<II", len(codebook), len(bitstream)) + codebook + bitstream
    return payload, len(codebook), len(bitstream), 8


def _unpack_huffman_payload(data: bytes, offset: int, n: int) -> tuple[np.ndarray, int]:
    table_len, stream_len = struct.unpack_from("<II", data, offset)
    offset += 8
    if table_len == 0 and n == 0:
        return np.zeros(0, dtype=np.uint8), offset
    table, end = load_table(data, offset)
    if end - offset != table_len:
        raise ValueError("Huffman table length mismatch")
    bitstream = data[end : end + stream_len]
    if len(bitstream) != stream_len:
        raise ValueError("Huffman stream truncated")
    return table.decode_symbols(bitstream, n), end + stream_len


def encode_raw8(exp: np.ndarray) -> ExpEncodeResult:
    flat = _as_u8(exp)
    raw = pack_exp_u8(flat)
    blob = bytes([EXP_RAW8]) + raw
    return ExpEncodeResult(
        mode=EXP_RAW8,
        blob=blob,
        n_weights=int(flat.size),
        table_bytes=0,
        stream_bytes=len(raw),
        mode_id_bytes=1,
        length_field_bytes=0,
        complete_bytes=len(blob),
    )


def encode_rans(exp: np.ndarray) -> ExpEncodeResult:
    flat = _as_u8(exp)
    payload, tb, sb, lb = _pack_rans_payload(flat)
    blob = bytes([EXP_RANS]) + payload
    return ExpEncodeResult(
        mode=EXP_RANS,
        blob=blob,
        n_weights=int(flat.size),
        table_bytes=tb,
        stream_bytes=sb,
        mode_id_bytes=1,
        length_field_bytes=lb,
        complete_bytes=len(blob),
    )


def encode_huffman(exp: np.ndarray) -> ExpEncodeResult:
    flat = _as_u8(exp)
    payload, tb, sb, lb = _pack_huffman_payload(flat)
    blob = bytes([EXP_HUFFMAN]) + payload
    return ExpEncodeResult(
        mode=EXP_HUFFMAN,
        blob=blob,
        n_weights=int(flat.size),
        table_bytes=tb,
        stream_bytes=sb,
        mode_id_bytes=1,
        length_field_bytes=lb,
        complete_bytes=len(blob),
    )


def encode_delta_rans(exp: np.ndarray) -> ExpEncodeResult:
    flat = _as_u8(exp)
    deltas = delta_u8(flat)
    payload, tb, sb, lb = _pack_rans_payload(deltas)
    blob = bytes([EXP_DELTA_RANS]) + payload
    return ExpEncodeResult(
        mode=EXP_DELTA_RANS,
        blob=blob,
        n_weights=int(flat.size),
        table_bytes=tb,
        stream_bytes=sb,
        mode_id_bytes=1,
        length_field_bytes=lb,
        complete_bytes=len(blob),
    )


def encode_run_rans(exp: np.ndarray) -> ExpEncodeResult:
    flat = _as_u8(exp)
    values, lengths = run_length_encode_u8(flat)
    n_runs = int(values.size)
    # Header: n_runs:u32 + values rANS + lengths rANS
    v_pay, v_tb, v_sb, v_lb = _pack_rans_payload(values)
    l_pay, l_tb, l_sb, l_lb = _pack_rans_payload(lengths)
    payload = struct.pack("<I", n_runs) + v_pay + l_pay
    blob = bytes([EXP_RUN_RANS]) + payload
    return ExpEncodeResult(
        mode=EXP_RUN_RANS,
        blob=blob,
        n_weights=int(flat.size),
        table_bytes=v_tb + l_tb,
        stream_bytes=v_sb + l_sb,
        mode_id_bytes=1,
        length_field_bytes=4 + v_lb + l_lb,
        complete_bytes=len(blob),
    )


def encode_exp_adaptive(
    exp: np.ndarray,
    *,
    modes: list[int] | None = None,
) -> ExpEncodeResult:
    """Encode exponents; pick the mode with minimum complete physical bytes."""
    flat = _as_u8(exp)
    n = int(flat.size)
    if modes is None:
        modes = [EXP_RAW8, EXP_RANS, EXP_HUFFMAN, EXP_DELTA_RANS, EXP_RUN_RANS]

    encoders = {
        EXP_RAW8: encode_raw8,
        EXP_RANS: encode_rans,
        EXP_HUFFMAN: encode_huffman,
        EXP_DELTA_RANS: encode_delta_rans,
        EXP_RUN_RANS: encode_run_rans,
    }

    candidates: list[ExpEncodeResult] = []
    # Always include RAW8.
    if EXP_RAW8 in modes:
        candidates.append(encode_raw8(flat))

    # Skip entropy modes on tiny tensors — overhead cannot win.
    do_entropy = n >= _MIN_N_FOR_ENTROPY
    for m in modes:
        if m == EXP_RAW8:
            continue
        if not do_entropy and m != EXP_RAW8:
            continue
        # RUN_RANS only when runs likely help (low unique or long plateaus).
        if m == EXP_RUN_RANS and n >= _MIN_N_FOR_ENTROPY:
            # Cheap reject: if mean run length < 2, skip.
            # Sample up to 64k for speed.
            sample = flat if n <= 65536 else flat[:65536]
            _, lens = run_length_encode_u8(sample)
            if float(np.mean(lens)) < 1.5:
                continue
        candidates.append(encoders[m](flat))

    if not candidates:
        candidates.append(encode_raw8(flat))

    best = min(candidates, key=lambda r: (r.complete_bytes, r.mode))
    return best


def decode_exp_blob(blob: bytes, n: int) -> tuple[np.ndarray, int]:
    """Decode a complete exponent section. Returns (exp_u8, mode)."""
    if n < 0:
        raise ValueError(n)
    if len(blob) < 1:
        raise ValueError("empty exp blob")
    mode = int(blob[0])
    if mode == EXP_RAW8:
        raw = blob[1:]
        return unpack_exp_u8(raw, n).astype(np.uint8, copy=False), mode

    if mode in (EXP_RANS, EXP_DELTA_RANS):
        stream, _end = _unpack_rans_payload(blob, 1, n)
        if mode == EXP_DELTA_RANS:
            stream = undelta_u8(stream)
        return stream, mode

    if mode == EXP_HUFFMAN:
        stream, _end = _unpack_huffman_payload(blob, 1, n)
        return stream, mode

    if mode == EXP_RUN_RANS:
        (n_runs,) = struct.unpack_from("<I", blob, 1)
        offset = 5
        values, offset = _unpack_rans_payload(blob, offset, n_runs)
        lengths, offset = _unpack_rans_payload(blob, offset, n_runs)
        return run_length_decode_u8(values, lengths, n), mode

    raise ValueError(f"unknown exp mode {mode}")


def complete_cost_breakdown(result: ExpEncodeResult) -> dict[str, Any]:
    """Honesty helper: account every byte charged to the exp section."""
    accounted = (
        result.mode_id_bytes
        + result.length_field_bytes
        + result.table_bytes
        + result.stream_bytes
    )
    # RAW8 has no length fields; payload is stream only.
    # RUN_RANS / others: accounted should equal complete_bytes.
    # For RAW8: mode + stream == complete.
    return {
        **result.as_dict(),
        "accounted_bytes": accounted,
        "matches_blob": accounted == result.complete_bytes
        or (
            result.mode == EXP_RAW8
            and result.mode_id_bytes + result.stream_bytes == result.complete_bytes
        ),
    }
