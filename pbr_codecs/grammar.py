"""Residual-sequence (n-gram) dictionary. Admit only if complete bytes save."""

from __future__ import annotations

import struct
from collections import Counter

import numpy as np

from pbr_codecs.xor_predictor import residuals_prev_value
from pbr_core.bitio import bits_needed, pack_ids, unpack_ids
from pbr_core.hashing import words_to_bytes
from pbr_core.types import (
    MODE_GRAMMAR,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)

MAX_ALPHABET = 8
PHRASE_LEN = 4
MIN_PHRASE_HITS = 3
MAX_PHRASES = 16


def _residual_ids(words: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    residuals = residuals_prev_value(words).ravel()
    unique, counts = np.unique(residuals, return_counts=True)
    if unique.size < 2 or unique.size > MAX_ALPHABET:
        return None
    order = np.argsort(-counts)
    palette = unique[order]
    index = {int(v): i for i, v in enumerate(palette)}
    ids = np.array([index[int(v)] for v in residuals], dtype=np.uint16)
    return palette, ids


def _find_phrases(ids: np.ndarray) -> list[tuple[int, ...]]:
    n = int(ids.size)
    if n < PHRASE_LEN * MIN_PHRASE_HITS:
        return []
    grams: Counter[tuple[int, ...]] = Counter()
    seq = [int(x) for x in ids.tolist()]
    for i in range(0, n - PHRASE_LEN + 1):
        grams[tuple(seq[i : i + PHRASE_LEN])] += 1
    ranked = [g for g, c in grams.most_common() if c >= MIN_PHRASE_HITS]
    return ranked[:MAX_PHRASES]


def _replace_phrases(ids: np.ndarray, phrases: list[tuple[int, ...]], k: int) -> np.ndarray:
    seq = [int(x) for x in ids.tolist()]
    out: list[int] = []
    i = 0
    lookup = {ph: k + j for j, ph in enumerate(phrases)}
    while i < len(seq):
        hit = None
        if i + PHRASE_LEN <= len(seq):
            gram = tuple(seq[i : i + PHRASE_LEN])
            if gram in lookup:
                hit = lookup[gram]
        if hit is not None:
            out.append(hit)
            i += PHRASE_LEN
        else:
            out.append(seq[i])
            i += 1
    return np.asarray(out, dtype=np.uint16)


class ResidualGrammarCodec:
    name = "residual_grammar"
    mode_id = MODE_GRAMMAR

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        mapped = _residual_ids(words)
        if mapped is None:
            return None
        palette, ids = mapped
        phrases = _find_phrases(ids)
        if not phrases:
            return None
        replaced = _replace_phrases(ids, phrases, int(palette.size))
        alphabet = int(palette.size) + len(phrases)
        nbits = bits_needed(alphabet)
        phrase_blob = b"".join(bytes(ph) for ph in phrases)
        payload = (
            struct.pack("<BBBH", int(palette.size), len(phrases), nbits, int(replaced.size))
            + words_to_bytes(palette)
            + phrase_blob
            + pack_ids(replaced, nbits)
        )
        n = int(np.asarray(words).size)
        if TILE_HEADER_BYTES + len(payload) >= TILE_HEADER_BYTES + n * 2:
            return None
        return EncodedBlock(mode_id=self.mode_id, mode_name=self.name, payload=payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        k, n_ph, nbits, n_tok = struct.unpack_from("<BBBH", encoded.payload, 0)
        offset = 5
        palette = np.frombuffer(encoded.payload, dtype="<u2", count=k, offset=offset).astype(np.uint16)
        offset += 2 * k
        phrases = []
        for _ in range(n_ph):
            phrases.append(tuple(encoded.payload[offset : offset + PHRASE_LEN]))
            offset += PHRASE_LEN
        tokens = unpack_ids(encoded.payload[offset:], n_tok, nbits)
        ids: list[int] = []
        for tok in tokens.tolist():
            t = int(tok)
            if t < k:
                ids.append(t)
            else:
                ids.extend(phrases[t - k])
        residual_ids = np.asarray(ids, dtype=np.uint16)
        residuals = palette[residual_ids]
        from pbr_codecs.xor_predictor import reconstruct_prev_value

        words = reconstruct_prev_value(residuals.reshape(encoded.rows, encoded.cols))
        return words
