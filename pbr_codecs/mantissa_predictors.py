"""Lightweight mantissa predictors: CTW/PPM, histogram GBDT, tiny AR, IDF-lite.

Numpy only (no sklearn/torch). Models dump to compact bytes; complete BPW
must count every dumped byte. Predictors are causal in raster order so a
decoder can reconstruct the same probabilities.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_core.binary_rans import binary_rans_decode, binary_rans_encode, p_to_freq1
from pbr_core.rans import dump_freq_table, load_freq_table, rans_decode, rans_encode, table_from_symbols

MANT_ALPH = 128
_F16 = np.dtype("<f2")


def bit_nll(p1: np.ndarray, bits: np.ndarray) -> float:
    p = np.clip(np.asarray(p1, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    b = np.asarray(bits, dtype=np.float64)
    return float(-(b * np.log2(p) + (1.0 - b) * np.log2(1.0 - p)).sum())


def cat_nll(logp: np.ndarray, y: np.ndarray) -> float:
    """``logp`` is log-prob (natural) of shape (n, alph); return bits."""
    idx = np.arange(int(y.size))
    lp = logp[idx, np.clip(y.astype(np.int64), 0, logp.shape[1] - 1)]
    return float(-lp.sum() / np.log(2.0))


def softmax_log(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return z - np.log(ez.sum(axis=1, keepdims=True))


def plane_bits(mant: np.ndarray, bit: int) -> np.ndarray:
    return ((np.ascontiguousarray(mant, dtype=np.uint8).ravel() >> np.uint8(bit)) & np.uint8(1)).astype(np.uint8)


def markov_context(bits: np.ndarray, depth: int) -> np.ndarray:
    """Raster previous-``depth`` bits as an integer context; ctx[0]=0."""
    b = np.ascontiguousarray(bits, dtype=np.uint8).ravel()
    n = int(b.size)
    ctx = np.zeros(n, dtype=np.int32)
    if depth <= 0 or n == 0:
        return ctx
    acc = np.zeros(n, dtype=np.int32)
    for k in range(depth):
        shifted = np.zeros(n, dtype=np.int32)
        take = n - 1 - k
        if take > 0:
            shifted[k + 1 :] = b[:take].astype(np.int32)
        acc |= shifted << k
    return acc


def fit_bit_markov(mant: np.ndarray, depth: int) -> np.ndarray:
    """Return counts shape (7, 2**depth, 2)."""
    n_ctx = 1 << max(int(depth), 0)
    counts = np.zeros((7, n_ctx, 2), dtype=np.int64)
    for bit in range(7):
        b = plane_bits(mant, bit)
        ctx = markov_context(b, depth) & (n_ctx - 1)
        idx = ctx.astype(np.int64) * 2 + b.astype(np.int64)
        counts[bit] = np.bincount(idx, minlength=n_ctx * 2).reshape(n_ctx, 2)
    return counts


def dump_bit_markov(counts: np.ndarray) -> bytes:
    c = np.clip(counts, 0, 65535).astype("<u2")
    header = struct.pack("<BBI", 7, int(np.log2(max(c.shape[1], 1))), int(c.size))
    return header + c.tobytes()


def load_bit_markov(blob: bytes) -> np.ndarray:
    n_planes, log_d, n = struct.unpack_from("<BBI", blob, 0)
    depth_ctx = 1 << log_d
    arr = np.frombuffer(blob, dtype="<u2", count=n, offset=7)
    return arr.reshape(n_planes, depth_ctx, 2).astype(np.int64)


def markov_p1(counts_bit: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    c = counts_bit[ctx]
    tot = c.sum(axis=1).astype(np.float64) + 1.0
    return (c[:, 1].astype(np.float64) + 0.5) / tot


def markov_nll(counts: np.ndarray, mant: np.ndarray, depth: int) -> float:
    nll = 0.0
    n_ctx = counts.shape[1]
    for bit in range(7):
        b = plane_bits(mant, bit)
        ctx = markov_context(b, depth) & (n_ctx - 1)
        nll += bit_nll(markov_p1(counts[bit], ctx), b)
    return nll


def ctw_mix_p1(order_counts: list[tuple[int, np.ndarray]], bits: np.ndarray) -> np.ndarray:
    """Equal-weight mixture of KT predictors at several Markov depths."""
    mix = None
    w = 1.0 / max(len(order_counts), 1)
    for depth, counts in order_counts:
        n_ctx = counts.shape[0]
        ctx = markov_context(bits, depth) & (n_ctx - 1)
        p = markov_p1(counts, ctx)
        mix = p * w if mix is None else mix + p * w
    return mix


def ctw_nll(order_models: list[tuple[int, np.ndarray]], mant: np.ndarray) -> float:
    nll = 0.0
    for bit in range(7):
        b = plane_bits(mant, bit)
        pairs = [(d, counts[bit]) for d, counts in order_models]
        nll += bit_nll(ctw_mix_p1(pairs, b), b)
    return nll


def fit_ppm_hash(mant: np.ndarray, buckets: int = 1024) -> np.ndarray:
    """Hashed order-2 128-ary counts. Shape (buckets, 128)."""
    y = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    n = int(y.size)
    prev = np.zeros(n, dtype=np.int64)
    prev2 = np.zeros(n, dtype=np.int64)
    if n:
        prev[1:] = y[:-1]
        prev2[2:] = y[:-2]
    key = (prev * 131 + prev2) & (buckets - 1)
    idx = key * 128 + y.astype(np.int64)
    return np.bincount(idx, minlength=buckets * 128).reshape(buckets, 128)


def dump_ppm_hash(counts: np.ndarray) -> bytes:
    c = np.clip(counts, 0, 65535).astype("<u2")
    return struct.pack("<HH", c.shape[0], c.shape[1]) + c.tobytes()


def ppm_hash_nll(counts: np.ndarray, mant: np.ndarray) -> float:
    y = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    n = int(y.size)
    if n == 0:
        return 0.0
    prev = np.zeros(n, dtype=np.int64)
    prev2 = np.zeros(n, dtype=np.int64)
    prev[1:] = y[:-1]
    prev2[2:] = y[:-2]
    buckets = counts.shape[0]
    key = (prev * 131 + prev2) & (buckets - 1)
    row = counts[key].astype(np.float64)
    tot = row.sum(axis=1) + 128.0
    p = (row[np.arange(n), y.astype(np.int64)] + 1.0) / tot
    return float(-np.log2(np.clip(p, 1e-12, 1.0)).sum())


def _prev_value(m: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(m, dtype=np.uint8).ravel()
    prev = np.zeros_like(flat)
    if flat.size:
        prev[1:] = flat[:-1]
    return prev.reshape(np.asarray(m).shape)


def _prev_row(m: np.ndarray) -> np.ndarray:
    tile = np.ascontiguousarray(m, dtype=np.uint8)
    if tile.ndim == 1:
        tile = tile.reshape(1, -1)
    pred = np.zeros_like(tile)
    if tile.shape[0] > 1:
        pred[1:, :] = tile[:-1, :]
    return pred


def feature_bins(sign: np.ndarray, exp: np.ndarray, mant: np.ndarray) -> np.ndarray:
    """Causal integer features: exp>>3, sign, prev>>2, prev_row>>2. Shape (n, 4)."""
    s = np.ascontiguousarray(sign, dtype=np.uint8).ravel()
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    m = np.ascontiguousarray(mant, dtype=np.uint8)
    prev = _prev_value(m).ravel()
    prow = _prev_row(m).ravel()
    return np.stack(
        [e >> 3, s, prev >> 2, prow >> 2],
        axis=1,
    ).astype(np.int32)


def _fit_stump(X: np.ndarray, g: np.ndarray, h: np.ndarray, n_bins: int) -> tuple[int, int, float]:
    best = (-1.0, 0, 1, 0.0)
    n = X.shape[1]
    for f in range(n):
        gs = np.bincount(X[:, f], weights=g, minlength=n_bins)
        hs = np.bincount(X[:, f], weights=h, minlength=n_bins) + 1e-6
        g_l = np.cumsum(gs)
        h_l = np.cumsum(hs)
        g_r = g_l[-1] - g_l
        h_r = h_l[-1] - h_l
        gain = (g_l * g_l) / h_l + (g_r * g_r) / np.maximum(h_r, 1e-6)
        gain[-1] = -np.inf
        k = int(np.argmax(gain))
        val = float(gain[k])
        if val > best[0]:
            best = (val, f, k + 1, val)
    return best[1], best[2], best[0]


def _newton_leaf(g: np.ndarray, h: np.ndarray) -> float:
    return float(-g.sum() / (h.sum() + 1e-6))


def _fit_depth2(X: np.ndarray, g: np.ndarray, h: np.ndarray, n_bins: int) -> dict:
    f0, t0, _ = _fit_stump(X, g, h, n_bins)
    left = X[:, f0] < t0
    f1, t1, _ = _fit_stump(X[left], g[left], h[left], n_bins) if left.any() else (0, 1, 0.0)
    f2, t2, _ = _fit_stump(X[~left], g[~left], h[~left], n_bins) if (~left).any() else (0, 1, 0.0)
    lmask = left & (X[:, f1] < t1)
    rmask = left & ~lmask
    lmask2 = (~left) & (X[:, f2] < t2)
    rmask2 = (~left) & ~lmask2
    leaves = (
        _newton_leaf(g[lmask], h[lmask]) if lmask.any() else 0.0,
        _newton_leaf(g[rmask], h[rmask]) if rmask.any() else 0.0,
        _newton_leaf(g[lmask2], h[lmask2]) if lmask2.any() else 0.0,
        _newton_leaf(g[rmask2], h[rmask2]) if rmask2.any() else 0.0,
    )
    return {"f0": f0, "t0": t0, "f1": f1, "t1": t1, "f2": f2, "t2": t2, "leaves": leaves}


def _eval_depth2(tree: dict, X: np.ndarray) -> np.ndarray:
    left = X[:, tree["f0"]] < tree["t0"]
    l2 = np.where(left, X[:, tree["f1"]] < tree["t1"], X[:, tree["f2"]] < tree["t2"])
    out = np.empty(X.shape[0], dtype=np.float64)
    lv = tree["leaves"]
    out[left & l2] = lv[0]
    out[left & ~l2] = lv[1]
    out[~left & l2] = lv[2]
    out[~left & ~l2] = lv[3]
    return out


def fit_gbdt_bits(
    X: np.ndarray,
    mant: np.ndarray,
    *,
    n_trees: int = 16,
    n_bins: int = 32,
    lr: float = 0.25,
) -> dict:
    y_all = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    models = []
    for bit in range(7):
        y = ((y_all >> bit) & 1).astype(np.float64)
        p0 = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
        F = np.full(y.size, np.log(p0 / (1.0 - p0)))
        trees = []
        for _ in range(n_trees):
            p = 1.0 / (1.0 + np.exp(-F))
            g = p - y
            h = np.clip(p * (1.0 - p), 1e-6, None)
            tree = _fit_depth2(X, g, h, n_bins)
            F = F + lr * _eval_depth2(tree, X)
            trees.append(tree)
        models.append({"p0": p0, "trees": trees})
    return {"n_trees": n_trees, "n_bins": n_bins, "lr": lr, "bits": models}


def gbdt_p1(model: dict, X: np.ndarray, bit: int) -> np.ndarray:
    rec = model["bits"][bit]
    F = np.full(X.shape[0], np.log(rec["p0"] / (1.0 - rec["p0"])))
    lr = float(model["lr"])
    for tree in rec["trees"]:
        F = F + lr * _eval_depth2(tree, X)
    return 1.0 / (1.0 + np.exp(-F))


def gbdt_nll(model: dict, X: np.ndarray, mant: np.ndarray) -> float:
    y = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    nll = 0.0
    for bit in range(7):
        nll += bit_nll(gbdt_p1(model, X, bit), (y >> bit) & 1)
    return nll


def dump_gbdt(model: dict) -> bytes:
    blob = struct.pack("<BBf", model["n_trees"], model["n_bins"], float(model["lr"]))
    for rec in model["bits"]:
        blob += struct.pack("<e", rec["p0"])
        for tree in rec["trees"]:
            blob += struct.pack(
                "<BBBBBB4e",
                tree["f0"] & 255,
                tree["t0"] & 255,
                tree["f1"] & 255,
                tree["t1"] & 255,
                tree["f2"] & 255,
                tree["t2"] & 255,
                *tree["leaves"],
            )
    return blob


def ar_features(sign: np.ndarray, exp: np.ndarray, mant: np.ndarray) -> np.ndarray:
    s = np.ascontiguousarray(sign, dtype=np.float32).ravel()
    e = np.ascontiguousarray(exp, dtype=np.float32).ravel()
    m = np.ascontiguousarray(mant, dtype=np.uint8)
    prev = _prev_value(m).ravel().astype(np.float32)
    prow = _prev_row(m).ravel().astype(np.float32)
    return np.stack([e / 255.0, s, prev / 127.0, prow / 127.0], axis=1)


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 32,
    epochs: int = 6,
    batch: int = 4096,
    lr: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = MANT_ALPH
    w1 = rng.normal(0, 0.1, size=(d, hidden)).astype(np.float64)
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.05, size=(hidden, k)).astype(np.float64)
    b2 = np.zeros(k)
    order = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, n, batch):
            sl = order[start : start + batch]
            xb = X[sl]
            yb = y[sl]
            h = np.maximum(xb @ w1 + b1, 0.0)
            logits = h @ w2 + b2
            logp = softmax_log(logits)
            p = np.exp(logp)
            one = np.zeros_like(p)
            one[np.arange(yb.size), yb] = 1.0
            dlog = (p - one) / max(yb.size, 1)
            dw2 = h.T @ dlog
            db2 = dlog.sum(axis=0)
            dh = (dlog @ w2.T) * (h > 0)
            dw1 = xb.T @ dh
            db1 = dh.sum(axis=0)
            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "hidden": hidden}


def mlp_logp(model: dict, X: np.ndarray) -> np.ndarray:
    h = np.maximum(X @ model["w1"] + model["b1"], 0.0)
    return softmax_log(h @ model["w2"] + model["b2"])


def mlp_nll(model: dict, X: np.ndarray, y: np.ndarray) -> float:
    return cat_nll(mlp_logp(model, X), np.ascontiguousarray(y, dtype=np.int64).ravel())


def dump_mlp(model: dict) -> bytes:
    w1 = np.ascontiguousarray(model["w1"], dtype=_F16)
    b1 = np.ascontiguousarray(model["b1"], dtype=_F16)
    w2 = np.ascontiguousarray(model["w2"], dtype=_F16)
    b2 = np.ascontiguousarray(model["b2"], dtype=_F16)
    header = struct.pack("<HHHH", w1.shape[0], w1.shape[1], w2.shape[1], 16)
    return header + w1.tobytes() + b1.tobytes() + w2.tobytes() + b2.tobytes()


def load_mlp(blob: bytes) -> dict:
    d, h, k, _bits = struct.unpack_from("<HHHH", blob, 0)
    off = 8
    def take(n):
        nonlocal off
        arr = np.frombuffer(blob, dtype=_F16, count=n, offset=off).astype(np.float64)
        off += n * 2
        return arr
    w1 = take(d * h).reshape(d, h)
    b1 = take(h)
    w2 = take(h * k).reshape(h, k)
    b2 = take(k)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "hidden": h}


def fit_idf_tables(mant: np.ndarray, n_layers: int = 3) -> list[np.ndarray]:
    """Per-layer additive coupling tables. Index = partner mantissa, value in 0..127."""
    x = np.ascontiguousarray(mant, dtype=np.uint8)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    tables: list[np.ndarray] = []
    work = x.copy()
    rows, cols = work.shape
    even_n = cols - (cols % 2)
    for layer in range(n_layers):
        t = np.zeros(128, dtype=np.int16)
        if even_n < 2:
            tables.append(t)
            continue
        a = work[:, 0:even_n:2]
        b = work[:, 1:even_n:2]
        src, dst = (a, b) if layer % 2 == 0 else (b, a)
        joint = np.bincount(
            src.ravel().astype(np.int64) * 128 + dst.ravel().astype(np.int64),
            minlength=128 * 128,
        ).reshape(128, 128)
        for v in range(128):
            if joint[v].sum() > 0:
                t[v] = int((-int(joint[v].argmax())) & 127)
        if layer % 2 == 0:
            b[:] = (b.astype(np.int16) + t[a]) & 127
        else:
            a[:] = (a.astype(np.int16) + t[b]) & 127
        work[:, 0:even_n:2] = a
        work[:, 1:even_n:2] = b
        tables.append(t)
    return tables


def idf_apply(mant: np.ndarray, tables: list[np.ndarray], *, inverse: bool = False) -> np.ndarray:
    x = np.ascontiguousarray(mant, dtype=np.uint8).copy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    rows, cols = x.shape
    even_n = cols - (cols % 2)
    seq = list(enumerate(tables))
    if inverse:
        seq = list(reversed(seq))
    for layer, t in seq:
        if even_n < 2:
            continue
        a = x[:, 0:even_n:2]
        b = x[:, 1:even_n:2]
        add = t.astype(np.int16)
        if inverse:
            add = (-add) & 127
        if layer % 2 == 0:
            b[:] = (b.astype(np.int16) + add[a]) & 127
        else:
            a[:] = (a.astype(np.int16) + add[b]) & 127
        x[:, 0:even_n:2] = a
        x[:, 1:even_n:2] = b
    return x.ravel() if squeeze else x


def dump_idf(tables: list[np.ndarray]) -> bytes:
    blob = struct.pack("<H", len(tables))
    for t in tables:
        u = np.zeros(128, dtype=np.uint8)
        src = np.ascontiguousarray(t, dtype=np.int16).ravel() & 127
        u[: min(128, src.size)] = src[:128].astype(np.uint8)
        blob += u.tobytes()
    return blob


def uncond_counts(mant: np.ndarray) -> np.ndarray:
    y = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    return np.bincount(y.astype(np.int64), minlength=MANT_ALPH)


def exp_cond_counts(exp: np.ndarray, mant: np.ndarray) -> np.ndarray:
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel().astype(np.int64)
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel().astype(np.int64)
    return np.bincount(e * MANT_ALPH + m, minlength=256 * MANT_ALPH).reshape(256, MANT_ALPH)


def laplace_nll_from_counts(counts_1d: np.ndarray, y: np.ndarray) -> float:
    c = counts_1d.astype(np.float64)
    tot = c.sum() + MANT_ALPH
    p = (c[np.clip(y.astype(np.int64), 0, MANT_ALPH - 1)] + 1.0) / tot
    return float(-np.log2(np.clip(p, 1e-12, 1.0)).sum())


def exp_cond_nll(counts: np.ndarray, exp: np.ndarray, mant: np.ndarray) -> float:
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel().astype(np.int64)
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel().astype(np.int64)
    row = counts[e].astype(np.float64)
    tot = row.sum(axis=1) + MANT_ALPH
    p = (row[np.arange(m.size), m] + 1.0) / tot
    return float(-np.log2(np.clip(p, 1e-12, 1.0)).sum())


def dump_uncond_table(counts: np.ndarray) -> bytes:
    from pbr_core.rans import normalize_counts

    freq = np.zeros(256, dtype=np.int64)
    n = min(MANT_ALPH, int(counts.size))
    freq[:n] = np.asarray(counts[:n], dtype=np.int64)
    return dump_freq_table(normalize_counts(freq))


def encode_mant_uncond(mant: np.ndarray) -> bytes:
    y = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    freq = table_from_symbols(y)
    table = dump_freq_table(freq)
    stream = rans_encode(y, freq)
    return struct.pack("<HI", len(table), len(stream)) + table + stream


def decode_mant_uncond(blob: bytes, n: int) -> np.ndarray:
    table_len, stream_len = struct.unpack_from("<HI", blob, 0)
    freq, end = load_freq_table(blob, 6)
    stream = blob[end : end + stream_len]
    return rans_decode(stream, n, freq)


def encode_bits_with_p1(bits: np.ndarray, p1: np.ndarray) -> bytes:
    return binary_rans_encode(bits, p_to_freq1(p1))


def decode_bits_with_p1(blob: bytes, n: int, p1: np.ndarray) -> np.ndarray:
    return binary_rans_decode(blob, n, p_to_freq1(p1))


def encode_mant_markov(mant: np.ndarray, counts: np.ndarray, depth: int) -> bytes:
    parts: list[bytes] = []
    n_ctx = counts.shape[1]
    for bit in range(7):
        b = plane_bits(mant, bit)
        ctx = markov_context(b, depth) & (n_ctx - 1)
        blob = encode_bits_with_p1(b, markov_p1(counts[bit], ctx))
        parts.append(struct.pack("<I", len(blob)) + blob)
    return b"".join(parts)


def decode_mant_markov(blob: bytes, n: int, counts: np.ndarray, depth: int, shape: tuple[int, ...]) -> np.ndarray:
    out_bits = np.zeros(n, dtype=np.uint8)
    pos = 0
    n_ctx = counts.shape[1]
    for bit in range(7):
        (ln,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        part = blob[pos : pos + ln]
        pos += ln
        # Causal: decode sequentially so context uses already-decoded bits.
        decoded = np.zeros(n, dtype=np.uint8)
        # Fast path: context of a plane depends only on that plane, so we can
        # decode the whole plane with encoder-matching ctx after a prefix walk.
        # Rebuild ctx from decoded prefix in one binary_rans pass by using the
        # same ctx construction as encode — requires the bits; so walk.
        x_blob = part
        # Decode one-by-one would be slow; planes are self-contained so recover
        # by running binary_rans with freq from *encoded* ctx. Encoder ctx used
        # true bits, which equal decoded bits if we decode in order. Walk:
        from pbr_core.binary_rans import M, RANS_L, SCALE_BITS

        state = struct.unpack_from("<I", x_blob, 0)[0]
        bp = 4
        nblob = len(x_blob)
        ctx = 0
        for i in range(n):
            c = counts[bit][ctx]
            tot = float(c.sum() + 1.0)
            p1 = (float(c[1]) + 0.5) / tot
            f1 = int(np.clip(round(p1 * M), 1, M - 1))
            f0 = M - f1
            cf = state & (M - 1)
            bitv = 0 if cf < f0 else 1
            f = f1 if bitv else f0
            start = 0 if bitv == 0 else f0
            state = f * (state >> SCALE_BITS) + cf - start
            while state < RANS_L and bp < nblob:
                state = (state << 8) | x_blob[bp]
                bp += 1
            decoded[i] = bitv
            ctx = ((ctx << 1) | bitv) & (n_ctx - 1)
        out_bits |= decoded << np.uint8(bit)
    return out_bits.reshape(shape)


def sequential_bit_p1_from_pmf(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LSB-first bit probabilities implied by a 128-way pmf, teacher-forced."""
    live = np.asarray(pmf, dtype=np.float64).copy()
    y = np.ascontiguousarray(y, dtype=np.int64).ravel()
    n = int(y.size)
    out = np.empty((n, 7), dtype=np.float64)
    idx = np.arange(128)
    for bit in range(7):
        denom = live.sum(axis=1).clip(1e-15)
        mask1 = ((idx >> bit) & 1) == 1
        out[:, bit] = live[:, mask1].sum(axis=1) / denom
        bitv = (y >> bit) & 1
        keep = ((idx[None, :] >> bit) & 1) == bitv[:, None]
        live *= keep
    return out


def encode_mant_gbdt(mant: np.ndarray, sign: np.ndarray, exp: np.ndarray, model: dict) -> bytes:
    X = feature_bins(sign, exp, mant)
    y = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    n = int(y.size)
    bits = np.empty(n * 7, dtype=np.uint8)
    p1 = np.empty(n * 7, dtype=np.float64)
    for bit in range(7):
        bits[bit::7] = (y >> bit) & 1
        p1[bit::7] = gbdt_p1(model, X, bit)
    return encode_bits_with_p1(bits, p1)


def decode_mant_gbdt(
    blob: bytes, sign: np.ndarray, exp: np.ndarray, model: dict, shape: tuple[int, ...]
) -> np.ndarray:
    """Causal decode: rebuild features from already-decoded mantissas."""
    n = int(np.prod(shape))
    s = np.ascontiguousarray(sign, dtype=np.uint8).ravel()
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    cols = int(shape[1]) if len(shape) > 1 else n
    out = np.zeros(n, dtype=np.uint8)
    # Decode the interleaved bitstream one bit at a time.
    from pbr_core.binary_rans import M, RANS_L, SCALE_BITS

    state = struct.unpack_from("<I", blob, 0)[0]
    bp = 4
    nblob = len(blob)
    prev = 0
    for i in range(n):
        prow = int(out[i - cols]) if i >= cols else 0
        X = np.array([[e[i] >> 3, s[i], prev >> 2, prow >> 2]], dtype=np.int32)
        word = 0
        for bit in range(7):
            p1 = float(gbdt_p1(model, X, bit)[0])
            f1 = int(np.clip(round(p1 * M), 1, M - 1))
            f0 = M - f1
            cf = state & (M - 1)
            bitv = 0 if cf < f0 else 1
            f = f1 if bitv else f0
            start = 0 if bitv == 0 else f0
            state = f * (state >> SCALE_BITS) + cf - start
            while state < RANS_L and bp < nblob:
                state = (state << 8) | blob[bp]
                bp += 1
            word |= bitv << bit
        out[i] = word
        prev = word
    return out.reshape(shape)


def encode_mant_ar(mant: np.ndarray, sign: np.ndarray, exp: np.ndarray, model: dict) -> bytes:
    X = ar_features(sign, exp, mant)
    y = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    n = int(y.size)
    pmf = np.exp(mlp_logp(model, X))
    p1s = sequential_bit_p1_from_pmf(pmf, y)
    bits = np.empty(n * 7, dtype=np.uint8)
    p1 = np.empty(n * 7, dtype=np.float64)
    for bit in range(7):
        bits[bit::7] = (y >> bit) & 1
        p1[bit::7] = p1s[:, bit]
    return encode_bits_with_p1(bits, p1)


def decode_mant_ar(
    blob: bytes, sign: np.ndarray, exp: np.ndarray, model: dict, shape: tuple[int, ...]
) -> np.ndarray:
    n = int(np.prod(shape))
    s = np.ascontiguousarray(sign, dtype=np.uint8).ravel()
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    cols = int(shape[1]) if len(shape) > 1 else n
    out = np.zeros(n, dtype=np.uint8)
    from pbr_core.binary_rans import M, RANS_L, SCALE_BITS

    state = struct.unpack_from("<I", blob, 0)[0]
    bp = 4
    nblob = len(blob)
    prev = 0
    idx = np.arange(128)
    for i in range(n):
        prow = int(out[i - cols]) if i >= cols else 0
        x = np.array([[e[i] / 255.0, s[i], prev / 127.0, prow / 127.0]], dtype=np.float64)
        live = np.exp(mlp_logp(model, x))[0]
        word = 0
        for bit in range(7):
            denom = float(live.sum()) + 1e-15
            p1 = float(live[(((idx >> bit) & 1) == 1)].sum() / denom)
            f1 = int(np.clip(round(p1 * M), 1, M - 1))
            f0 = M - f1
            cf = state & (M - 1)
            bitv = 0 if cf < f0 else 1
            f = f1 if bitv else f0
            start = 0 if bitv == 0 else f0
            state = f * (state >> SCALE_BITS) + cf - start
            while state < RANS_L and bp < nblob:
                state = (state << 8) | blob[bp]
                bp += 1
            word |= bitv << bit
            live = live * ((((idx >> bit) & 1) == bitv).astype(np.float64))
        out[i] = word
        prev = word
    return out.reshape(shape)


def encode_mant_idf(mant: np.ndarray, tables: list[np.ndarray]) -> bytes:
    y = idf_apply(np.ascontiguousarray(mant, dtype=np.uint8), tables, inverse=False)
    return encode_mant_uncond(y)


def decode_mant_idf(blob: bytes, n: int, tables: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    y = decode_mant_uncond(blob, n).reshape(shape)
    return idf_apply(y, tables, inverse=True)
