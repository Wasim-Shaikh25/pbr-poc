/* Bit-exact match of pbr_core.rans.rans_decode (byte-renormalized 32-bit rANS).
 *
 * Single-stream rANS has a serial state dependency, so SIMD does not change
 * the bitstream. Speed comes from fused tile decode (rANS + BF16 join), uint32
 * tables, and parallel decode across tensors in Python threads (GIL released).
 */
#include <stdint.h>
#include <string.h>

#define RANS_L (1u << 23)
#define SCALE_BITS 12
#define M (1u << SCALE_BITS)

static uint32_t load_u16(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

static uint32_t load_u32(const uint8_t *p) {
    return (uint32_t)p[0]
        | ((uint32_t)p[1] << 8)
        | ((uint32_t)p[2] << 16)
        | ((uint32_t)p[3] << 24);
}

static void build_lut_cumul(
    const uint32_t *freq,
    uint32_t *cumul,
    uint8_t *lut
) {
    uint32_t pos = 0;
    int s;
    for (s = 0; s < 256; s++) {
        uint32_t f = freq[s];
        uint32_t j;
        cumul[s] = pos;
        for (j = 0; j < f; j++) {
            lut[pos + j] = (uint8_t)s;
        }
        pos += f;
    }
    cumul[256] = pos;
    while (pos < M) {
        lut[pos] = lut[pos - 1];
        pos++;
    }
}

int pbr_rans_decode(
    const uint8_t *blob,
    int nblob,
    int count,
    const int64_t *freq,  /* 256 */
    const int64_t *cumul, /* 257 */
    const uint8_t *lut,   /* 4096 */
    uint8_t *out
) {
    const uint32_t mask = (uint32_t)(M - 1);
    uint64_t x;
    int pos;
    int i;

    if (count < 0 || nblob < 0) {
        return -1;
    }
    if (count == 0) {
        return 0;
    }
    if (nblob < 4 || blob == NULL || freq == NULL || cumul == NULL || lut == NULL || out == NULL) {
        return -2;
    }
    x = (uint64_t)load_u32(blob);
    pos = 4;
    for (i = 0; i < count; i++) {
        uint32_t cf = (uint32_t)(x & mask);
        uint8_t s = lut[cf];
        uint32_t f = (uint32_t)freq[s];
        uint32_t start = (uint32_t)cumul[s];
        x = (uint64_t)f * (x >> SCALE_BITS) + (uint64_t)cf - (uint64_t)start;
        out[i] = s;
        while (x < (uint64_t)RANS_L && pos < nblob) {
            x = (x << 8) | (uint64_t)blob[pos];
            pos += 1;
        }
    }
    return 0;
}

/* Same rANS, uint32 tables (hot path). */
int pbr_rans_decode_u32(
    const uint8_t *blob,
    int nblob,
    int count,
    const uint32_t *freq,
    const uint32_t *cumul,
    const uint8_t *lut,
    uint8_t *out
) {
    const uint32_t mask = (uint32_t)(M - 1);
    uint64_t x;
    int pos;
    int i;

    if (count == 0) {
        return 0;
    }
    if (count < 0 || nblob < 4 || blob == NULL || freq == NULL || cumul == NULL || lut == NULL || out == NULL) {
        return -1;
    }
    x = (uint64_t)load_u32(blob);
    pos = 4;
    for (i = 0; i < count; i++) {
        uint32_t cf = (uint32_t)(x & mask);
        uint8_t s = lut[cf];
        uint32_t f = freq[s];
        uint32_t start = cumul[s];
        x = (uint64_t)f * (x >> SCALE_BITS) + (uint64_t)cf - (uint64_t)start;
        out[i] = s;
        while (x < (uint64_t)RANS_L && pos < nblob) {
            x = (x << 8) | (uint64_t)blob[pos];
            pos += 1;
        }
    }
    return 0;
}

/* BF16: exp byte + packed sign/mantissa byte -> uint16 word. */
int pbr_join_bf16(const uint8_t *exp, const uint8_t *packed_sm, int n, uint16_t *out) {
    int i;
    if (n < 0 || (n > 0 && (exp == NULL || packed_sm == NULL || out == NULL))) {
        return -1;
    }
    i = 0;
    for (; i + 4 <= n; i += 4) {
        uint8_t sm0 = packed_sm[i];
        uint8_t sm1 = packed_sm[i + 1];
        uint8_t sm2 = packed_sm[i + 2];
        uint8_t sm3 = packed_sm[i + 3];
        out[i] = (uint16_t)(((uint16_t)(sm0 >> 7) << 15) | ((uint16_t)exp[i] << 7) | (uint16_t)(sm0 & 0x7Fu));
        out[i + 1] = (uint16_t)(((uint16_t)(sm1 >> 7) << 15) | ((uint16_t)exp[i + 1] << 7) | (uint16_t)(sm1 & 0x7Fu));
        out[i + 2] = (uint16_t)(((uint16_t)(sm2 >> 7) << 15) | ((uint16_t)exp[i + 2] << 7) | (uint16_t)(sm2 & 0x7Fu));
        out[i + 3] = (uint16_t)(((uint16_t)(sm3 >> 7) << 15) | ((uint16_t)exp[i + 3] << 7) | (uint16_t)(sm3 & 0x7Fu));
    }
    for (; i < n; i++) {
        uint8_t sm = packed_sm[i];
        out[i] = (uint16_t)(((uint16_t)(sm >> 7) << 15) | ((uint16_t)exp[i] << 7) | (uint16_t)(sm & 0x7Fu));
    }
    return 0;
}

/* Canonical Huffman, LSB-first peek, lut of (sym, nbits) packed as uint16: sym | (nbits<<8). */
int pbr_huffman_decode(
    const uint8_t *data,
    int ndata,
    int count,
    int max_len,
    const uint16_t *lut,
    uint8_t *out
) {
    uint32_t mask;
    int bitpos = 0;
    int i;
    if (count == 0) {
        return 0;
    }
    if (count < 0 || max_len < 1 || max_len > 16 || data == NULL || lut == NULL || out == NULL) {
        return -1;
    }
    mask = (1u << max_len) - 1u;
    for (i = 0; i < count; i++) {
        int byte_i = bitpos >> 3;
        int shift = bitpos & 7;
        uint32_t b0 = (byte_i < ndata) ? data[byte_i] : 0;
        uint32_t b1 = (byte_i + 1 < ndata) ? data[byte_i + 1] : 0;
        uint32_t b2 = (byte_i + 2 < ndata) ? data[byte_i + 2] : 0;
        uint32_t peek = (b0 | (b1 << 8) | (b2 << 16)) >> shift;
        uint16_t ent = lut[peek & mask];
        uint8_t nbits = (uint8_t)(ent >> 8);
        out[i] = (uint8_t)(ent & 0xFF);
        bitpos += (int)nbits;
    }
    return 0;
}

/* PBR-E exp-rANS tile: parse freq table, rANS exponents, optional prev-exp XOR, join BF16.
 * Payload: pred:u8, book_len:u16, bit_len:u32, table[book_len], bitstream[bit_len], packed_sm[n].
 */
int pbr_decode_exp_rans_tile(
    const uint8_t *payload,
    int npayload,
    int n,
    uint16_t *out
) {
    uint32_t freq[256];
    uint32_t cumul[257];
    uint8_t lut[M];
    uint8_t pred;
    uint32_t book_len;
    uint32_t bit_len;
    int offset;
    uint32_t scale, nitems, sum;
    const uint8_t *blob;
    const uint8_t *packed_sm;
    uint32_t mask = (uint32_t)(M - 1);
    uint64_t x;
    int blob_pos;
    int i;
    uint8_t acc = 0;
    uint32_t s_i;

    if (n == 0) {
        return 0;
    }
    if (n < 0 || npayload < 7 || payload == NULL || out == NULL) {
        return -1;
    }
    pred = payload[0];
    book_len = load_u16(payload + 1);
    bit_len = load_u32(payload + 3);
    if ((uint32_t)npayload < 7u + book_len + bit_len + (uint32_t)n) {
        return -2;
    }
    offset = 7;
    if (book_len < 4) {
        return -3;
    }
    scale = load_u16(payload + offset);
    nitems = load_u16(payload + offset + 2);
    offset += 4;
    if (scale != SCALE_BITS) {
        return -4;
    }
    memset(freq, 0, sizeof(freq));
    for (s_i = 0; s_i < nitems; s_i++) {
        uint8_t sym;
        uint32_t f;
        if ((uint32_t)(offset + 3) > 7u + book_len) {
            return -5;
        }
        sym = payload[offset];
        f = load_u16(payload + offset + 1);
        freq[sym] = f;
        offset += 3;
    }
    if ((uint32_t)offset != 7u + book_len) {
        return -6;
    }
    sum = 0;
    for (i = 0; i < 256; i++) {
        sum += freq[i];
    }
    if (sum != M) {
        return -7;
    }
    blob = payload + offset;
    packed_sm = blob + bit_len;
    build_lut_cumul(freq, cumul, lut);

    x = (uint64_t)load_u32(blob);
    blob_pos = 4;
    for (i = 0; i < n; i++) {
        uint32_t cf = (uint32_t)(x & mask);
        uint8_t s = lut[cf];
        uint32_t f = freq[s];
        uint32_t start = cumul[s];
        uint8_t exp;
        uint8_t sm;
        x = (uint64_t)f * (x >> SCALE_BITS) + (uint64_t)cf - (uint64_t)start;
        if (pred == 0) {
            exp = s;
        } else {
            acc = (uint8_t)(acc ^ s);
            exp = acc;
        }
        sm = packed_sm[i];
        out[i] = (uint16_t)(((uint16_t)(sm >> 7) << 15) | ((uint16_t)exp << 7) | (uint16_t)(sm & 0x7Fu));
        while (x < (uint64_t)RANS_L && blob_pos < (int)bit_len) {
            x = (x << 8) | (uint64_t)blob[blob_pos];
            blob_pos += 1;
        }
    }
    return 0;
}

/* Bit-exact match of pbr_core.rans.rans_encode.
 * out_cap must be >= 4 + n (overflow worst-case roughly n bytes).
 * Returns encoded length, or negative on error.
 */
int pbr_rans_encode(
    const uint8_t *symbols,
    int n,
    const uint32_t *freq,   /* 256 */
    const uint32_t *cumul,  /* 257: cumul[s]=start, cumul[256]=M */
    uint8_t *out,
    int out_cap
) {
    uint64_t x;
    int i;
    int ovr_n;
    uint8_t *overflow;
    const uint32_t rans_l = RANS_L;
    const uint32_t scale = SCALE_BITS;

    if (n < 0 || (n > 0 && (symbols == NULL || freq == NULL || cumul == NULL || out == NULL))) {
        return -1;
    }
    if (n == 0) {
        if (out_cap < 4) {
            return -2;
        }
        out[0] = (uint8_t)(rans_l & 0xFF);
        out[1] = (uint8_t)((rans_l >> 8) & 0xFF);
        out[2] = (uint8_t)((rans_l >> 16) & 0xFF);
        out[3] = (uint8_t)((rans_l >> 24) & 0xFF);
        return 4;
    }
    /* overflow grows backward from end of out, state written at front */
    if (out_cap < 4 + n + 16) {
        return -2;
    }
    overflow = out + out_cap; /* grow downward */
    ovr_n = 0;
    x = (uint64_t)rans_l;
    for (i = n - 1; i >= 0; i--) {
        uint8_t s = symbols[i];
        uint32_t f = freq[s];
        uint32_t start = cumul[s];
        uint64_t x_max;
        if (f == 0) {
            return -3;
        }
        x_max = ((uint64_t)((rans_l >> scale) << 8)) * (uint64_t)f;
        while (x >= x_max) {
            ovr_n += 1;
            if (4 + ovr_n > out_cap) {
                return -4;
            }
            overflow[-ovr_n] = (uint8_t)(x & 0xFFu);
            x >>= 8;
        }
        x = ((x / (uint64_t)f) << scale) + (x % (uint64_t)f) + (uint64_t)start;
    }
    if (4 + ovr_n > out_cap) {
        return -5;
    }
    out[0] = (uint8_t)(x & 0xFFu);
    out[1] = (uint8_t)((x >> 8) & 0xFFu);
    out[2] = (uint8_t)((x >> 16) & 0xFFu);
    out[3] = (uint8_t)((x >> 24) & 0xFFu);
    if (ovr_n) {
        memcpy(out + 4, overflow - ovr_n, (size_t)ovr_n);
    }
    return 4 + ovr_n;
}
