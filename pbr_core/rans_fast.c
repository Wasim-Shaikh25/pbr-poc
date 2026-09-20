/* Bit-exact match of pbr_core.rans.rans_decode (byte-renormalized 32-bit rANS). */
#include <stdint.h>
#include <string.h>

#define RANS_L (1u << 23)
#define SCALE_BITS 12
#define M (1u << SCALE_BITS)

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
    x = (uint64_t)blob[0]
        | ((uint64_t)blob[1] << 8)
        | ((uint64_t)blob[2] << 16)
        | ((uint64_t)blob[3] << 24);
    pos = 4;
    for (i = 0; i < count; i++) {
        uint32_t cf = (uint32_t)(x & mask);
        uint8_t s = lut[cf];
        int64_t f = freq[s];
        int64_t start = cumul[s];
        x = (uint64_t)f * (x >> SCALE_BITS) + (uint64_t)cf - (uint64_t)start;
        out[i] = s;
        while (x < (uint64_t)RANS_L && pos < nblob) {
            x = (x << 8) | (uint64_t)blob[pos];
            pos += 1;
        }
    }
    return 0;
}
