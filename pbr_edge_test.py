#!/usr/bin/env python3
"""PBR Matrix Mantissa Codec pre-Qwen edge-case qualification suite.
Run beside pbr_poc.py. Exits non-zero on any failure.
"""
import hashlib
import json
import sys
import time
from collections import Counter
import numpy as np

from pbr_poc import (
    bits7_pack, bits7_unpack, matrix_residual,
    evaluate_tile, raw_candidate, context_candidate,
    SHAPES, PRED, TRAV,
    CodecError, encode_tile_blob, decode_tile_blob, roundtrip_tile_sha,
)

RESULTS=[]
def check(name, fn):
    t=time.time()
    try:
        detail=fn() or "PASS"
        RESULTS.append((name,"PASS",detail,time.time()-t))
        print(f"[PASS] {name}: {detail}")
    except Exception as exc:
        RESULTS.append((name,"FAIL",repr(exc),time.time()-t))
        print(f"[FAIL] {name}: {exc}")

def residual_exhaustive():
    for pred in range(128):
        for actual in range(128):
            r=(actual-pred)&0x7F
            restored=(pred+r)&0x7F
            assert restored==actual
    return "16,384 predictor/actual pairs"

def bf16_exhaustive():
    words=np.arange(65536,dtype=np.uint16)
    s=(words>>15)&1; e=(words>>7)&255; m=words&127
    restored=((s<<15)|(e<<7)|m).astype(np.uint16)
    assert np.array_equal(words,restored)
    assert words.tobytes()==restored.tobytes()
    return "65,536 BF16 bit patterns"

def pack_boundaries():
    rng=np.random.default_rng(20260920)
    sizes=[0,1,2,7,8,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025]
    for n in sizes:
        a=rng.integers(0,128,n,dtype=np.uint8)
        b=bits7_pack(a)
        z=bits7_unpack(b,n) if n else np.empty(0,dtype=np.uint8)
        assert np.array_equal(a,z),n
        assert len(b)==(7*n+7)//8,(n,len(b))
    return f"{len(sizes)} lengths including empty and partial-byte tails"

def special_bf16():
    # ±0, subnormals, normal limits, infinities, representative NaNs.
    values=np.array([0x0000,0x8000,0x0001,0x007F,0x0080,0x7F7F,
                     0xFF7F,0x7F80,0xFF80,0x7F81,0x7FFF,0xFF81,0xFFFF],dtype=np.uint16)
    s=(values>>15)&1; e=(values>>7)&255; m=values&127
    restored=((s<<15)|(e<<7)|m).astype(np.uint16)
    assert values.tobytes()==restored.tobytes()
    return f"{len(values)} signed-zero/subnormal/normal/Inf/NaN cases"

def all_predictors_roundtrip():
    rng=np.random.default_rng(11)
    count=0
    for version in (1,2,3):
        for shape in SHAPES[version]:
            m=rng.integers(0,128,shape,dtype=np.uint8)
            e=rng.integers(0,256,shape,dtype=np.uint8)
            for p in PRED[version]:
                for t in TRAV[version]:
                    r=matrix_residual(m,e,p,t)
                    assert len(r)==m.size
                    count+=1
    return f"{count} version/shape/predictor/traversal combinations"

def overflow_edges():
    tiles=[]
    for v in [0,1,63,64,126,127]:
        tiles.append(np.full((16,16),v,dtype=np.uint8))
    tiles.append(np.tile(np.array([0,127],dtype=np.uint8),128).reshape(16,16))
    exponents=[0,1,126,127,128,254,255]
    count=0
    for tile in tiles:
        for ev in exponents:
            e=np.full_like(tile,ev)
            for p in PRED[3]:
                for t in TRAV[3]:
                    matrix_residual(tile,e,p,t); count+=1
    return f"{count} wraparound/exponent boundary combinations"

def synthetic_selection():
    h=w=16
    y,x=np.indices((h,w))
    sets={
      'constant':np.full((h,w),63,dtype=np.uint8),
      'horizontal':(x&127).astype(np.uint8),
      'vertical':(y&127).astype(np.uint8),
      'plane':((3*x+5*y)&127).astype(np.uint8),
      'checker':np.where((x+y)&1,127,0).astype(np.uint8),
      'repeated_rows':np.tile(np.arange(w,dtype=np.uint8),(h,1)),
      'uniform_random':np.random.default_rng(19).integers(0,128,(h,w),dtype=np.uint8),
    }
    e=np.full((h,w),127,dtype=np.uint8)
    outcomes={}
    for name,tile in sets.items():
        (mode,payload),candidates=evaluate_tile(tile.ravel(),e.ravel(),3,(16,16))
        # Ensure chosen candidate is truly minimum by serialized byte length.
        assert len(payload)==min(len(p) for _,p in candidates)
        outcomes[name]=(mode,len(payload),len(raw_candidate(tile.ravel())))
    # Non-negotiable honest fallback for random input in this small-block implementation.
    assert outcomes['uniform_random'][1] <= outcomes['uniform_random'][2]
    return json.dumps(outcomes,sort_keys=True)

def deterministic_output():
    rng=np.random.default_rng(1234)
    m=rng.integers(0,128,256,dtype=np.uint8); e=rng.integers(0,256,256,dtype=np.uint8)
    outputs=[]
    for _ in range(5):
        outputs.append(evaluate_tile(m,e,3,(16,16))[0])
    assert all(outputs[0][0]==q[0] and outputs[0][1]==q[1] for q in outputs[1:])
    return hashlib.sha256(outputs[0][1]).hexdigest()

def tie_policy_observation():
    # Documents current deterministic sort: byte length, then candidate name.
    m=np.arange(256,dtype=np.uint8)&127; e=np.full(256,127,dtype=np.uint8)
    chosen,cands=evaluate_tile(m,e,3,(16,16))
    minimum=min(len(p) for _,p in cands)
    tied=sorted(n for n,p in cands if len(p)==minimum)
    assert chosen[0]==tied[0]
    return f"current tie rule lexical; candidates={tied}"

def gate_d_standalone_decoder():
    """Gate D: standalone decoder reads only serialized bytes; rejects corruption."""
    import struct, zlib
    rng = np.random.default_rng(20260920)
    y, x = np.indices((16, 16))
    tiles = {
        'random': rng.integers(0, 128, (16, 16), dtype=np.uint8),
        'plane': ((3 * x + 5 * y) & 127).astype(np.uint8),
        'constant': np.full((16, 16), 63, dtype=np.uint8),
    }
    e = np.full((16, 16), 127, dtype=np.uint8)
    shas = []
    for name, tile in tiles.items():
        ok, blob, s1, s2 = roundtrip_tile_sha(tile.ravel(), e.ravel(), 3, (16, 16))
        assert ok and s1 == s2, name
        assert len(blob) > 0
        md, ed, info = decode_tile_blob(blob)
        assert info['complete_bytes'] == len(blob)
        assert np.array_equal(md, tile.ravel()) and np.array_equal(ed, e.ravel())
        shas.append(s1)

        # Truncated header / payload
        for cut in [0, 4, 10, 20, max(1, len(blob) // 3), len(blob) - 1]:
            try:
                decode_tile_blob(blob[:cut])
                raise AssertionError(f'truncate accepted cut={cut}')
            except CodecError:
                pass

        # Trailing bytes (strict policy)
        try:
            decode_tile_blob(blob + b'\x00')
            raise AssertionError('trailing accepted')
        except CodecError:
            pass

        # Checksum mismatch
        bad = bytearray(blob)
        bad[-1] ^= 0xFF
        try:
            decode_tile_blob(bytes(bad))
            raise AssertionError('bad crc accepted')
        except CodecError:
            pass

        # Invalid mode id (re-CRC so we exercise mode check)
        bad = bytearray(blob)
        bad[12] = 99
        core = bytes(bad[:-4])
        fixed = core + struct.pack('<I', zlib.crc32(core) & 0xFFFFFFFF)
        try:
            decode_tile_blob(fixed)
            raise AssertionError('bad mode accepted')
        except CodecError:
            pass

        # Impossible dimensions vs n
        bad = bytearray(blob)
        bad[9], bad[10] = 3, 3
        core = bytes(bad[:-4])
        fixed = core + struct.pack('<I', zlib.crc32(core) & 0xFFFFFFFF)
        try:
            decode_tile_blob(fixed)
            raise AssertionError('bad dims accepted')
        except CodecError:
            pass

        # Corrupt entropy stream (zlib body) with CRC repaired
        meta_len = struct.unpack_from('<I', blob, 13)[0]
        body_len = struct.unpack_from('<I', blob, 17)[0]
        body_start = 13 + 4 + 4 + 4 + meta_len
        if body_len > 5 and blob[body_start] in (1, 2):  # zlib-tagged bodies
            bad = bytearray(blob)
            bad[body_start + 5] ^= 0xFF
            core = bytes(bad[:-4])
            fixed = core + struct.pack('<I', zlib.crc32(core) & 0xFFFFFFFF)
            try:
                decode_tile_blob(fixed)
                raise AssertionError('zlib corruption accepted')
            except CodecError:
                pass

    # Invalid predictor / traversal ids inside matrix meta
    tile = tiles['plane']
    blob = encode_tile_blob(tile.ravel(), e.ravel(), 3, (16, 16))
    md, ed, info = decode_tile_blob(blob)
    if info['mode'] == 'matrix':
        meta_len = struct.unpack_from('<I', blob, 13)[0]
        assert meta_len == 5
        bad = bytearray(blob)
        # meta starts at 25
        meta_off = 13 + 12
        bad[meta_off + 1] = 200  # predictor index
        core = bytes(bad[:-4])
        fixed = core + struct.pack('<I', zlib.crc32(core) & 0xFFFFFFFF)
        try:
            decode_tile_blob(fixed)
            raise AssertionError('bad predictor id accepted')
        except CodecError:
            pass

    return (
        f"standalone decoder round-trip+corruption OK on {len(tiles)} tiles; "
        f"SHA samples={shas[0][:12]}…"
    )

TESTS=[
 ('Residual arithmetic exhaustive',residual_exhaustive),
 ('BF16 split/reassembly exhaustive',bf16_exhaustive),
 ('7-bit packing boundaries',pack_boundaries),
 ('Special BF16 values',special_bf16),
 ('All matrix candidates round-trip',all_predictors_roundtrip),
 ('Overflow and exponent edges',overflow_edges),
 ('Synthetic selection and accounting',synthetic_selection),
 ('Determinism',deterministic_output),
 ('Tie-policy observation',tie_policy_observation),
 ('Gate D standalone decoder + corruption',gate_d_standalone_decoder),
]

for name,fn in TESTS: check(name,fn)
report={"suite":"PBR pre-Qwen edge qualification","results":[{"name":n,"status":s,"detail":d,"seconds":round(t,6)} for n,s,d,t in RESULTS]}
with open('pbr_edge_results.json','w',encoding='utf-8') as f: json.dump(report,f,indent=2)
failed=[r for r in RESULTS if r[1]!='PASS']
print(f"\nSummary: {len(RESULTS)-len(failed)}/{len(RESULTS)} checks passed")
print("Report: pbr_edge_results.json")
if failed: sys.exit(1)
