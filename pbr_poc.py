#!/usr/bin/env python3
import struct,zlib,numpy as np
SHAPES={1:[(16,16)],2:[(8,8),(8,16),(16,16),(16,32)],3:[(8,8),(8,16),(16,16),(16,32),(32,32)]}
PRED={1:['left','up','previous'],2:['left','up','previous','avg'],3:['left','up','previous','avg','paeth','exp_left']}
TRAV={1:['row','serp'],2:['row','serp','col'],3:['row','serp','col','col_serp']}
def bits7_pack(a):
 a=np.asarray(a,dtype=np.uint8).ravel(); shifts=np.arange(6,-1,-1,dtype=np.uint8)
 return np.packbits(((a[:,None]>>shifts)&1).reshape(-1)).tobytes()
def bits7_unpack(b,n):
 if n==0:return np.empty(0,dtype=np.uint8)
 q=np.unpackbits(np.frombuffer(b,dtype=np.uint8))[:n*7].reshape(n,7)
 return np.sum(q*(1<<np.arange(6,-1,-1)),axis=1,dtype=np.uint16).astype(np.uint8)
def positions(h,w,t):
 if t.startswith('row'):
  for y in range(h):
   xs=range(w-1,-1,-1) if t=='serp' and y&1 else range(w)
   for x in xs:yield y,x
 else:
  for x in range(w):
   ys=range(h-1,-1,-1) if t=='col_serp' and x&1 else range(h)
   for y in ys:yield y,x
def paeth(a,b,c):
 p=a+b-c; d=[abs(p-a),abs(p-b),abs(p-c)];return (a,b,c)[d.index(min(d))]
def predict(out,ex,y,x,k,prev):
 l=int(out[y,x-1]) if x else 0;u=int(out[y-1,x]) if y else 0;ul=int(out[y-1,x-1]) if y and x else 0
 return {'left':l,'up':u,'previous':prev,'avg':(l+u)//2,'paeth':paeth(l,u,ul)&127,'exp_left':(l^((int(ex[y,x])-127)*19))&127}[k]
def matrix_residual(tile,exp,kind,trav):
 out=np.zeros_like(tile,dtype=np.uint8);rs=[];prev=0
 for y,x in positions(*tile.shape,trav):
  p=predict(out,exp,y,x,kind,prev);v=int(tile[y,x]);r=(v-p)&127;rs.append(r);out[y,x]=(p+r)&127;prev=int(out[y,x])
 assert np.array_equal(out,tile);return np.asarray(rs,dtype=np.uint8)
def raw_candidate(a):
 p=bits7_pack(a);return b'R'+struct.pack('<I',len(p))+p
def zpayload(tag,a):
 c=zlib.compress(np.asarray(a,dtype=np.uint8).tobytes(),9);return bytes([tag])+struct.pack('<I',len(c))+c
def context_candidate(m,e):
 m=np.asarray(m,dtype=np.uint8);e=np.asarray(e,dtype=np.uint8);flags=[];lits=[];prev=int(m[0])
 for i in range(1,len(m)):
  p=(prev^((int(e[i])-124)*19)^((i%32)*7))&127;v=int(m[i]);flags.append(v!=p)
  if v!=p:lits.append(v)
  prev=v
 fb=np.packbits(np.asarray(flags,dtype=np.uint8)).tobytes();body=bytes([int(m[0])])+struct.pack('<I',len(fb))+fb+bits7_pack(lits)
 return b'C'+struct.pack('<I',len(body))+body
def evaluate_tile(m,e,version,shape):
 h,w=shape;tile=np.asarray(m,dtype=np.uint8).reshape(h,w);ex=np.asarray(e,dtype=np.uint8).reshape(h,w)
 cand=[('raw',raw_candidate(m)),('symbol_zlib',zpayload(1,m)),('context',context_candidate(m,e))]
 for t in TRAV[version]:
  for p in PRED[version]:
   r=matrix_residual(tile,ex,p,t);head=bytes([version,PRED[version].index(p),TRAV[version].index(t),h,w])
   cand.append((f'matrix:{shape}:{t}:{p}',head+zpayload(2,r)))
 cand.sort(key=lambda q:(len(q[1]),q[0]));return cand[0],cand


# --- Gate D: standalone container + corruption-rejecting decoder ---
import hashlib as _hashlib

class CodecError(ValueError):
    """Raised when a serialized stream is truncated, inconsistent, or corrupt."""

MAGIC = b'PBRM'
CONT_VER = 1
MODE_RAW, MODE_ZLIB, MODE_CTX, MODE_MAT = 0, 1, 2, 3
# Strict trailing-byte policy: any extra bytes after a complete frame are rejected.

def _crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF

def _u32(n: int) -> bytes:
    if not 0 <= n <= 0xFFFFFFFF:
        raise CodecError(f'u32 out of range: {n}')
    return struct.pack('<I', n)

def _read_exact(buf: bytes, off: int, n: int) -> tuple[bytes, int]:
    if off < 0 or n < 0 or off + n > len(buf):
        raise CodecError(f'truncated read off={off} n={n} len={len(buf)}')
    return buf[off:off+n], off + n

def encode_tile_blob(m, e, version, shape, mode_name=None, payload=None):
    """Serialize one tile into a self-contained blob (bytes-only decode).

    If mode_name/payload omitted, picks via evaluate_tile (shortest, then name).
    Exponent plane is always stored (needed for context / exp_left and for
    independent SHA round-trips of the full BF16 split).
    """
    m = np.asarray(m, dtype=np.uint8).ravel()
    e = np.asarray(e, dtype=np.uint8).ravel()
    if m.size != e.size:
        raise CodecError('mantissa/exponent length mismatch')
    h, w = shape
    if h * w != m.size:
        raise CodecError(f'shape {shape} does not match n={m.size}')
    if mode_name is None or payload is None:
        (mode_name, payload), _ = evaluate_tile(m, e, version, shape)
    if mode_name == 'raw':
        mode = MODE_RAW
        meta = b''
        body = payload  # already R+len+bits7
    elif mode_name == 'symbol_zlib':
        mode = MODE_ZLIB
        meta = b''
        body = payload
    elif mode_name == 'context':
        mode = MODE_CTX
        meta = b''
        body = payload
    elif mode_name.startswith('matrix:'):
        mode = MODE_MAT
        # payload = head(5) + zpayload(2, residual)
        if len(payload) < 5:
            raise CodecError('matrix payload too short for head')
        meta = payload[:5]
        body = payload[5:]
        v, pi, ti, hh, ww = meta
        if v != version or hh != h or ww != w:
            raise CodecError('matrix head does not match encode args')
        if pi >= len(PRED[version]) or ti >= len(TRAV[version]):
            raise CodecError('matrix predictor/traversal index out of range')
    else:
        raise CodecError(f'unknown mode {mode_name!r}')

    # Header without CRC: magic, cont_ver, n, h, w, codec_version, mode, meta_len, body_len, exp_len
    exp_bytes = e.tobytes()  # raw uint8 exponents; honest size accounting
    meta = bytes(meta)
    body = bytes(body)
    hdr = (
        MAGIC
        + bytes([CONT_VER])
        + _u32(m.size)
        + bytes([h & 0xFF, w & 0xFF, version & 0xFF, mode & 0xFF])
        + _u32(len(meta))
        + _u32(len(body))
        + _u32(len(exp_bytes))
        + meta
    )
    core = hdr + body + exp_bytes
    return core + _u32(_crc(core))

def decode_tile_blob(blob: bytes):
    """Decode a blob produced by encode_tile_blob. Returns (mantissas, exponents, info).

    Raises CodecError on any malformation. Strict: no trailing bytes allowed.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise CodecError('blob must be bytes')
    blob = bytes(blob)
    if len(blob) < 4 + 1 + 4 + 4 + 4 + 4 + 4 + 4:  # rough minimum
        raise CodecError('blob too short for container header')

    off = 0
    mag, off = _read_exact(blob, off, 4)
    if mag != MAGIC:
        raise CodecError(f'bad magic {mag!r}')
    cont_ver_b, off = _read_exact(blob, off, 1)
    cont_ver = cont_ver_b[0]
    if cont_ver != CONT_VER:
        raise CodecError(f'unsupported container version {cont_ver}')
    n_b, off = _read_exact(blob, off, 4)
    n = struct.unpack('<I', n_b)[0]
    if n > 1 << 24:
        raise CodecError(f'n too large: {n}')
    hwvm, off = _read_exact(blob, off, 4)
    h, w, version, mode = hwvm
    if version not in PRED or version not in TRAV or version not in SHAPES:
        raise CodecError(f'invalid codec version {version}')
    if h == 0 or w == 0:
        raise CodecError('impossible dimensions (zero)')
    if h * w != n:
        raise CodecError(f'dimensions {h}x{w} != n {n}')
    # reject absurd tiles outside known shape sets (still allow exact listed shapes)
    if (h, w) not in SHAPES[version]:
        # allow only declared shapes for this qualification codec
        raise CodecError(f'shape {(h,w)} not allowed for version {version}')
    meta_len_b, off = _read_exact(blob, off, 4)
    body_len_b, off = _read_exact(blob, off, 4)
    exp_len_b, off = _read_exact(blob, off, 4)
    meta_len = struct.unpack('<I', meta_len_b)[0]
    body_len = struct.unpack('<I', body_len_b)[0]
    exp_len = struct.unpack('<I', exp_len_b)[0]
    if meta_len > 64 or body_len > (1 << 26) or exp_len > (1 << 26):
        raise CodecError('declared section length too large')
    meta, off = _read_exact(blob, off, meta_len)
    body, off = _read_exact(blob, off, body_len)
    exp_bytes, off = _read_exact(blob, off, exp_len)
    crc_b, off = _read_exact(blob, off, 4)
    if off != len(blob):
        raise CodecError(f'trailing bytes: {len(blob) - off}')
    expect = _crc(blob[:-4])
    got = struct.unpack('<I', crc_b)[0]
    if got != expect:
        raise CodecError(f'checksum mismatch got=0x{got:08x} expect=0x{expect:08x}')
    if exp_len != n:
        raise CodecError(f'exp_len {exp_len} != n {n}')
    exponents = np.frombuffer(exp_bytes, dtype=np.uint8).copy()

    if mode == MODE_RAW:
        mant = _decode_raw_body(body, n)
        mode_name = 'raw'
    elif mode == MODE_ZLIB:
        mant = _decode_zlib_syms(body, n)
        mode_name = 'symbol_zlib'
    elif mode == MODE_CTX:
        mant = _decode_context_body(body, exponents)
        mode_name = 'context'
    elif mode == MODE_MAT:
        mant = _decode_matrix_body(meta, body, exponents, version, h, w)
        mode_name = 'matrix'
    else:
        raise CodecError(f'invalid mode id {mode}')

    if mant.size != n:
        raise CodecError('decoded mantissa length mismatch')
    info = {
        'n': n, 'shape': (h, w), 'version': version, 'mode': mode_name,
        'complete_bytes': len(blob),
    }
    return mant, exponents, info

def _decode_raw_body(body: bytes, n: int) -> np.ndarray:
    if len(body) < 5 or body[0:1] != b'R':
        raise CodecError('raw body framing')
    plen = struct.unpack('<I', body[1:5])[0]
    if plen != len(body) - 5:
        raise CodecError('raw payload length mismatch')
    need = (7 * n + 7) // 8
    if plen != need:
        raise CodecError(f'raw packed size {plen} != need {need}')
    return bits7_unpack(body[5:], n)

def _decode_zlib_syms(body: bytes, n: int) -> np.ndarray:
    if len(body) < 5:
        raise CodecError('zlib body too short')
    tag = body[0]
    if tag != 1:
        raise CodecError(f'unexpected zlib tag {tag}')
    plen = struct.unpack('<I', body[1:5])[0]
    if plen != len(body) - 5:
        raise CodecError('zlib payload length mismatch')
    try:
        raw = zlib.decompress(body[5:])
    except zlib.error as exc:
        raise CodecError(f'corrupted zlib stream: {exc}') from exc
    if len(raw) != n:
        raise CodecError('zlib symbol count mismatch')
    return np.frombuffer(raw, dtype=np.uint8).copy()

def _decode_context_body(body: bytes, exponents: np.ndarray) -> np.ndarray:
    if len(body) < 5 or body[0:1] != b'C':
        raise CodecError('context body framing')
    blen = struct.unpack('<I', body[1:5])[0]
    if blen != len(body) - 5:
        raise CodecError('context length mismatch')
    inner = body[5:]
    if len(inner) < 1 + 4:
        raise CodecError('context inner truncated')
    n = exponents.size
    out = np.empty(n, dtype=np.uint8)
    out[0] = inner[0]
    fb_len = struct.unpack('<I', inner[1:5])[0]
    if 5 + fb_len > len(inner):
        raise CodecError('context flags truncated')
    fb = inner[5:5+fb_len]
    lit_packed = inner[5+fb_len:]
    if n == 0:
        return out
    flags = np.unpackbits(np.frombuffer(fb, dtype=np.uint8))[: n - 1]
    if flags.size != n - 1:
        # packbits pads to byte; we already sliced to n-1
        pass
    n_miss = int(flags.sum()) if flags.size else 0
    need = (7 * n_miss + 7) // 8 if n_miss else 0
    if len(lit_packed) < need:
        raise CodecError('context literals truncated')
    # reject extra literal bytes
    if len(lit_packed) != need:
        raise CodecError('context literal packing size mismatch')
    lits = bits7_unpack(lit_packed, n_miss) if n_miss else np.empty(0, dtype=np.uint8)
    li = 0
    prev = int(out[0])
    for i in range(1, n):
        p = (prev ^ ((int(exponents[i]) - 124) * 19) ^ ((i % 32) * 7)) & 127
        if flags[i - 1]:
            if li >= len(lits):
                raise CodecError('context literal underrun')
            v = int(lits[li]); li += 1
        else:
            v = p
        out[i] = v
        prev = v
    if li != len(lits):
        raise CodecError('context literal overrun')
    return out

def _decode_matrix_body(meta: bytes, body: bytes, exponents: np.ndarray, version: int, h: int, w: int) -> np.ndarray:
    if len(meta) != 5:
        raise CodecError('matrix meta must be 5 bytes')
    v, pi, ti, hh, ww = meta
    if v != version or hh != h or ww != w:
        raise CodecError('matrix meta mismatch')
    if pi >= len(PRED[version]) or ti >= len(TRAV[version]):
        raise CodecError('invalid predictor/traversal id')
    kind = PRED[version][pi]
    trav = TRAV[version][ti]
    if len(body) < 5:
        raise CodecError('matrix residual body truncated')
    tag = body[0]
    if tag != 2:
        raise CodecError(f'unexpected matrix residual tag {tag}')
    plen = struct.unpack('<I', body[1:5])[0]
    if plen != len(body) - 5:
        raise CodecError('matrix residual length mismatch')
    try:
        raw = zlib.decompress(body[5:])
    except zlib.error as exc:
        raise CodecError(f'corrupted matrix residual zlib: {exc}') from exc
    if len(raw) != h * w:
        raise CodecError('matrix residual count mismatch')
    residuals = np.frombuffer(raw, dtype=np.uint8)
    ex = exponents.reshape(h, w)
    out = np.zeros((h, w), dtype=np.uint8)
    prev = 0
    for (y, x), r in zip(positions(h, w, trav), residuals):
        p = predict(out, ex, y, x, kind, prev)
        out[y, x] = (p + int(r)) & 127
        prev = int(out[y, x])
    return out.ravel()

def roundtrip_tile_sha(m, e, version, shape):
    """Encode then decode; return (ok, blob, sha_src, sha_dst)."""
    m = np.asarray(m, dtype=np.uint8).ravel()
    e = np.asarray(e, dtype=np.uint8).ravel()
    blob = encode_tile_blob(m, e, version, shape)
    md, ed, info = decode_tile_blob(blob)
    if info['complete_bytes'] != len(blob):
        raise CodecError('complete_bytes != physical length')
    sha_src = _hashlib.sha256(m.tobytes() + e.tobytes()).hexdigest()
    sha_dst = _hashlib.sha256(md.tobytes() + ed.tobytes()).hexdigest()
    if sha_src != sha_dst or not np.array_equal(m, md) or not np.array_equal(e, ed):
        return False, blob, sha_src, sha_dst
    return True, blob, sha_src, sha_dst
