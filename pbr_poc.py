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
