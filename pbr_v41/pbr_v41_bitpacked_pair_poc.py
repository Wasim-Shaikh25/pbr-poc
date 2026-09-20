#!/usr/bin/env python3
"""PBR V4.1 bit-packed residual-pair codec PoC.
Exact synthetic qualification and statistics probe; not a Qwen result.
"""
import argparse,csv,hashlib,math,struct
from collections import Counter
import numpy as np
PRED=('left','up','previous','avg','paeth'); TRAV=('row','serp','col','col_serp'); PAIR=('traversal','horizontal','vertical'); KS=(16,32,64,128,256)
class BW:
 def __init__(self):self.a=[]
 def put(self,v,n):self.a.extend((v>>i)&1 for i in range(n-1,-1,-1))
 def bytes(self):return np.packbits(np.array(self.a,dtype=np.uint8)).tobytes(),len(self.a)
class BR:
 def __init__(self,b,nb):self.a=np.unpackbits(np.frombuffer(b,dtype=np.uint8))[:nb];self.i=0
 def get(self,n):
  if self.i+n>len(self.a):raise ValueError('truncated bits')
  v=0
  for x in self.a[self.i:self.i+n]:v=(v<<1)|int(x)
  self.i+=n;return v
def positions(h,w,t):
 if t in ('row','serp'):
  for y in range(h):
   xs=range(w-1,-1,-1) if t=='serp' and y&1 else range(w)
   for x in xs:yield y,x
 else:
  for x in range(w):
   ys=range(h-1,-1,-1) if t=='col_serp' and x&1 else range(h)
   for y in ys:yield y,x
def pae(a,b,c):
 p=a+b-c;d=(abs(p-a),abs(p-b),abs(p-c));return (a,b,c)[d.index(min(d))]
def pred(o,s,y,x,k,prev):
 l=int(o[y,x-1]) if x and s[y,x-1] else None;u=int(o[y-1,x]) if y and s[y-1,x] else None;ul=int(o[y-1,x-1]) if y and x and s[y-1,x-1] else None
 if k=='previous':return prev
 if k=='left':return l if l is not None else prev
 if k=='up':return u if u is not None else prev
 if k=='avg':return (l+u)//2 if l is not None and u is not None else (l if l is not None else (u if u is not None else prev))
 return pae(l,u,ul)&127 if l is not None and u is not None and ul is not None else (l if l is not None else (u if u is not None else prev))
def residual(a,k,t):
 a=np.asarray(a,dtype=np.uint8);o=np.zeros_like(a);s=np.zeros_like(a,bool);r=np.zeros_like(a);p=0
 for y,x in positions(*a.shape,t):q=pred(o,s,y,x,k,p);r[y,x]=(int(a[y,x])-q)&127;o[y,x]=(q+int(r[y,x]))&127;s[y,x]=1;p=int(o[y,x])
 assert np.array_equal(a,o);return r
def rebuild(r,k,t):
 o=np.zeros_like(r);s=np.zeros_like(r,bool);p=0
 for y,x in positions(*r.shape,t):q=pred(o,s,y,x,k,p);o[y,x]=(q+int(r[y,x]))&127;s[y,x]=1;p=int(o[y,x])
 return o
def coords(h,w,q,t):
 if q=='traversal':
  z=list(positions(h,w,t))
  for i in range(0,len(z),2):yield z[i],z[i+1] if i+1<len(z) else None
 elif q=='horizontal':
  for y in range(h):
   for x in range(0,w,2):yield (y,x),(y,x+1) if x+1<w else None
 else:
  for y in range(0,h,2):
   for x in range(w):yield (y,x),(y+1,x) if y+1<h else None
def pairs(r,q,t):return [(int(r[a]),int(r[b]) if b else 128) for a,b in coords(*r.shape,q,t)]
def unpairs(ps,h,w,q,t):
 r=np.zeros((h,w),np.uint8);cs=list(coords(h,w,q,t));assert len(cs)==len(ps)
 for (a,b),(x,y) in zip(cs,ps):r[a]=x
 for (a,b),(x,y) in zip(cs,ps):
  if b:r[b]=y
  elif y!=128:raise ValueError('tail')
 return r
def train(ps,k):return [p for p,c in sorted(Counter(ps).items(),key=lambda z:(-z[1],z[0]))[:k]]
def entropy(ps):
 c=Counter(ps);n=len(ps);return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0
def encode(a,k,t,q,d):
 r=residual(a,k,t);ps=pairs(r,q,t);idx={p:i for i,p in enumerate(d)};ib=max(1,math.ceil(math.log2(max(1,len(d)))));bw=BW();hits=0
 for x,y in ps:
  if (x,y) in idx:bw.put(0,1);bw.put(idx[(x,y)],ib);hits+=1
  else:bw.put(1,1);bw.put(x,7);bw.put(1 if y==128 else 0,1);bw.put(0 if y==128 else y,7)
 body,nb=bw.bytes();head=struct.pack('<4sBBBBHHHHII',b'P41B',1,PRED.index(k),TRAV.index(t),PAIR.index(q),a.shape[0],a.shape[1],len(d),ib,len(ps),nb)
 db=b''.join(struct.pack('<H',(x<<8)|y) for x,y in d);return head+db+body,dict(hits=hits,pairs=len(ps),bits=len(head+db)*8+nb,entropy=entropy(ps))
def decode(p):
 fmt='<4sBBBBHHHHII';hs=struct.calcsize(fmt)
 if len(p)<hs:raise ValueError('header')
 m,v,ki,ti,qi,h,w,n,ib,npair,nb=struct.unpack(fmt,p[:hs])
 if m!=b'P41B' or v!=1 or ki>=len(PRED) or ti>=len(TRAV) or qi>=len(PAIR):raise ValueError('ids')
 off=hs;need=off+2*n
 if need>len(p):raise ValueError('dict')
 d=[]
 for i in range(off,need,2):z=struct.unpack('<H',p[i:i+2])[0];d.append((z>>8,z&255))
 needbytes=(nb+7)//8
 if need+needbytes!=len(p):raise ValueError('length')
 br=BR(p[need:],nb);ps=[]
 for _ in range(npair):
  if br.get(1)==0:
   j=br.get(ib)
   if j>=len(d):raise ValueError('dict id')
   ps.append(d[j])
  else:
   x=br.get(7);tail=br.get(1);y=br.get(7);ps.append((x,128 if tail else y))
 if br.i!=nb:raise ValueError('trailing bits')
 return rebuild(unpairs(ps,h,w,PAIR[qi],TRAV[ti]),PRED[ki],TRAV[ti])
def synth(kind,h,w,seed):
 y,x=np.indices((h,w));g=np.random.default_rng(seed)
 if kind=='random':return g.integers(0,128,(h,w),dtype=np.uint8)
 if kind=='constant':return np.full((h,w),63,np.uint8)
 if kind=='horizontal':return (x&127).astype(np.uint8)
 if kind=='vertical':return (y&127).astype(np.uint8)
 if kind=='plane':return ((3*x+5*y)&127).astype(np.uint8)
 base=(x+2*y).astype(np.int16);return ((base+g.choice([-1,0,1],(h,w)))&127).astype(np.uint8)
def probe(kind,h,w):
 tr=synth(kind,h,w,11);ev=synth(kind,h,w,29);out=[]
 for k in PRED:
  for t in TRAV:
   rt=residual(tr,k,t);re=residual(ev,k,t)
   for q in PAIR:
    tp=pairs(rt,q,t);ep=pairs(re,q,t);ec=Counter(ep)
    for K in KS:
     d=train(tp,K);p,s=encode(ev,k,t,q,d);assert np.array_equal(decode(p),ev)
     rawbits=7*ev.size;out.append(dict(dataset=kind,predictor=k,traversal=t,pairing=q,K=K,dictionary_entries=len(d),pair_entropy=s['entropy'],coverage=sum(ec[z] for z in d)/len(ep),hits=s['hits'],pairs=s['pairs'],encoded_bits=s['bits'],raw_bits=rawbits,bpw=s['bits']/ev.size,ratio=s['bits']/rawbits,sha256=hashlib.sha256(p).hexdigest()))
 return out
def selftest():
 for p in range(128):
  for a in range(128):assert (p+((a-p)&127))&127==a
 for shape in [(1,1),(1,17),(17,1),(15,17),(16,16)]:
  a=np.random.default_rng(sum(shape)).integers(0,128,shape,dtype=np.uint8)
  r=residual(a,'paeth','serp');d=train(pairs(r,'horizontal','serp'),16);z,_=encode(a,'paeth','serp','horizontal',d);assert np.array_equal(decode(z),a)
  for bad in (z[:3],z[:-1],z+b'x'):
   try:decode(bad);raise AssertionError('accepted corruption')
   except ValueError:pass
 print('PBR V4.1 self-test PASS')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--dataset',default='all');ap.add_argument('--height',type=int,default=64);ap.add_argument('--width',type=int,default=64);ap.add_argument('--output',default='pbr_v41.csv');a=ap.parse_args()
 if a.self_test:selftest();return
 kinds=('random','constant','horizontal','vertical','plane','near') if a.dataset=='all' else (a.dataset,);rows=[]
 for k in kinds:rows+=probe(k,a.height,a.width)
 with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 for k in kinds:
  b=min((r for r in rows if r['dataset']==k),key=lambda r:r['encoded_bits']);print(k,{x:b[x] for x in ('predictor','traversal','pairing','K','coverage','pair_entropy','bpw','ratio')})
 print('Wrote',len(rows),'rows')
if __name__=='__main__':main()
