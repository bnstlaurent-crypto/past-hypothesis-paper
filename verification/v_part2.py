import numpy as np, json, mpmath as mp, sys
from fractions import Fraction
sys.path.insert(0,'/home/claude')
exec(open('/home/claude/ph2.py').read().rsplit('if __name__',1)[0])
out=[]
def rep(tag,ok,msg): out.append(f"[{'PASS' if ok else 'FAIL'}] {tag}: {msg}"); print(out[-1],flush=True)
# ---- constants in Theorems B/C
r0=1.0; eps=0.1; t=eps/2; delta=eps/(8*np.sqrt(2))
cP=2/delta*1; cP=2/(delta)   # (2n/delta) per n
cM=5/delta
rep("Theorem constants", abs(4*np.sqrt(2)*delta-eps/2)<1e-15 and abs(cP*1-226.27)<0.01 and abs(cM-565.69)<0.01 and abs(1/(t*t/48)-19200)<1e-6
    and (np.pi/4 and (1-np.pi/4)-0.1>=0.114) and 0.32<np.sqrt(0.114),
    f"4*sqrt2*delta = eps/2; 2/delta = {cP:.2f} (paper 227); 5/delta = {cM:.2f} (paper 566); 48/t^2 = {48/(t*t):.0f}; (1-pi/4)-0.1 = {1-np.pi/4-0.1:.4f}; sqrt(0.114)={np.sqrt(0.114):.4f} > 0.32")
def nontriv(n,r0=1.0,base=227,expo=6): return expo*n*np.log(base*n/r0**2) - 2**n*r0**4/19200 < 0
nmin=next(n for n in range(2,60) if nontriv(n))
rep("Nontriviality threshold (paper: n~24 at r0=1)",nmin in (23,24,25),f"smallest n with non-trivial bound = {nmin}")
# 'optimistic constants' claim: net (1/delta)^{3n} per state, Gaussian-type concentration exp(-N t^2/2)
def opt(n): return 6*n*np.log(1/(eps/8)) - 2**n*(eps/2)**2/2 < 0
nopt=next(n for n in range(2,60) if opt(n))
rep("Optimistic-constants threshold (paper says n>=15)",nopt>=15,f"even with optimistic constants the bound needs n >= {nopt}")
# ---- MPS dimension for chi=2 via Jacobian rank (paper: d = 8n-10)
def mps_state(params,n,chi,dims):
    k=0; W=np.eye(1)
    for i in range(n):
        r,c=dims[i]*2,dims[i+1]; A=params[k:k+r*c].reshape(r,c)+1j*params[k+r*c:k+2*r*c].reshape(r,c); k+=2*r*c
        W=np.kron(W,np.eye(2))@A
    return W[:,0]
res=[]
for n in range(3,9):
    dims=[1]+[min(2,2**min(i,n-i)) for i in range(1,n)]+[1]; P=sum(2*dims[i]*2*dims[i+1] for i in range(n))
    x=np.random.default_rng(n).normal(size=P); f=lambda y: mps_state(y,n,2,dims)
    v0=f(x); J=[]
    for j in range(P):
        e=np.zeros(P); e[j]=1e-6; d=(f(x+e)-f(x-e))/2e-6; J.append(np.concatenate([d.real,d.imag]))
    rk=np.linalg.matrix_rank(np.array(J).T,tol=1e-6)
    res.append((n,rk-2))          # subtract norm and global phase (vector -> ray)
ok=all(d==8*n-10 for n,d in res if n>=4)
rep("MPS chi=2 ray dimension (paper: 8n-10)",ok,"; ".join(f"n={n}: {d}" for n,d in res))
thr=[n for n,d in res if 2**n-1>2*d]
rep("MPS threshold (paper: n>=7)",min(thr)==7,f"first n with N-1 > 2d: {min(thr)}")
rep("Product threshold (paper: n>=5)",min(n for n in range(2,12) if 2**n-1>4*n)==5,"first n with N-1>4n is 5")
# ---- chaos check: level-spacing ratio for the generic 2-local chains used
for n in (8,10):
    rs=[]
    for sd in (1,2,3):
        rng=np.random.default_rng(sd); H=generic_complex(n,rng); E=np.linalg.eigvalsh(H)
        m=len(E); E=E[m//4:3*m//4]; s=np.diff(E); r=np.minimum(s[1:],s[:-1])/np.maximum(s[1:],s[:-1]); rs.append(r.mean())
    rep(f"Chaos check n={n} (GUE 0.600, GOE 0.536, Poisson 0.386)",np.mean(rs)>0.56,f"mean level-spacing ratio <r> = {np.mean(rs):.3f}")
open('log_part2.txt','w').write("\n".join(out)+"\n")
