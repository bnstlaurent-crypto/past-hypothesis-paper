"""Adversarial worst-case separation search.
usage: python3 adv.py n model(chaotic|integrable) seed base(hot|cold) nbases restarts iters
For each base product state q, searches over partners q' differing on the two middle sites (D^2 >= 0.1)
for the smallest ratio  Hel^2(q,q') / [1 - (pi/4) F(1-D^2)]  (1 = random-matrix value)."""
import numpy as np, scipy.sparse as sps, sys, time
from functools import reduce
from scipy.special import hyp2f1
from scipy.optimize import minimize
F=lambda x: hyp2f1(-0.5,-0.5,1,x)
P=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1]).astype(complex)]
def sop(A,i,n): return reduce(lambda a,b: sps.kron(a,b,format='csr'),[sps.csr_matrix(A) if k==i else sps.identity(2,format='csr') for k in range(n)])
def chaotic(n,rng):
    H=sps.csr_matrix((2**n,2**n),dtype=complex)
    for i in range(n):
        for A in P: H=H+rng.normal()*sop(A,i,n)
    for i in range(n-1):
        for A in P:
            for B in P: H=H+0.7*rng.normal()*(sop(A,i,n)@sop(B,i+1,n))
    return H
def integrable(n,rng):   # random-coupling transverse-field Ising chain (free fermions)
    X,Z=P[0],P[2]
    return sum(rng.uniform(0.5,1.5)*(sop(Z,i,n)@sop(Z,i+1,n)) for i in range(n-1))+sum(rng.uniform(0.5,1.5)*sop(X,i,n) for i in range(n))
def spin(th,ph): return np.array([np.cos(th/2),np.exp(1j*ph)*np.sin(th/2)])
def prod(a): n=len(a)//2; return reduce(np.kron,[spin(a[2*i],a[2*i+1]) for i in range(n)])
def search(n,Hs,E,Q,r,base,nb,rs,it):
    m=2*(n//2-1); worst=[]; efr=[]
    for b in range(nb):
        if base=="hot": a0=np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)])
        else: a0=minimize(lambda a: np.vdot(prod(a),Hs@prod(a)).real,r.uniform(0,np.pi,2*n),method='L-BFGS-B').x
        q=prod(a0); p=np.abs(Q.conj().T@q)**2; efr.append((np.vdot(q,Hs@q).real-E[0])/(E[-1]-E[0]))
        def ratio(x):
            a=a0.copy(); a[m:m+4]=x; qq=prod(a); D2=1-abs(np.vdot(q,qq))**2
            if D2<0.1: return 10+(0.1-D2)*100
            pp=np.abs(Q.conj().T@qq)**2; return (1-np.sum(np.sqrt(p*pp)))/(1-np.pi/4*F(1-D2))
        worst.append(min(minimize(ratio,a0[m:m+4]+r.normal(size=4),method='Nelder-Mead',options={'maxiter':it}).fun for _ in range(rs)))
    return worst,np.mean(efr)
if __name__=="__main__":
    n=int(sys.argv[1]); model=sys.argv[2]; seed=int(sys.argv[3]); base=sys.argv[4]; nb,rs,it=map(int,sys.argv[5:8])
    r=np.random.default_rng(seed); Hs=(chaotic if model=="chaotic" else integrable)(n,r)
    E,Q=np.linalg.eigh(Hs.toarray()); w,e=search(n,Hs,E,Q,r,base,nb,rs,it)
    print(f"n={n:2d} {model:10s} seed={seed} {base:4s} [energy {e:.2f}]: worst ratios {np.round(w,2)}  min {min(w):.2f}",flush=True)
