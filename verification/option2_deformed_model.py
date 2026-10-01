import numpy as np, sys, time
from scipy.special import hyp2f1
exec(open('ph2.py').read().rsplit('if __name__',1)[0])
F=lambda x: hyp2f1(-0.5,-0.5,1,x)
def spinor(r):
    v=r.normal(size=2)+1j*r.normal(size=2); return v/np.linalg.norm(v)
def local_pairs(n,r,count):
    out=[]
    for _ in range(count):
        sp=[spinor(r) for _ in range(n)]; k=r.integers(1,3); sites=r.choice(n,k,replace=False)
        sp2=[spinor(r) if i in sites else s for i,s in enumerate(sp)]
        from functools import reduce
        q=reduce(np.kron,sp); qq=reduce(np.kron,sp2); out.append((q,qq))
    return out
for n in [int(x) for x in sys.argv[1].split(',')]:
    N=2**n; rng=np.random.default_rng(100+n); Hl=generic_complex(n,rng)
    A=rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)); W=(A+A.conj().T)/np.sqrt(2*N)   # GUE, spectrum ~[-2,2]
    pairs=local_pairs(n,np.random.default_rng(7),40)
    row=f"n={n:2d}:"
    for lam in (0.0,0.5,2.0):
        t0=time.time(); E,Q=np.linalg.eigh(Hl+lam*W); R=[]
        for q,qq in pairs:
            p=np.abs(Q.conj().T@q)**2; pp=np.abs(Q.conj().T@qq)**2
            D2=1-abs(np.vdot(q,qq))**2
            if D2<0.05: continue
            R.append((1-np.sum(np.sqrt(p*pp)))/(1-np.pi/4*F(1-D2)))
        row+=f"  lambda={lam}: separation / random-matrix value = {np.mean(R):.3f} (min {np.min(R):.2f})"
    print(row,flush=True)
