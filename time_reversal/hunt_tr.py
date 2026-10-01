"""All-sites counterexample hunt in a REAL-coupled (time-reversal-invariant) chaotic chain, quotienting by
short time shifts of q AND all time shifts |t|<=100 of conj(q) (the time-reversed history)."""
import numpy as np, sys, time
from scipy.optimize import minimize
from scipy.special import hyp2f1
exec(open('adv2.py').read().rsplit('if __name__',1)[0])
F=lambda x: hyp2f1(-0.5,-0.5,1,x)
X,Y,Z=P
def real_chaotic(n,r):
    H=sum(r.normal()*sop(A,i,n) for i in range(n) for A in (X,Z))
    pairs=[(X,X),(Z,Z),(X,Z),(Z,X),(Y,Y)]                      # all real two-site terms
    return H+sum(0.7*r.normal()*(sop(A,i,n)@sop(B,i+1,n)) for i in range(n-1) for A,B in pairs)
ts=np.linspace(-0.6,0.6,25); tl=np.linspace(-100,100,4001)
for n in [int(x) for x in sys.argv[1].split(',')]:
    for seed in (1,2):
        r=np.random.default_rng(seed); Hs=real_chaotic(n,r); Hd=Hs.toarray(); assert np.allclose(Hd.imag,0)
        E,Q=np.linalg.eigh(Hd); s=np.diff(E); rr=np.minimum(s[1:],s[:-1])/np.maximum(s[1:],s[:-1]); rmean=rr[len(rr)//4:3*len(rr)//4].mean()
        phs=np.exp(-1j*np.outer(ts,E)); phl=np.exp(-1j*np.outer(tl,E)); best=9; t0=time.time()
        for b in range(int(sys.argv[2])):
            a0=at_energy(Hs,E,n,r,0.10) if b%2 else np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)])
            q=prod(a0); c=Q.T@q; cb=Q.T@np.conj(q); p=np.abs(c)**2
            def ratio(x):
                c2=Q.T@prod(x); p2=np.abs(c2)**2
                D2=min(1-np.max(np.abs(phs@(np.conj(c)*c2))**2), 1-np.max(np.abs(phl@(np.conj(cb)*c2))**2))
                if D2<0.1: return 10+(0.1-D2)*100
                return (1-np.sum(np.sqrt(p*p2)))/(1-np.pi/4*F(1-D2))
            for _ in range(int(sys.argv[3])):
                res=minimize(ratio,np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)]),method='L-BFGS-B',options={'maxiter':150})
                best=min(best,res.fun)
        print(f"n={n} real chaotic seed={seed} (level-spacing ratio {rmean:.3f}, GOE 0.536): worst ratio excluding time shifts and time reversal = {best:.3f}  [{time.time()-t0:.0f}s]",flush=True)
