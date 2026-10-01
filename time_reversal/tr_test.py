"""Test: for random REAL orthogonal eigenbases, are complex conjugates the only product-state twins once N-1 > 4n?
Joint search over pairs (q, q'); distance taken modulo phase AND conjugation; complex Haar bases as control."""
import numpy as np, sys, time
from scipy.optimize import minimize
from scipy.stats import ortho_group, unitary_group
exec(open('adv.py').read().split('if __name__')[0])
def dmod(a,b,conj=True):
    o=abs(np.vdot(a,b)); 
    if conj: o=max(o,abs(np.vdot(np.conj(a),b)))
    return 1-o**2
def worst(Q,n,rng,starts,conj):
    best=9
    for _ in range(starts):
        def f(x):
            q=prod(x[:2*n]); q2=prod(x[2*n:]); D2=dmod(q,q2,conj)
            if D2<0.1: return 10+(0.1-D2)*100
            p=np.abs(Q.T.conj()@q)**2; p2=np.abs(Q.T.conj()@q2)**2
            return (1-np.sum(np.sqrt(p*p2)))/D2
        r=minimize(f,rng.uniform(0,np.pi,4*n)*np.tile([1,2],2*n),method='L-BFGS-B',options={'maxiter':300})
        best=min(best,r.fun)
    return best
for n in (3,4,5,6,7):
    rng=np.random.default_rng(n); N=2**n; st=int(sys.argv[1])
    t=time.time()
    vals_real=[worst(ortho_group.rvs(N,random_state=10*n+s),n,rng,st,True) for s in range(2)]
    vals_cplx=[worst(unitary_group.rvs(N,random_state=10*n+s),n,rng,st,False) for s in range(2)]
    print(f"n={n} (N-1={N-1}, 4n={4*n}): min Hel^2/D^2 real bases (mod conjugation) {np.round(vals_real,4)} | complex bases {np.round(vals_cplx,4)}  [{time.time()-t:.0f}s]",flush=True)
