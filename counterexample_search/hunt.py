"""Counterexample hunt: for each Hamiltonian, find the product-state pair (q, q') with the smallest separation ratio
Hel^2 / [1-(pi/4)F(1-D_T^2)], partner free on ALL sites, D_T = distance modulo time shifts |t|<=0.6, D_T^2 >= 0.1.
Then test whether the unitary V in Gamma relating them would change the division (operator entanglement of V across the middle cut)."""
import numpy as np, sys, time
from scipy.optimize import minimize
from scipy.special import hyp2f1
exec(open('adv2.py').read().rsplit('if __name__',1)[0])
F=lambda x: hyp2f1(-0.5,-0.5,1,x)
def near_integrable(n,r,eps=0.1):
    X,Z=P[0],P[2]
    H=sum(r.uniform(0.5,1.5)*(sop(Z,i,n)@sop(Z,i+1,n)) for i in range(n-1))+sum(r.uniform(0.5,1.5)*sop(X,i,n) for i in range(n))
    return H+eps*(sum(r.uniform(-1,1)*sop(Z,i,n) for i in range(n))+sum(r.uniform(-1,1)*(sop(Z,i,n)@sop(Z,i+2,n)) for i in range(n-2)))
def noninteracting(n,r):
    return sum(r.normal()*sop(A,i,n) for i in range(n) for A in P)
ts=np.linspace(-0.6,0.6,25)
def run(n,model,seed,nb,rs,it):
    r=np.random.default_rng(seed)
    Hs={'chaotic':chaotic,'near-integrable':near_integrable,'non-interacting':noninteracting}[model](n,r)
    E,Q=np.linalg.eigh(Hs.toarray()); ph=np.exp(-1j*np.outer(ts,E)); best=(9,None)
    for b in range(nb):
        a0=at_energy(Hs,E,n,r,0.10) if b%2 else np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)])
        if a0 is None: continue
        c=Q.conj().T@prod(a0); p=np.abs(c)**2
        def ratio(x,full=False):
            c2=Q.conj().T@prod(x); p2=np.abs(c2)**2
            DT2=1-np.max(np.abs(ph@(np.conj(c)*c2))**2)
            if DT2<0.1: return 10+(0.1-DT2)*100
            R=(1-np.sum(np.sqrt(p*p2)))/(1-np.pi/4*F(1-DT2))
            return (R,c2,DT2) if full else R
        for _ in range(rs):
            res=minimize(ratio,np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)]),method='L-BFGS-B',options={'maxiter':it})
            if res.fun<best[0]:
                R,c2,DT2=ratio(res.x,True); best=(R,(c,c2,DT2))
    R,(c,c2,DT2)=best
    # unitary in Gamma that maps the reading q' to (approximately) q: V = Q diag(phase) Q^dag
    V=(Q*np.exp(1j*(np.angle(c)-np.angle(c2))))@Q.conj().T
    d=2**(n//2); T=V.reshape(d,d,d,d).transpose(0,2,1,3).reshape(d*d,d*d)
    sv=np.linalg.svd(T,compute_uv=False); w=(sv**2)/np.sum(sv**2); opent=1-np.sum(w**2)
    return R,DT2,opent
for n in [int(x) for x in sys.argv[1].split(',')]:
    for model in sys.argv[2].split(','):
        for seed in (1,2):
            t=time.time(); R,DT2,oe=run(n,model,seed,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
            print(f"n={n:2d} {model:16s} seed={seed}: worst ratio {R:.3f} (D_T^2={DT2:.2f}); operator entanglement of relating unitary {oe:.3f}  [{time.time()-t:.0f}s]",flush=True)
