import numpy as np, sys
from functools import reduce
exec(open('hunt.py').read().split('for n in [int(x)')[0])
def find(n,model,seed,nb=4,rs=3,it=150):
    r=np.random.default_rng(seed)
    Hs={'chaotic':chaotic,'near-integrable':near_integrable}[model](n,r)
    E,Q=np.linalg.eigh(Hs.toarray()); ph=np.exp(-1j*np.outer(ts,E)); best=(9,None)
    for b in range(nb):
        a0=at_energy(Hs,E,n,r,0.10) if b%2 else np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)])
        q=prod(a0); c=Q.conj().T@q; p=np.abs(c)**2
        def ratio(x):
            c2=Q.conj().T@prod(x); p2=np.abs(c2)**2; DT2=1-np.max(np.abs(ph@(np.conj(c)*c2))**2)
            if DT2<0.1: return 10+(0.1-DT2)*100
            return (1-np.sum(np.sqrt(p*p2)))/(1-np.pi/4*F(1-DT2))
        for _ in range(rs):
            res=minimize(ratio,np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)]),method='L-BFGS-B',options={'maxiter':it})
            if res.fun<best[0]: best=(res.fun,(q,prod(res.x)))
    return best,E,Q,Hs
for n in (6,8):
  for model in ('near-integrable','chaotic'):
    for seed in (1,2):
        (R,(q,q2)),E,Q,Hs=find(n,model,seed)
        c=Q.conj().T@q; c2=Q.conj().T@q2
        T=np.linspace(0,400,40001)        # long time window, both directions via |overlap|
        def maxov(a,b): return max(np.max(np.abs(np.exp(-1j*np.outer(T[i:i+4000],E))@(np.conj(a)*b))) for i in range(0,len(T),4000))
        D_long=np.sqrt(max(0,1-maxov(c,c2)**2))
        Px=reduce(np.kron,[P[0]]*n); cP=Q.conj().T@(Px@q)
        D_flip=np.sqrt(max(0,1-maxov(cP,c2)**2))
        s=np.diff(np.sort(E)); rr=np.minimum(s[1:],s[:-1])/np.maximum(s[1:],s[:-1]); rmean=rr[len(rr)//4:3*len(rr)//4].mean()
        print(f"n={n} {model:15s} seed={seed}: ratio {R:.2e} | distance to ANY time-shift of q (|t|<=400): {D_long:.3f} | to time-shifts of flipped q: {D_flip:.3f} | level-spacing ratio {rmean:.3f}",flush=True)
