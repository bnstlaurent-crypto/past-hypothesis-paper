import numpy as np
from functools import reduce
rng=np.random.default_rng(3)
def iso(r,c):
    Z=rng.normal(size=(r,c))+1j*rng.normal(size=(r,c)); return np.linalg.qr(Z)[0]
print("Claim: MPS telescoping needs an extra factor sqrt(2): ||psi-psi~|| <= sqrt2 * sum ||V_i - V~_i||.")
worst=0
for trial in range(3000):
    n=rng.integers(3,8); chi=2; dims=[1]+[min(chi,2**min(i,n-i)) for i in range(1,n)]+[1]
    eps=10**rng.uniform(-3,0.3); V=[];Vt=[]
    for i in range(n):
        a=iso(2*dims[i],dims[i+1]); b=a+eps*(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))
        V.append(a); Vt.append(np.linalg.qr(b)[0])       # net points are isometries
    st=lambda L: reduce(lambda W,Vi: np.kron(W,np.eye(2))@Vi, L, np.eye(1))[:,0]
    lhs=np.linalg.norm(st(V)-st(Vt)); rhs=sum(np.linalg.norm(a-b,2) for a,b in zip(V,Vt))
    worst=max(worst,lhs/rhs)
print(f"  max of ||psi-psi~|| / sum||V_i-V~_i||_op over 3000 random chains (perturbations 0.001 to 2): {worst:.3f}  (the paper's bound needs <= 1; no sqrt2 needed if so)")
print("Claim: D(prod q_i, prod q~_i) <= sum D_i is 'incorrect'.")
bad=0; tight=[]
def rs(): v=rng.normal(size=2)+1j*rng.normal(size=2); return v/np.linalg.norm(v)
for _ in range(20000):
    n=rng.integers(2,8); q=[rs() for _ in range(n)]; qt=[]
    for x in q:
        w=x+10**rng.uniform(-2,0.5)*rs(); qt.append(w/np.linalg.norm(w))
    d=[np.sqrt(max(0,1-abs(np.vdot(a,b))**2)) for a,b in zip(q,qt)]
    D=np.sqrt(max(0,1-np.prod([abs(np.vdot(a,b))**2 for a,b in zip(q,qt)])))
    bad+= D>sum(d)+1e-12; tight.append(D<=np.sqrt(sum(x*x for x in d))+1e-12)
print(f"  violations of D <= sum D_i: {bad} of 20000; the sharper D <= sqrt(sum D_i^2) also holds in all cases: {all(tight)}")
