import numpy as np, sys, time
from functools import reduce
from scipy.special import hyp2f1
from scipy.optimize import minimize
exec(open('ph2.py').read().rsplit('if __name__',1)[0])
F=lambda x: hyp2f1(-0.5,-0.5,1,x)
I2=np.eye(2); X=np.array([[0,1],[1,0]]); Zm=np.diag([1.,-1])
def tfim(n,r):   # random-coupling transverse-field Ising chain: free-fermion integrable
    op=lambda o,i: reduce(np.kron,[o if k==i else I2 for k in range(n)])
    return sum(r.uniform(0.5,1.5)*op(Zm,i)@op(Zm,i+1) for i in range(n-1))+sum(r.uniform(0.5,1.5)*op(X,i) for i in range(n))
def spin(th,ph): return np.array([np.cos(th/2),np.exp(1j*ph)*np.sin(th/2)])
def prod(a): n=len(a)//2; return reduce(np.kron,[spin(a[2*i],a[2*i+1]) for i in range(n)])
def cold_state(H,n,r):
    f=lambda a: np.vdot(prod(a),H@prod(a)).real
    return prod(minimize(f,r.uniform(0,np.pi,2*n),method='L-BFGS-B').x)
def variants(q_angles,n,r,k):
    out=[]
    for _ in range(k):
        a=q_angles.copy(); s=r.choice(n,r.integers(1,3),replace=False)
        for i in s: a[2*i]=np.arccos(r.uniform(-1,1)); a[2*i+1]=r.uniform(0,2*np.pi)
        out.append(prod(a))
    return out
def run(n,model):
    r=np.random.default_rng(50+n); H=generic_complex(n,r) if model=="chaotic" else tfim(n,r)
    import scipy.sparse as sps
    Hs=sps.csr_matrix(H); t0=time.time(); E,Q=np.linalg.eigh(H); tdiag=time.time()-t0; res={}; nb=int(sys.argv[3]) if len(sys.argv)>3 else 10
    for cls in ("hot","cold"):
        R=[]; efr=[]
        for b in range(nb):
            if cls=="hot": a=np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)])
            else:
                fa=lambda a: np.vdot(prod(a),Hs@prod(a)).real
                a=minimize(fa,r.uniform(0,np.pi,2*n),method='L-BFGS-B').x
            q=prod(a); p=np.abs(Q.conj().T@q)**2; efr.append((np.vdot(q,H@q).real-E[0])/(E[-1]-E[0]))
            for qq in variants(a,n,r,4):
                D2=1-abs(np.vdot(q,qq))**2
                if D2<0.05: continue
                pp=np.abs(Q.conj().T@qq)**2
                R.append((1-np.sum(np.sqrt(p*pp)))/(1-np.pi/4*F(1-D2)))
        res[cls]=(np.mean(R),np.min(R),np.mean(efr),len(R))
    print(f"n={n:2d} {model:10s} (diag {tdiag:4.0f}s) | hot: mean {res['hot'][0]:.2f}, min {res['hot'][1]:.2f} [energy {res['hot'][2]:.2f}] | cold: mean {res['cold'][0]:.2f}, min {res['cold'][1]:.2f} [energy {res['cold'][2]:.2f}]",flush=True)
for n in [int(x) for x in sys.argv[1].split(',')]:
    for model in sys.argv[2].split(','): run(n,model)
