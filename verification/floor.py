import numpy as np, sys, time
from scipy.optimize import minimize
from scipy.stats import unitary_group
exec(open('ph2.py').read().rsplit('if __name__',1)[0])
def floor(n, seed, kind, starts, budget):
    rng=np.random.default_rng(seed); N=2**n
    H=generic_complex(n,rng); E,Q=np.linalg.eigh(H)
    if kind=="haar": Q=unitary_group.rvs(N,random_state=seed)
    Ec=np.round(E/0.05)*0.05; T=2*np.pi/0.05; dt=0.05/(Ec.max()-Ec.min())
    p=random_product(n,rng); c0=Q.conj().T@p; pw=np.abs(c0)**2; fg=make_obj(Q,c0,n)
    srng=np.random.default_rng(900+seed); best=np.inf; t0=time.time(); k=0
    while k<starts and time.time()-t0<budget:
        r=minimize(fg,srng.uniform(0,2*np.pi,N),jac=True,method='BFGS',options={'gtol':1e-10}); k+=1
        if r.fun<1e-8: continue                                   # reached the true bounce (exact product)
        if fid_to_loop(r.x,pw,Ec,T,dt)>=0.99: continue           # another moment of the history
        best=min(best,S(Q@(np.exp(-1j*r.x)*c0),n))
    return best,k
for n in [int(x) for x in sys.argv[1].split(',')]:
    for kind in ("local","haar"):
        vals=[floor(n,sd,kind,int(sys.argv[2]),float(sys.argv[3])) for sd in (1,2)]
        print(f"n={n} {kind:5s}: rival floor (lowest per-site entropy of any rival found) = {[round(v[0],3) for v in vals]} ({[v[1] for v in vals]} starts)",flush=True)
