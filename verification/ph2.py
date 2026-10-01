import numpy as np, sys, json, time
from scipy.optimize import minimize
exec(open('ph_scan.py').read().split('if __name__')[0])   # generic_complex, haar_state, make_obj, fid_to_loop, S, random_product ...

def ground_bounce(n, g, rng):
    """Ground state of a pre-bounce local Hamiltonian in the SAME division: -sum ZZ - g sum X - 0.1 sum Z.
       Large g -> nearly a product state; g near 1 -> critical, more entangled (area-law, not random)."""
    N=2**n; Hb=np.zeros((N,N),complex)
    for i in range(n-1): Hb-=op(Z,i,n)@op(Z,i+1,n)
    for i in range(n): Hb-=g*op(X,i,n)+0.1*op(Z,i,n)
    w,v=np.linalg.eigh(Hb); return v[:,0]

def scan(n, seed, kind, params, starts, out, budget):
    rng=np.random.default_rng(seed); N=2**n
    H=generic_complex(n,rng); E,Q=np.linalg.eigh(H)
    omega=0.05; Ec=np.round(E/omega)*omega; T=2*np.pi/omega; dt=0.05/(Ec.max()-Ec.min())
    p=random_product(n,rng); r=haar_state(N,rng); r-=np.vdot(p,r)*p; r/=np.linalg.norm(r)
    srng=np.random.default_rng(1000+seed)
    for par in params:
        t0=time.time()
        psi0 = np.cos(par)*p+np.sin(par)*r if kind=="perturbed" else ground_bounce(n,par,rng)
        c0=Q.conj().T@psi0; pw=np.abs(c0)**2; fg=make_obj(Q,c0,n)
        true=minimize(fg,np.zeros(N),jac=True,method='BFGS',options={'gtol':1e-10})
        vals=[];best=true
        for k in range(starts):
            res=minimize(fg,srng.uniform(0,2*np.pi,N),jac=True,method='BFGS',options={'gtol':1e-10})
            vals.append(res.fun)
            if res.fun<best.fun: best=res
            if time.time()-t0>budget: break
        vals=np.array(vals)
        Sr=lambda x: S(Q@(np.exp(-1j*x)*c0),n)
        Fb=fid_to_loop(best.x,pw,Ec,T,dt)
        rec=dict(n=n,seed=seed,kind=kind,par=par,S0=S(psi0,n),S_true=Sr(true.x),S_best=Sr(best.x),F_best=Fb,
                 on_history=bool(Fb>=0.99 or best.fun>=true.fun*(1-1e-6)),
                 starts=len(vals),frac_true_basin=float(np.mean(np.abs(vals-true.fun)<1e-6*max(true.fun,1e-12)+1e-12)),
                 secs=round(time.time()-t0,1))
        out.write(json.dumps(rec)+"\n"); out.flush(); print(rec); sys.stdout.flush()

if __name__=="__main__":
    n=int(sys.argv[1]); seeds=[int(s) for s in sys.argv[2].split(',')]; kind=sys.argv[3]
    params=[float(x) for x in sys.argv[4].split(',')]; starts=int(sys.argv[5]); budget=float(sys.argv[6])
    with open(f"ph2_{kind}_n{n}.jsonl","a") as out:
        for sd in seeds: scan(n,sd,kind,params,starts,out,budget)
