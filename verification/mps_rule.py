import numpy as np, sys, json, time
from scipy.optimize import minimize
exec(open('ph2.py').read().rsplit('if __name__',1)[0])

def make_mps_obj(Q, c0, n, chi):
    """Total discarded Schmidt weight across all chain cuts at bond dimension chi (proxy for distance to the MPS manifold),
       as a function of the torus phases, with analytic gradient."""
    Qh=Q.conj().T
    def fg(phi):
        a=np.exp(-1j*phi)*c0; psi=Q@a; L=0.0; acc=np.zeros_like(psi)
        for c in range(1,n):
            M=psi.reshape(2**c,-1); U,s,Vh=np.linalg.svd(M,full_matrices=False)
            L+=1-np.sum(s[:chi]**2)
            acc+=((U[:,:chi]*s[:chi])@Vh[:chi]).reshape(-1)          # truncated state at this cut
        g=-2*np.real(np.conj(Qh@acc)*(-1j*a))
        return L, g
    return fg

def scan(n, seed, gs, chi, starts, budget, out):
    rng=np.random.default_rng(seed); N=2**n
    H=generic_complex(n,rng); E,Q=np.linalg.eigh(H)
    omega=0.05; Ec=np.round(E/omega)*omega; T=2*np.pi/omega; dt=0.05/(Ec.max()-Ec.min())
    p=random_product(n,rng); r=haar_state(N,rng)
    srng=np.random.default_rng(5000+seed)
    for g in gs:
        t0=time.time(); psi0=ground_bounce(n,g,rng); c0=Q.conj().T@psi0; pw=np.abs(c0)**2
        fg=make_mps_obj(Q,c0,n,chi)
        true=minimize(fg,np.zeros(N),jac=True,method='BFGS',options={'gtol':1e-10}); best=true; vals=[]
        while len(vals)<starts and time.time()-t0<budget:
            res=minimize(fg,srng.uniform(0,2*np.pi,N),jac=True,method='BFGS',options={'gtol':1e-10}); vals.append(res.fun)
            if res.fun<best.fun: best=res
        Fb=fid_to_loop(best.x,pw,Ec,T,dt); L0=fg(np.zeros(N))[0]
        rec=dict(n=n,seed=seed,g=g,chi=chi,L0=L0,L_true=true.fun,L_best=best.fun,F_best=float(Fb),
                 on_history=bool(Fb>=0.99 or best.fun>=true.fun*(1-1e-6)),starts=len(vals),
                 frac_true=float(np.mean(np.abs(np.array(vals)-true.fun)<1e-6*max(true.fun,1e-12)+1e-12)) if vals else None,
                 secs=round(time.time()-t0,1))
        out.write(json.dumps(rec)+"\n"); out.flush()
        print(f"n={n} seed={seed} g={g}: discarded weight at bounce {L0:.2e}; true-basin min {true.fun:.2e}; best found {best.fun:.2e}; "
              f"{'PICKS HISTORY' if rec['on_history'] else 'RIVAL'} ({len(vals)} starts, {rec['secs']}s)",flush=True)

if __name__=="__main__":
    # gradient check
    rng=np.random.default_rng(0); n=4; H=generic_complex(n,rng); E,Q=np.linalg.eigh(H); c0=Q.conj().T@haar_state(16,rng)
    fg=make_mps_obj(Q,c0,n,2); x=rng.uniform(0,6,16); L,g=fg(x); eps=1e-6
    gn=np.array([(fg(x+eps*np.eye(16)[k])[0]-fg(x-eps*np.eye(16)[k])[0])/(2*eps) for k in range(16)])
    print("grad check max err:",np.abs(g-gn).max(),flush=True)
    n=int(sys.argv[1]); seeds=[int(s) for s in sys.argv[2].split(',')]; gs=[float(x) for x in sys.argv[3].split(',')]
    chi=int(sys.argv[4]); starts=int(sys.argv[5]); budget=float(sys.argv[6])
    with open(f"mps_n{n}_chi{chi}.jsonl","a") as out:
        for sd in seeds: scan(n,sd,gs,chi,starts,budget,out)
