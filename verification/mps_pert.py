import numpy as np, sys, json, time
exec(open('mps_rule.py').read().rsplit('if __name__',1)[0])
def run(n, seeds, thetas, chi, starts, budget):
    for seed in seeds:
        rng=np.random.default_rng(seed); N=2**n
        H=generic_complex(n,rng); E,Q=np.linalg.eigh(H)
        omega=0.05; Ec=np.round(E/omega)*omega; T=2*np.pi/omega; dt=0.05/(Ec.max()-Ec.min())
        p=random_product(n,rng); r=haar_state(N,rng)
        gs=ground_bounce(n,1.2,rng); r-=np.vdot(gs,r)*gs; r/=np.linalg.norm(r); srng=np.random.default_rng(7000+seed)
        res_line=[]
        for th in thetas:
            psi0=np.cos(th)*gs+np.sin(th)*r; c0=Q.conj().T@psi0; pw=np.abs(c0)**2; fg=make_mps_obj(Q,c0,n,chi)
            true=minimize(fg,np.zeros(N),jac=True,method='BFGS',options={'gtol':1e-10}); best=true; k=0; t0=time.time()
            while k<starts and time.time()-t0<budget:
                s=minimize(fg,srng.uniform(0,2*np.pi,N),jac=True,method='BFGS',options={'gtol':1e-10}); k+=1
                if s.fun<best.fun: best=s
            ok=fid_to_loop(best.x,pw,Ec,T,dt)>=0.99 or best.fun>=true.fun*(1-1e-6)
            res_line.append(f"θ={th}(L0={fg(np.zeros(N))[0]:.2f}):{'ok' if ok else 'RIVAL'}")
        print(f"n={n} seed={seed}: "+"  ".join(res_line),flush=True)
n=int(sys.argv[1]); run(n,[int(s) for s in sys.argv[2].split(',')],[float(x) for x in sys.argv[3].split(',')],2,int(sys.argv[4]),float(sys.argv[5]))
