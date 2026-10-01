import numpy as np, sys, json
from scipy.optimize import minimize
exec(open('ph_test.py').read().split("def run(")[0])

def make_obj(Q, c0, n):
    N=len(c0)
    def fg(phi):
        a=np.exp(-1j*phi)*c0; psi=Q@a; t=psi.reshape([2]*n); L=0; g=np.zeros(N)
        for i in range(n):
            m=np.moveaxis(t,i,0).reshape(2,-1); rho=m@m.conj().T
            L+=1-np.real(np.vdot(rho,rho))
            gi=np.moveaxis((rho@m).reshape([2]+[2]*(n-1)),0,i).reshape(-1)   # (rho_i ⊗ I) psi
            h=Q.conj().T@gi
            g+= -4*np.real(-1j*a*np.conj(h))
        return L/n, g/n
    return fg

def scan(n, seed, thetas, starts, out):
    rng=np.random.default_rng(seed); N=2**n
    H=generic_complex(n,rng); E,Q=np.linalg.eigh(H)
    omega=0.05; Ec=np.round(E/omega)*omega; T=2*np.pi/omega; dt=0.05/(Ec.max()-Ec.min())
    p=random_product(n,rng); r=haar_state(N,rng); r-=np.vdot(p,r)*p; r/=np.linalg.norm(r)
    for th in thetas:
        psi0=np.cos(th)*p+np.sin(th)*r; c0=Q.conj().T@psi0; pw=np.abs(c0)**2
        fg=make_obj(Q,c0,n)
        res=[minimize(fg,x0,jac=True,method='BFGS',options={'gtol':1e-10}) for x0 in
             [np.zeros(N)]+[rng.uniform(0,2*np.pi,N) for _ in range(starts)]]
        true=res[0]; best=min(res,key=lambda z:z.fun)
        Sr=lambda x: S(Q@(np.exp(-1j*x)*c0),n)
        Fb=fid_to_loop(best.x,pw,Ec,T,dt)
        rec=dict(n=n,seed=seed,theta=th,S0=S(psi0,n),S_true=Sr(true.x),S_best=Sr(best.x),F_best=Fb,
                 on_history=bool(Fb>=0.99 or best.fun>=true.fun*(1-1e-6)))
        out.write(json.dumps(rec)+"\n"); out.flush()

if __name__=="__main__":
    # gradient check
    rng=np.random.default_rng(0); n=3; H=generic_complex(n,rng); E,Q=np.linalg.eigh(H); c0=Q.conj().T@haar_state(8,rng)
    fg=make_obj(Q,c0,n); x=rng.uniform(0,6,8); L,g=fg(x); eps=1e-6
    gn=np.array([(fg(x+eps*np.eye(8)[k])[0]-fg(x-eps*np.eye(8)[k])[0])/(2*eps) for k in range(8)])
    print("grad check max err:", np.abs(g-gn).max()); sys.stdout.flush()
    n=int(sys.argv[1]); seeds=[int(s) for s in sys.argv[2].split(',')]; starts=int(sys.argv[3])
    thetas=[0.0,0.025,0.05,0.1,0.15,0.2,0.3]
    with open(f"ph_n{n}.jsonl","a") as out:
        for sd in seeds: scan(n,sd,thetas,starts,out)
    print("done")
