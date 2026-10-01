"""Low-energy worst-case study. usage: python3 adv2.py n seed target_energy_fraction nbases restarts iters [site]
Base states: distinct product states with <H> at the target fraction of the spectrum, reached from random starts."""
import numpy as np, sys
exec(open('adv.py').read().split('if __name__')[0])
from scipy.optimize import minimize
def at_energy(Hs,E,n,r,frac):
    Et=E[0]+frac*(E[-1]-E[0])
    for _ in range(20):
        a=minimize(lambda a:(np.vdot(prod(a),Hs@prod(a)).real-Et)**2,np.concatenate([[np.arccos(r.uniform(-1,1)),r.uniform(0,6.28)] for _ in range(n)]),method='L-BFGS-B').x
        if abs(np.vdot(prod(a),Hs@prod(a)).real-Et)<1e-3*(E[-1]-E[0]): return a
    return None
def worst_for(a0,n,Q,r,rs,it,site):
    q=prod(a0); p=np.abs(Q.conj().T@q)**2; m=2*site
    def ratio(x):
        a=a0.copy(); a[m:m+4]=x; qq=prod(a); D2=1-abs(np.vdot(q,qq))**2
        if D2<0.1: return 10+(0.1-D2)*100
        pp=np.abs(Q.conj().T@qq)**2; return (1-np.sum(np.sqrt(p*pp)))/(1-np.pi/4*F(1-D2))
    return min(minimize(ratio,a0[m:m+4]+r.normal(size=4),method='Nelder-Mead',options={'maxiter':it}).fun for _ in range(rs))
if __name__=="__main__":
    n,seed=int(sys.argv[1]),int(sys.argv[2]); frac=float(sys.argv[3]); nb,rs,it=map(int,sys.argv[4:7]); site=int(sys.argv[7]) if len(sys.argv)>7 else n//2-1
    r=np.random.default_rng(seed); Hs=chaotic(n,r); E,Q=np.linalg.eigh(Hs.toarray()); w=[]
    for b in range(nb):
        a0=at_energy(Hs,E,n,r,frac)
        if a0 is not None: w.append(worst_for(a0,n,Q,r,rs,it,site))
    print(f"n={n:2d} seed={seed} energy {frac:.2f}: worst ratios {np.round(w,2)}  min {min(w):.2f}",flush=True)
