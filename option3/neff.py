import numpy as np, sys
exec(open('adv2.py').read().rsplit('if __name__',1)[0])
from scipy.stats import spearmanr
rows=[]
for n in (6,8,10):
    for seed in (1,2,3):
        cr=np.random.default_rng(seed)      # replay coupling draws to get bond strengths
        for i in range(n):
            for A in P: cr.normal()
        bonds=[np.sqrt(sum((0.7*cr.normal())**2 for _ in range(9))) for i in range(n-1)]
        r=np.random.default_rng(seed); Hs=chaotic(n,r); E,Q=np.linalg.eigh(Hs.toarray()); m=n//2-1
        iso=min(bonds[m-1],bonds[m+1]) if m+1<n-1 else bonds[m-1]
        for frac in (0.10,0.25,0.50):
            for b in range(3):
                a0=at_energy(Hs,E,n,r,frac)
                if a0 is None: continue
                p=np.abs(Q.conj().T@prod(a0))**2
                rows.append((n,seed,frac,np.log2(1/np.sum(p**2)),iso,worst_for(a0,n,Q,r,3,300,m)))
        print(f"n={n} seed={seed} done",flush=True)
R=np.array(rows); np.save('neff_rows.npy',R)
rho1,p1=spearmanr(R[:,3],R[:,5]); rho2,p2=spearmanr(R[:,4],R[:,5]); rhon,pn=spearmanr(R[:,0],R[:,5])
print(f"\n{len(R)} base states. Spearman correlation of worst ratio with: log2 N_eff {rho1:.2f} (p={p1:.1e}); isolating-bond strength {rho2:.2f} (p={p2:.1e}); qubit number {rhon:.2f} (p={pn:.1e})")
# does N_eff explain the size and energy dependence? regress ratio on log2 N_eff, then check residual trends
A=np.c_[np.ones(len(R)),R[:,3]]; coef,*_=np.linalg.lstsq(A,R[:,5],rcond=None); res=R[:,5]-A@coef
print(f"fit: ratio = {coef[0]:.2f} + {coef[1]:.3f} log2 N_eff;  residual correlation with n: {spearmanr(R[:,0],res)[0]:.2f}, with energy: {spearmanr(R[:,2],res)[0]:.2f}, with bond: {spearmanr(R[:,4],res)[0]:.2f}")
for frac in (0.10,0.25,0.50):
    for n in (6,8,10):
        s=R[(R[:,0]==n)&(R[:,2]==frac)]; print(f"  energy {frac:.2f}, n={n:2d}: mean log2 N_eff {s[:,3].mean():5.2f}, median worst ratio {np.median(s[:,5]):.2f}")
