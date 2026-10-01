import numpy as np
from scipy.special import hyp2f1
from scipy.stats import unitary_group
F=lambda x: hyp2f1(-0.5,-0.5,1,x)
rng=np.random.default_rng(0)
print("Claim 1 (reviewer): L=2 in unnormalised HS norm makes the concentration bound fail.")
print("Test: if the bound e^{-N t^2/48} is right, fluctuations of g must shrink like 1/sqrt(N).")
for n in (4,6,8,10):
    N=2**n; m=np.zeros(N,complex); m[0]=1; mp=np.zeros(N,complex); mp[0]=0.6; mp[1]=0.8
    S=400 if N<=256 else 120
    g=np.array([np.sum(np.abs(U.conj().T@m)*np.abs(U.conj().T@mp)) for U in (unitary_group.rvs(N,random_state=rng) for _ in range(S))])
    t=0.05; emp=np.mean(g-g.mean()>=t)
    print(f"  N={N:5d}: mean g={g.mean():.4f} (exact pi/4 F = {np.pi/4*F(0.36):.4f}), std={g.std():.4f}, std*sqrt(N)={g.std()*np.sqrt(N):.2f}, "
          f"P[g-Eg>=0.05] empirical {emp:.3f} vs bound {min(1,np.exp(-N*t*t/48)):.3f}")
print("\nClaim 2 (reviewer): the pi/4 bound 'reverses sign'. Test G(x)=(pi/4)(1-F(x))+(1-pi/4)x >= 0 on [0,1]:")
x=np.linspace(0,1,100001); G=np.pi/4*(1-F(x))+(1-np.pi/4)*x
print(f"  min G = {G.min():.2e} at x={x[G.argmin()]:.3f}; G(0)={G[0]:.1e}, G(1)={G[-1]:.1e}; G concave: {np.all(np.diff(G,2)<=1e-12)}")
