import numpy as np
from scipy.stats import ks_2samp
from mereo import *
n=3; N=8
m = Model(n, seed=1)
psi0 = random_product(n, np.random.default_rng(5)); c0 = m.to_energy(psi0)
omega = 0.05
Ecomm = np.round(m.E/omega)*omega                 # commensurable version (period 2pi/omega ~ 126)
T = 2*np.pi/omega
print("max |E - Ecomm| = %.3f ; period T* = %.1f"%(np.abs(m.E-Ecomm).max(), T))
Hc = m.Q@np.diag(Ecomm)@m.Q.conj().T
# locality check: weight of Hc outside 1- and 2-local nearest-neighbour Pauli strings
from itertools import product
P=[I2,X,np.array([[0,-1j],[1j,0]]),Z]
def nonlocal_weight(H):
    tot=0; bad=0
    for idx in product(range(4),repeat=3):
        O=reduce(np.kron,[P[i] for i in idx]); c=np.trace(O@H)/N
        w=abs(c)**2; tot+=w
        support=[k for k,i in enumerate(idx) if i]
        if len(support)>2 or (len(support)==2 and support[1]-support[0]>1): bad+=w
    return bad/tot
print("non-local weight: generic H %.2e, commensurable H %.2e"%(nonlocal_weight(m.H), nonlocal_weight(Hc)))

def dist_to_orbit(E, phi, tmax, dt):
    best=np.inf; t0=0.0; chunk=1_000_000
    while t0<tmax:
        t=t0+dt*np.arange(chunk); t=t[t<tmax]
        z=np.exp(-1j*(np.outer(t,E)+phi)).mean(1)
        best=min(best,np.sqrt(max(0,2-2*np.abs(z).max()))); t0+=dt*chunk
    return best
rng=np.random.default_rng(3)
phis=[rng.uniform(0,2*np.pi,N) for _ in range(20)]
dt=0.3/(m.E.max()-m.E.min())
for label,E,tmax in (("generic, t<=1e3",m.E,1e3),("generic, t<=1e5",m.E,1e5),
                     ("commensurable, one period",Ecomm,T),("commensurable, t<=1e5",Ecomm,1e5)):
    d=[dist_to_orbit(E,p,tmax,dt) for p in phis]
    print(f"{label:28s}: median distance of random torus point to history = {np.median(d):.3f} (min {np.min(d):.3f})")

def Sbatch(C, Q):
    Pm=(C@Q.T).reshape(-1,2,2,2); out=0
    for i in range(3):
        A=np.moveaxis(Pm,i+1,1).reshape(len(Pm),2,4); rho=A@A.conj().transpose(0,2,1)
        w=np.clip(np.linalg.eigvalsh(rho),1e-15,1); out+=-(w*np.log2(w)).sum(1)
    return out/3
M=100000; r=np.random.default_rng(9)
St = Sbatch(np.exp(-1j*r.uniform(0,2*np.pi,(M,N)))*c0, m.Q)
Sg = Sbatch(np.exp(-1j*np.outer(r.uniform(-1e6,1e6,M),m.E))*c0, m.Q)
Sc = Sbatch(np.exp(-1j*np.outer(r.uniform(0,T,M),Ecomm))*c0, m.Q)
print("\nEntanglement (avg single-qubit, bits):")
print(f"  torus (all local TPSs): mean {St.mean():.4f}  P(S<0.3) {np.mean(St<.3):.4f}")
print(f"  generic history       : mean {Sg.mean():.4f}  P(S<0.3) {np.mean(Sg<.3):.4f}  KS vs torus p={ks_2samp(St,Sg).pvalue:.3f}")
print(f"  commensurable history : mean {Sc.mean():.4f}  P(S<0.3) {np.mean(Sc<.3):.4f}  KS vs torus p={ks_2samp(St,Sc).pvalue:.1e}")
