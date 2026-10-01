"""
Toy-model test: quantum mereology, time translation, and the Past Hypothesis.
Claims tested (3 qubits unless noted):
  A. H-only selection criteria are exactly flat on the torus of local TPSs {V T0 V^dag : [V,H]=0}.
  B. Time evolution approaches any torus point at rate d ~ t^(-1/(N-1)) (Kronecker density).
  C. Entanglement over random torus TPSs has the same distribution as over random times (Weyl equidistribution).
  D. Choosing the TPS by minimal entanglement of psi_now recovers the low-entropy initial moment,
     uniquely for generic H, but only up to time reversal when H has a local antiunitary symmetry.
  E. With commensurable energies the history is a closed loop: most local TPSs are never visited.
Runtime: several minutes.
"""
# ---- core ----
import numpy as np
from functools import reduce

I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Z = np.diag([1.,-1.]).astype(complex)

def op(single, site, n):
    return reduce(np.kron, [single if k==site else I2 for k in range(n)])

def ising(n, rng):
    """Mixed-field Ising chain with random couplings (breaks all symmetries -> generic spectrum)."""
    N = 2**n; H = np.zeros((N,N),complex)
    for i in range(n-1):
        H += rng.uniform(0.7,1.3)*op(Z,i,n)@op(Z,i+1,n)
    for i in range(n):
        H += rng.uniform(0.6,1.1)*op(X,i,n) + rng.uniform(0.2,0.6)*op(Z,i,n)
    return H

def entropies(psi, n):
    """Von Neumann entropy (bits) of each qubit vs the rest, in the standard TPS."""
    t = psi.reshape([2]*n); out=[]
    for i in range(n):
        m = np.moveaxis(t,i,0).reshape(2,-1)
        rho = m@m.conj().T
        w = np.clip(np.linalg.eigvalsh(rho),1e-15,1)
        out.append(-(w*np.log2(w)).sum())
    return np.array(out)

def S(psi,n): return entropies(psi,n).mean()

def random_product(n, rng):
    vs=[]
    for _ in range(n):
        v = rng.normal(size=2)+1j*rng.normal(size=2); vs.append(v/np.linalg.norm(v))
    return reduce(np.kron, vs)

class Model:
    def __init__(self, n, seed):
        self.n=n; self.rng=np.random.default_rng(seed)
        self.H = ising(n,self.rng)
        self.E, self.Q = np.linalg.eigh(self.H)
    def U_diag(self,t): return np.exp(-1j*self.E*t)          # U(t) in energy basis
    def to_comp(self, c): return self.Q@c                     # energy basis -> computational
    def to_energy(self, psi): return self.Q.conj().T@psi
    # A torus element V = diag(e^{i phi}) in the energy basis commutes with H.
    # Reading state psi in the TPS V T0 V^dag is the same as reading V^dag psi in T0.
    def read(self, psi_energy, phi):
        return self.to_comp(np.exp(-1j*phi)*psi_energy)

# ---- helpers ----
from scipy.optimize import minimize
from scipy.stats import ks_2samp
Y = np.array([[0,-1j],[1j,0]]); P=[X,Y,Z]

def generic_complex(n, rng):
    N=2**n; H=np.zeros((N,N),complex)
    for i in range(n):
        for A in P: H += rng.normal()*op(A,i,n)
    for i in range(n-1):
        for A in P:
            for B in P: H += 0.7*rng.normal()*op(A,i,n)@op(B,i+1,n)
    return H

def lin_entropy(psi,n):
    t=psi.reshape([2]*n); tot=0
    for i in range(n):
        a=np.moveaxis(t,i,0).reshape(2,-1); rho=a@a.conj().T; tot+=1-np.real(np.trace(rho@rho))
    return tot

def test_A(m):
    tau=0.4; Ut=m.Q@np.diag(m.U_diag(tau))@m.Q.conj().T
    def score(W):
        r=np.random.default_rng(123); return np.mean([S(W.conj().T@(Ut@(W@random_product(m.n,r))),m.n) for _ in range(300)])
    rng=np.random.default_rng(7); N=2**m.n
    tor=[score(m.Q@np.diag(np.exp(1j*rng.uniform(0,2*np.pi,N)))@m.Q.conj().T) for _ in range(3)]
    from scipy.stats import unitary_group
    rnd=[score(unitary_group.rvs(N,random_state=j)) for j in range(3)]
    print("A. H-only robustness: T0 %.6f | torus"%score(np.eye(N)), np.round(tor,6), "| random TPS", np.round(rnd,3))

def test_B(n, Tmax, dphase=0.3, chunk=2_000_000, trials=3):
    m=Model(n,seed=1); rng=np.random.default_rng(11); N=2**n; dt=dphase/(m.E.max()-m.E.min()); slopes=[]
    for _ in range(trials):
        phi=rng.uniform(0,2*np.pi,N); best=np.inf; recs=[]; t0=0.0
        while t0<Tmax:
            t=t0+dt*np.arange(chunk); z=np.exp(-1j*(np.outer(t,m.E)+phi)).mean(1)
            d=np.sqrt(np.clip(2-2*np.abs(z),0,None)); run=np.minimum.accumulate(d)
            for i in np.nonzero(run<best)[0]:
                if run[i]<best: best=run[i]; recs.append((t[i],best))
            t0=t[-1]+dt
        r=np.array(recs); r=r[r[:,0]>10]; slopes.append(np.polyfit(np.log(r[:,0]),np.log(r[:,1]),1)[0])
    print(f"B. n={n}: slope {np.mean(slopes):.3f} +/- {np.std(slopes):.3f}, predicted {-1/(N-1):.3f}")

def test_C(m, c0, M=100000):
    N=2**m.n
    def Sbatch(C):
        Pm=(C@m.Q.T).reshape(-1,2,2,2); out=0
        for i in range(3):
            A=np.moveaxis(Pm,i+1,1).reshape(len(Pm),2,4); rho=A@A.conj().transpose(0,2,1)
            w=np.clip(np.linalg.eigvalsh(rho),1e-15,1); out+=-(w*np.log2(w)).sum(1)
        return out/3
    for seed in (1,2,3):
        rng=np.random.default_rng(100+seed)
        St=Sbatch(np.exp(-1j*rng.uniform(0,2*np.pi,(M,N)))*c0)
        Sh=Sbatch(np.exp(-1j*np.outer(rng.uniform(-1e6,1e6,M),m.E))*c0)
        print(f"C. seed {seed}: mean torus {St.mean():.4f} history {Sh.mean():.4f}  KS p={ks_2samp(St,Sh).pvalue:.3f}")

def test_D(m, rng, label, starts=200):
    N=2**m.n; psi0=random_product(m.n,rng); c_now=m.U_diag(41.7)*m.to_energy(psi0)
    a=b=o=0
    for _ in range(starts):
        r=minimize(lambda p: lin_entropy(m.read(c_now,p),m.n), rng.uniform(0,2*np.pi,N), method='BFGS', options={'gtol':1e-12})
        if r.fun<1e-10:
            st=m.read(c_now,r.x)
            if abs(np.vdot(psi0,st))>0.99999: a+=1
            elif abs(np.vdot(psi0.conj(),st))>0.99999: b+=1
            else: o+=1
    print(f"D. {label}: exact product readings -> psi0: {a}, psi0* (time-reverse): {b}, other: {o}")


# ---- E. generic vs commensurable spectrum (run as a separate section) ----
def test_E():
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

if __name__=="__main__":
    m=Model(3,seed=1)
    test_A(m)
    test_B(2,1e5); test_B(3,1e6)
    test_C(m, m.to_energy(random_product(3,np.random.default_rng(5))))
    test_D(Model(3,seed=1), np.random.default_rng(21), "real Ising (time-reversal symmetric)")
    mc=Model(3,seed=1); r=np.random.default_rng(21); mc.H=generic_complex(3,r); mc.E,mc.Q=np.linalg.eigh(mc.H)
    test_D(mc, r, "generic complex 2-local")
    test_E()
