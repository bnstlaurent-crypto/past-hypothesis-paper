"""
Past Hypothesis as a division-selection rule (companion to mereology_toy_model.py).

Rule: among all divisions (tensor product structures) in which H is equally local -- the torus
{V T V^dag : [V,H]=0} -- choose the one in which some moment of the history is least entangled.
With a commensurable spectrum, check whether the choice lies on the actual history (only a "when"
ambiguity) or is a rival world the history never visits.

Bounce models:  perturbed : psi0 = cos(par)*product + sin(par)*random      (par = theta)
                ground    : ground state of -sum ZZ - par*sum X - 0.1 sum Z   (par = field g)
Usage:
  python3 past_hypothesis_scan.py <n> <seeds e.g. 1,2,3> <perturbed|ground> <params e.g. 0,0.1,0.2> <starts> <seconds per point>
  python3 past_hypothesis_scan.py analyze
Results append to ph2_<kind>_n<n>.jsonl. Runtime: seconds per point at 3-5 qubits, ~10 s at 7.
"""
import json, sys, time, collections
from scipy.optimize import minimize
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

Y = np.array([[0,-1j],[1j,0]]); P=[X,Y,Z]
def generic_complex(n, rng):
    N=2**n; H=np.zeros((N,N),complex)
    for i in range(n):
        for A in P: H += rng.normal()*op(A,i,n)
    for i in range(n-1):
        for A in P:
            for B in P: H += 0.7*rng.normal()*op(A,i,n)@op(B,i+1,n)
    return H
def haar_state(N, rng):
    v=rng.normal(size=N)+1j*rng.normal(size=N); return v/np.linalg.norm(v)
def lin_ent(psi,n):
    t=psi.reshape([2]*n); tot=0
    for i in range(n):
        a=np.moveaxis(t,i,0).reshape(2,-1); rho=a@a.conj().T; tot+=1-np.real(np.vdot(rho,rho))
    return tot/n
def dist(phiA, phiB):            # distance between torus elements mod global phase
    return np.sqrt(max(0,2-2*abs(np.mean(np.exp(1j*(phiA-phiB))))))
def fid_to_loop(phi, p, Ec, T, dt):
    """max over the history of |<psi(t)|reading>|: 1 = the reading is a moment of the actual history"""
    t=np.arange(0,T,dt); z=(np.exp(1j*(phi[None,:]+np.outer(t,Ec)))*p).sum(1)
    return np.abs(z).max()

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


def analyze():
    import glob
    for kind in ("perturbed","ground"):
        print(f"\n{kind.upper()} BOUNCE")
        for f in sorted(glob.glob(f"ph2_{kind}_n*.jsonl")):
            rs=[json.loads(l) for l in open(f)]; n=rs[0]['n']
            by=collections.defaultdict(list)
            for r in rs: by[r['par']].append(r)
            cells=[f"par={p}: {sum(r['on_history'] for r in by[p])}/{len(by[p])} (S0={np.mean([r['S0'] for r in by[p]]):.2f})" for p in sorted(by)]
            print(f" n={n}: " + "; ".join(cells))

if __name__=="__main__":
    if sys.argv[1]=="analyze": analyze()
    else:
        n=int(sys.argv[1]); seeds=[int(s) for s in sys.argv[2].split(',')]; kind=sys.argv[3]
        params=[float(x) for x in sys.argv[4].split(',')]; starts=int(sys.argv[5]); budget=float(sys.argv[6])
        with open(f"ph2_{kind}_n{n}.jsonl","a") as out:
            for sd in seeds: scan(n,sd,kind,params,starts,out,budget)
