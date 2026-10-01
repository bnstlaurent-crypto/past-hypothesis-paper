"""
ETH check for local Hamiltonians: how far apart are the energy profiles of two product states q, q' = L q?
Compares the exact Hellinger distance with (a) the Haar formula 1-(pi/4) 2F1(-1/2,-1/2;1;|<q|q'>|^2) and
(b) an ETH random-amplitude model (Gaussian amplitudes with smooth energy envelopes and smooth correlation),
and reports the "coarse" Hellinger distance of profiles smoothed at energy resolution 0.5.
Usage: python3 eth_check.py <n qubits> <seed> <number of pairs>
"""
import time
from scipy.special import hyp2f1
from functools import reduce
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


F=lambda x: hyp2f1(-0.5,-0.5,1,np.clip(x,0,1))
def rot(axis, th): return np.cos(th)*np.eye(2)-1j*np.sin(th)*(axis[0]*X+axis[1]*Y+axis[2]*Z)
n=int(sys.argv[1]); seed=int(sys.argv[2]); trials=int(sys.argv[3])
rng=np.random.default_rng(seed); t0=time.time()
H=generic_complex(n,rng); E,Q=np.linalg.eigh(H); Qh=Q.conj().T
w=0.5; K=np.exp(-((E[:,None]-E[None,:])/w)**2/2); Ks=K/K.sum(1,keepdims=True)
grid=np.linspace(E.min(),E.max(),400); Kg=np.exp(-((grid[:,None]-E[None,:])/w)**2/2)
res=[]
for t in range(trials):
    q=random_product(n,rng); a=Qh@q
    m=[1,1,2,4][t%4]; th=[np.pi/2,np.pi/5,np.pi/2,np.pi/2][t%4]
    ops=[np.eye(2)]*n
    for i in range(n//2,n//2+m):
        ax=rng.normal(size=3); ax/=np.linalg.norm(ax); ops[i]=rot(ax,th)
    qp=reduce(np.kron,ops)@q; b=Qh@qp
    s2=abs(np.vdot(q,qp))**2
    act=1-np.sum(np.abs(a)*np.abs(b))
    haar=1-(np.pi/4)*F(s2)                                             # infinite-temperature (Haar) formula
    pa=Ks@np.abs(a)**2; pb=Ks@np.abs(b)**2; cab=Ks@(np.conj(a)*b)      # smooth envelopes and correlation
    c2=np.abs(cab)**2/(pa*pb)
    eth=1-(np.pi/4)*np.sum(np.sqrt(pa*pb)*F(c2))/np.sum(np.sqrt(pa*pb))*np.sum(np.sqrt(pa*pb))  # ETH random-amplitude model
    ga=Kg@np.abs(a)**2; gb=Kg@np.abs(b)**2; ga/=ga.sum(); gb/=gb.sum()
    coarse=1-np.sum(np.sqrt(ga*gb))
    res.append((act,haar,eth,coarse,1-s2))
res=np.array(res)
err_h=np.abs(res[:,0]-res[:,1]); err_e=np.abs(res[:,0]-res[:,2])
print(f"n={n} ({time.time()-t0:.0f}s, {trials} pairs): mean Hel^2 actual {res[:,0].mean():.4f} | "
      f"mean |actual - Haar formula| {err_h.mean():.4f} | mean |actual - ETH model| {err_e.mean():.4f} | "
      f"mean coarse Hel^2 {res[:,3].mean():.5f} | coarse/actual {np.mean(res[:,3]/res[:,0]):.3f}",flush=True)
