import numpy as np, sys, time
from scipy.optimize import minimize
from mereo import *
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

def run(n, seed, thetas, starts):
    rng=np.random.default_rng(seed); N=2**n
    H=generic_complex(n,rng); E,Q=np.linalg.eigh(H)
    omega=0.05; Ec=np.round(E/omega)*omega; T=2*np.pi/omega; dt=0.05/(Ec.max()-Ec.min())
    p=random_product(n,rng); r=haar_state(N,rng); r-=np.vdot(p,r)*p; r/=np.linalg.norm(r)
    print(f"\n=== n={n} qubits (N={N}), torus dim {N-1}, product-state manifold dim {2*n} ===")
    print(" theta | S(psi0) | S at true-bounce min | best S anywhere | best reading's fidelity with history | rivals more orderly than truth, off-history (F<0.9) / distinct minima")
    for th in thetas:
        psi0=np.cos(th)*p+np.sin(th)*r; c0=Q.conj().T@psi0
        f=lambda phi: lin_ent(Q@(np.exp(-1j*phi)*c0),n)
        mins=[]
        x0s=[np.zeros(N)]+[rng.uniform(0,2*np.pi,N) for _ in range(starts)]
        for x0 in x0s:
            res=minimize(f,x0,method='BFGS',options={'gtol':1e-9})
            mins.append((res.fun,res.x))
        # cluster
        clusters=[]
        for v,x in sorted(mins,key=lambda z:z[0]):
            if not any(dist(x,c[1])<1e-3 for c in clusters): clusters.append((v,x))
        Sof=lambda x: S(Q@(np.exp(-1j*x)*c0),n)
        true_v=mins[0][0]; true_x=mins[0][1]           # local min reached from the true bounce
        best_v,best_x=clusters[0]
        pw=np.abs(c0)**2
        F_best=fid_to_loop(best_x,pw,Ec,T,dt); F_true=fid_to_loop(true_x,pw,Ec,T,dt)
        rivals=[c for c in clusters if c[0]<true_v*(1-1e-6) and fid_to_loop(c[1],pw,Ec,T,dt)<0.9]
        print(f" {th:5.2f} | {S(psi0,n):7.4f} | {Sof(true_x):8.4f} (F={F_true:.3f}) | {Sof(best_x):15.4f} | {F_best:36.3f} | {len(rivals)} / {len(clusters)}")
        sys.stdout.flush()

t0=time.time()
run(3, 11, [0.0,0.05,0.1,0.2,0.3,0.5], 150)
run(4, 11, [0.0,0.05,0.1,0.2,0.3,0.5], 90)
print(f"({time.time()-t0:.0f}s)")
