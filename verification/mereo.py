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
