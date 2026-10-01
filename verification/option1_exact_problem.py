import numpy as np, math
from functools import reduce
exec(open('ph2.py').read().rsplit('if __name__',1)[0])
rng=np.random.default_rng(3); n=5; N=2**n
H=generic_complex(n,rng); E,Q=np.linalg.eigh(H)
# (1) twins <=> equal energy moments <q|H^j|q>, j=1..N-1  (nondegenerate spectrum)
q=random_product(n,rng); c=Q.conj().T@q
qt=Q@(np.abs(c)*np.exp(1j*rng.uniform(0,6.3,N)))          # same profile, different state
mom=lambda v: np.array([np.vdot(v,np.linalg.matrix_power(H,j)@v).real for j in range(1,N)])
V=np.vander(E,N,increasing=True)[:,1:]                       # moments = V^T profile
print("(1) equal profiles -> equal moments:",np.allclose(mom(q),mom(qt),rtol=1e-8),
      "| moments determine profile (Vandermonde invertible, cond %.1e)"%np.linalg.cond(np.vander(E,N,increasing=True)))
# (2) each moment is multilinear (affine) in each site's Bloch vector for product states
def prod_from_bloch(rs):
    sp=[]
    for r in rs:
        th=np.arccos(np.clip(r[2],-1,1)); ph=np.arctan2(r[1],r[0]); sp.append(np.array([np.cos(th/2),np.exp(1j*ph)*np.sin(th/2)]))
    return reduce(np.kron,sp)
def affine_test():
    rs=[x/np.linalg.norm(x) for x in rng.normal(size=(n,3))]
    a,b=[x/np.linalg.norm(x) for x in rng.normal(size=(2,3))]
    # affine in r_0: value at a mixed state along the chord = mixture of values -> use density-matrix form
    P=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1])]
    rho=lambda r: (np.eye(2)+sum(r[i]*P[i] for i in range(3)))/2
    def val(r0,j):
        R=reduce(np.kron,[rho(r0)]+[rho(r) for r in rs[1:]]); return np.trace(R@np.linalg.matrix_power(H,j)).real
    ok=True
    for j in (1,3,7,20):
        for lam in (0.3,0.7):
            ok&=abs(val(lam*a+(1-lam)*b,j)-(lam*val(a,j)+(1-lam)*val(b,j)))<1e-8*max(1,abs(val(a,j)))
    return ok
print("(2) moments are affine in each site's Bloch vector:",affine_test())
# (3) size of the certification problem: 20 real unknowns (10 Bloch spheres), >=20 multilinear equations
for nn in (5,6,7):
    blocks=2*nn; eqs=2*blocks
    bez=math.factorial(eqs)//(math.factorial(2)**blocks)
    print(f"(3) n={nn}: {2**nn-1} moment equations, {2*blocks} unknowns; multihomogeneous Bezout count for a square subsystem ~ {bez:.1e} paths")
# (4) symmetry twins: chaotic Z2-symmetric chain (P = product of X's commutes with H)
n=6; I2=np.eye(2); X=np.array([[0,1],[1,0]]); Z=np.diag([1.,-1])
def op(o,i): return reduce(np.kron,[o if k==i else I2 for k in range(n)])
Hs=sum(op(Z,i)@op(Z,i+1) for i in range(n-1))+0.5*sum(op(Z,i)@op(Z,i+2) for i in range(n-2))+sum((0.9+0.3*np.sin(i))*op(X,i) for i in range(n))
P=reduce(np.kron,[X]*n); Es,Qs=np.linalg.eigh(Hs)
q=random_product(n,rng); prof=lambda v: np.abs(Qs.conj().T@v)**2
Dq=np.sqrt(1-abs(np.vdot(q,P@q))**2)
print(f"(4) Z2-symmetric chaotic chain: [H,P]=0: {np.allclose(Hs@P,P@Hs)}; profiles of q and Pq identical: {np.allclose(prof(q),prof(P@q))}; distance D(q,Pq)={Dq:.2f}")
