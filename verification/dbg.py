import numpy as np, mpmath as mp
from functools import reduce
mp.mp.dps=30
for s in (0.1,0.4,0.7,0.95):
    c=mp.sqrt(1-s*s); x0=mp.mpf('0.8'); nu=s*x0
    rice=mp.quad(lambda r: 2*r*r/c**2*mp.e**(-(r*r+nu*nu)/c**2)*mp.besseli(0,2*r*nu/c**2),[0,mp.inf])
    rf=c*mp.sqrt(mp.pi)/2*mp.hyp1f1(-0.5,1,-nu*nu/c**2)
    lap=mp.quad(lambda u: mp.sqrt(u)*mp.e**(-u)*mp.hyp1f1(-0.5,1,-(s*s/c**2)*u),[0,mp.inf])
    lf=mp.gamma(1.5)*mp.hyp2f1(-0.5,1.5,1,-s*s/c**2)
    pf=c*mp.pi/4*mp.hyp2f1(-0.5,1.5,1,-s*s/c**2)-mp.pi/4*mp.hyp2f1(-0.5,-0.5,1,s*s)
    print(f"s={s}: Rice {float(rice-rf):.2e} | Laplace {float(lap-lf):.2e} | Pfaff {float(pf):.2e}")
# direct end-to-end check of E|X||Y| for correlated complex Gaussians by 2D quadrature over |X| (exact inner Rice mean)
for s in (0.3,0.8):
    c=mp.sqrt(1-s*s)
    val=mp.quad(lambda u: mp.sqrt(u)*mp.e**(-u)*c*mp.sqrt(mp.pi)/2*mp.hyp1f1(-0.5,1,-(s*s/c**2)*u),[0,mp.inf])
    print(f"s={s}: E|X||Y| by quadrature {float(val):.12f} vs (pi/4)F = {float(mp.pi/4*mp.hyp2f1(-0.5,-0.5,1,s*s)):.12f}")
# fixed nets test (bug: unnormalised perturbed states in the test)
rng=np.random.default_rng(1)
def rs(N): v=rng.normal(size=N)+1j*rng.normal(size=N); return v/np.linalg.norm(v)
D=lambda a,b: np.sqrt(max(0,1-abs(np.vdot(a,b))**2)); ok=True
P=(np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]))
for _ in range(3000):
    n=rng.integers(2,6); qs=[rs(2) for _ in range(n)]; qt=[]
    for q in qs:
        w=q+0.3*rs(2); qt.append(w/np.linalg.norm(w))
    ok&= D(reduce(np.kron,qs),reduce(np.kron,qt))<=sum(D(a,b) for a,b in zip(qs,qt))+1e-12
    bl=lambda v: np.real([np.vdot(v,M@v) for M in P])
    ok&= abs(D(qs[0],qt[0])-np.linalg.norm(bl(qs[0])-bl(qt[0]))/2)<1e-12
    a=rs(8); b=rs(8); ok&= D(a,b)<=np.linalg.norm(a-b)+1e-12
print("Lemma A4 ingredients with corrected test:", "PASS" if ok else "FAIL")
