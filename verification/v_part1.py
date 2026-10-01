# Verification part 1: analytic lemmas and constants
import numpy as np, mpmath as mp
from scipy.special import hyp2f1
from scipy.optimize import minimize
from scipy.stats import unitary_group
rng=np.random.default_rng(12345); out=[]
def rep(tag,ok,msg): out.append(f"[{'PASS' if ok else 'FAIL'}] {tag}: {msg}"); print(out[-1],flush=True)
def rstate(N,r=rng): v=r.normal(size=N)+1j*r.normal(size=N); return v/np.linalg.norm(v)
def haar(N): Z=(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))/np.sqrt(2); Q,R=np.linalg.qr(Z); return Q*(np.diag(R)/np.abs(np.diag(R)))
F=lambda x: hyp2f1(-0.5,-0.5,1,x)
Dist=lambda a,b: np.sqrt(max(0,1-abs(np.vdot(a,b))**2))
def hel(p,q): return np.sqrt(max(0,1-np.sum(np.sqrt(p*q))))

# L1: Lemma 1 (fibre) - both directions
N=16; U=haar(N); psi=rstate(N); c=U.conj().T@psi
phi=rng.uniform(0,2*np.pi,N); V=U@np.diag(np.exp(1j*phi))@U.conj().T
chi=V.conj().T@psi
ok1=np.allclose(np.abs(U.conj().T@chi),np.abs(c))
chi2=U@(np.abs(c)*np.exp(1j*rng.uniform(0,2*np.pi,N)))           # arbitrary state with same profile
ph=np.angle(U.conj().T@psi)-np.angle(U.conj().T@chi2)
V2=U@np.diag(np.exp(1j*ph))@U.conj().T
rep("Lemma 1 fibre",ok1 and abs(abs(np.vdot(V2.conj().T@psi,chi2))-1)<1e-12,"readings keep the profile; any same-profile state is a reading")
H=U@np.diag(rng.normal(size=N))@U.conj().T
rep("Locality preserved",np.allclose(V.conj().T@H@V,H),"V^dag H V = H for V in Gamma")

# L2: Hel <= D, random + adversarial
worst=-1
for N in (2,4,8,32):
    for _ in range(3000):
        U=haar(N); a=rstate(N); b=rstate(N) if rng.random()<.5 else (a+0.1*rstate(N)); b/=np.linalg.norm(b)
        worst=max(worst,hel(np.abs(U.conj().T@a)**2,np.abs(U.conj().T@b)**2)-Dist(a,b))
    def negs(x):
        a=x[:N]+1j*x[N:2*N]; b=x[2*N:3*N]+1j*x[3*N:]; a/=np.linalg.norm(a); b/=np.linalg.norm(b)
        return -(hel(np.abs(a)**2,np.abs(b)**2)-Dist(a,b))
    for _ in range(20): worst=max(worst,-minimize(negs,rng.normal(size=4*N),method='Nelder-Mead',options={'maxiter':4000}).fun)
rep("Lemma A1 Hel<=D",worst<=1e-9,f"max(Hel-D) over random+adversarial search = {worst:.2e}")

# L3: exact expectation at small N, and each derivation step
ok=True; msgs=[]
for N in (2,3,4,8):
    for s in (0.0,0.5,0.9):
        q=np.zeros(N,complex); q[0]=1; qp=np.zeros(N,complex); qp[0]=s; qp[1]=np.sqrt(1-s*s)
        vals=[]
        for _ in range(30000 if N<=4 else 12000):
            u=haar(N); vals.append(np.sum(np.abs(u.conj().T@q)*np.abs(u.conj().T@qp)))
        m=np.mean(vals); se=np.std(vals)/np.sqrt(len(vals)); pred=np.pi/4*F(s*s)
        if abs(m-pred)>4*se+1e-4: ok=False
        msgs.append(f"N={N},s={s}: {m:.4f}±{se:.4f} vs {pred:.4f}")
rep("Lemma A2 exact identity at small N",ok,"; ".join(msgs[:4])+" ...")
mp.mp.dps=30; ok=True
for s in (0.1,0.4,0.7,0.95):
    c=mp.sqrt(1-s*s)
    # Rice mean check at a sample |X|=x0
    x0=mp.mpf('0.8'); nu=s*x0
    rice=mp.quad(lambda r: r*r/(c*c/2)*mp.e**(-(r*r+nu*nu)/c**2)*mp.besseli(0,2*r*nu/c**2)*1,[0,mp.inf])
    rice_formula=c*mp.sqrt(mp.pi)/2*mp.hyp1f1(-0.5,1,-nu*nu/c**2)
    lap=mp.quad(lambda u: mp.sqrt(u)*mp.e**(-u)*mp.hyp1f1(-0.5,1,-(s*s/c**2)*u),[0,mp.inf])
    lap_formula=mp.gamma(1.5)*mp.hyp2f1(-0.5,1.5,1,-s*s/c**2)
    pfaff=c*mp.pi/4*mp.hyp2f1(-0.5,1.5,1,-s*s/c**2)-mp.pi/4*mp.hyp2f1(-0.5,-0.5,1,s*s)
    ok&= abs(rice-rice_formula)<1e-12 and abs(lap-lap_formula)<1e-12 and abs(pfaff)<1e-15
rep("Lemma A2 derivation steps (Rice mean, Laplace transform, Pfaff)",ok,"all three identities hold to 1e-12 at s=0.1,0.4,0.7,0.95")
x=np.linspace(0,1,200001); G=np.pi/4*(1-F(x))+(1-np.pi/4)*x
coef=[float(mp.rf(-0.5,m)**2/mp.factorial(m)**2) for m in range(1,30)]
rep("Lemma A2 G>=0, F convex, F(1)=4/pi",G.min()>=-1e-15 and min(coef)>0 and abs(F(1.0)-4/np.pi)<1e-12,
    f"min G = {G.min():.2e}; Taylor coefficients positive; F(1)={F(1.0):.12f} vs 4/pi={4/np.pi:.12f}")
# chord bound used in Prop ETH
rep("Chord bound F(x) <= 4/pi-(4/pi-1)(1-x)",np.all(F(x)<=4/np.pi-(4/np.pi-1)*(1-x)+1e-15),"holds on a 200k-point grid")

# L4: Lipschitz constants
worst=0; worstD=0
for N in (4,16,64):
    for _ in range(400):
        U=haar(N); m=rstate(N); mp_=rstate(N)
        A=rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)); A=(A-A.conj().T)/2
        for eps in (1e-1,1e-3):
            U2=U@__import__('scipy.linalg',fromlist=['expm']).expm(eps*A/np.linalg.norm(A))
            g=lambda W: np.sum(np.abs(W.conj().T@m)*np.abs(W.conj().T@mp_))
            worst=max(worst,abs(g(U)-g(U2))/np.linalg.norm(U-U2))
        mt=m+0.2*rstate(N); mt/=np.linalg.norm(mt)
        gm=lambda a,b: np.sum(np.abs(U.conj().T@a)*np.abs(U.conj().T@b))
        worstD=max(worstD,abs(gm(m,mp_)-gm(mt,mp_))/Dist(m,mt))
rep("Lemma A3 Lipschitz in U (bound 2)",worst<=2,f"max observed ratio {worst:.3f}")
rep("Lemma A3 Lipschitz in states (bound sqrt2)",worstD<=np.sqrt(2),f"max observed ratio {worstD:.3f}")

# L5: nets ingredients
ok=True
for _ in range(2000):
    n=rng.integers(2,6); qs=[rstate(2) for _ in range(n)]; qt=[]
    for q in qs:
        w=q+0.3*rstate(2); qt.append(w/np.linalg.norm(w))
    from functools import reduce
    Q=reduce(np.kron,qs); Qt=reduce(np.kron,qt)
    ok&= Dist(Q,Qt)<=sum(Dist(a,b) for a,b in zip(qs,qt))+1e-12
    bl=lambda v: np.real([np.vdot(v,M@v) for M in (np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]))])
    ok&= abs(Dist(qs[0],qt[0])-np.linalg.norm(bl(qs[0])-bl(qt[0]))/2)<1e-12
    a=rstate(8); b=rstate(8)*np.exp(1j*rng.uniform(0,6)); ok&= Dist(a,b)<=np.linalg.norm(a-b)+1e-12
rep("Lemma A4 ingredients",ok,"D subadditive on products; D = |r-r'|/2 for qubits; D <= Euclidean distance")
# MPS telescoping with random isometries
ok=True
for _ in range(300):
    n=5; chi=2; Vs=[];Vt=[]; dims=[1]+[min(chi,2**min(i,n-i)) for i in range(1,n)]+[1]
    def iso(r,c):
        Z=rng.normal(size=(r,c))+1j*rng.normal(size=(r,c)); q,_=np.linalg.qr(Z); return q
    for i in range(n):
        Vi=iso(2*dims[i],dims[i+1]); W=Vi+0.05*(rng.normal(size=Vi.shape)+1j*rng.normal(size=Vi.shape)); Wi,_=np.linalg.qr(W)
        Vs.append(Vi); Vt.append(Wi)
    def state(Vl):
        W=np.eye(1)
        for Vi in Vl: W=np.kron(W,np.eye(2))@Vi if False else (np.kron(W,np.eye(2))@Vi)
        return W[:,0]
    psi=state(Vs); psit=state(Vt)
    ok&= abs(np.linalg.norm(psi)-1)<1e-10 and np.linalg.norm(psi-psit)<=sum(np.linalg.norm(a-b,2) for a,b in zip(Vs,Vt))+1e-10
rep("Lemma A4 MPS telescoping",ok,"||psi - psi~|| <= sum of isometry distances (300 random chains)")
open('log_part1.txt','w').write("\n".join(out)+"\n")
