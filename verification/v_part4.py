import sys; sys.path.insert(0,"/home/claude")
import numpy as np, subprocess, re
out=[]
def rep(tag,ok,msg): out.append(f"[{'PASS' if ok else 'FAIL'}] {tag}: {msg}"); print(out[-1],flush=True)
exec(open('ph2.py').read().rsplit('if __name__',1)[0])
# Lemma robust: history-reproduction identity D(V^dag psi(t+s), psi(t0+s)) independent of s
rng=np.random.default_rng(7); N=32
U=np.linalg.qr(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))[0]; E=rng.normal(size=N)
Ut=lambda t: U@np.diag(np.exp(-1j*E*t))@U.conj().T
V=U@np.diag(np.exp(1j*rng.uniform(0,6,N)))@U.conj().T; psi0=haar_state(N,rng)
d=[np.sqrt(1-abs(np.vdot(V.conj().T@Ut(1.3+s)@psi0,Ut(s)@psi0))**2) for s in np.linspace(-50,50,21)]
rep("Lemma robust: history reproduced uniformly in s",np.ptp(d)<1e-10,f"spread of D over s = {np.ptp(d):.1e}")
# Prop ETH algebra with random parameters (exact model expectation vs stated lower bound)
from scipy.special import hyp2f1
F=lambda x: hyp2f1(-0.5,-0.5,1,x); ok=True
for _ in range(20000):
    K=rng.integers(2,40); rho=rng.dirichlet(np.ones(K)); rhop=rng.dirichlet(np.ones(K)); c2=rng.uniform(0,1,K)**rng.uniform(0.2,3)
    w=np.sqrt(rho*rhop); BC=w.sum(); gam=np.sum(w*(1-c2))/BC; Eg=np.pi/4*np.sum(w*F(c2))
    ok&= 1-Eg >= (1-BC)+(1-np.pi/4)*BC*gam-1e-12 and (1-BC)+BC*(1-np.pi/4)*gam >= (1-np.pi/4)*gam-1e-12 and BC<=1+1e-12
rep("Prop ETH-A inequality chain",ok,"holds for 20,000 random envelope/correlation profiles")
open('verify/log_part4a.txt','w').write("\n".join(out)+"\n")
