import numpy as np
exec(open('diag.py').read().split('for n in (6,8):')[0])
T=np.linspace(-400,400,80001)
def mind(a,b,E):
    m=0
    for i in range(0,len(T),4000): m=max(m,np.max(np.abs(np.exp(-1j*np.outer(T[i:i+4000],E))@(np.conj(a)*b))))
    return np.sqrt(max(0,1-m**2))
for n in (6,8):
    for seed in (1,2):
        (R,(q,q2)),E,Q,Hs=find(n,'near-integrable',seed)
        c2=Q.conj().T@q2; cconj=Q.conj().T@np.conj(q)
        Hd=Hs.toarray(); real=np.allclose(Hd.imag,0)
        # what the alternative division shows: the history read in T_V is K psi(-t), i.e. time-reversed
        print(f"n={n} seed={seed}: H real: {real} | distance from twin to time-shifts of conj(q): {mind(cconj,c2,E):.1e} | ratio {R:.1e}",flush=True)
# control: give the chaotic model's couplings real values only -> twins should appear; complex couplings -> none
r=np.random.default_rng(1); n=6
Hc=chaotic(n,r).toarray(); Hreal=Hc.real.copy(); Hreal=(Hreal+Hreal.T)/2
for name,H in (("chaotic, complex couplings",Hc),("same model with couplings made real",Hreal)):
    E,Q=np.linalg.eigh(H); q=prod(np.random.default_rng(5).uniform(0,3,2*n)); c=Q.conj().T@q; cc=Q.conj().T@np.conj(q)
    hel=1-np.sum(np.abs(c)*np.abs(cc)); D=np.sqrt(1-abs(np.vdot(q,np.conj(q)))**2)
    print(f"{name}: Hel^2(q, conj q) = {hel:.2e}, distance D = {D:.2f}")
