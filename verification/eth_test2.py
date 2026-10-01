import numpy as np, sys, time
from scipy.special import hyp2f1
from functools import reduce
exec(open('ph2.py').read().rsplit('if __name__',1)[0])
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
