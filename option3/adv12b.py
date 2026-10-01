import numpy as np, sys, time
exec(open('adv.py').read().split('if __name__')[0])
Q=np.load('Q12.npy'); E=np.load('E12.npy'); n=12
r=np.random.default_rng(int(sys.argv[1])); Hs=chaotic(n,np.random.default_rng(1))
t=time.time(); w,e=search(n,Hs,E,Q,r,"hot",int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
print(f"n=12 chaotic seed=1 hot [energy {e:.2f}]: worst ratios {np.round(w,2)} min {min(w):.2f} ({time.time()-t:.0f}s)",flush=True)
