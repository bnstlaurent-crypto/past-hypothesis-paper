import numpy as np
from scipy.special import hyp2f1
# Random-matrix prediction for the separation ratio Hel/D between profiles of two states with overlap |s|
s=np.linspace(0,0.999,2000); D=np.sqrt(1-s**2)
hel=np.sqrt(1-(np.pi/4)*hyp2f1(-0.5,-0.5,1,s**2))
r=hel/D
print(f"far pairs (s=0): Hel/D = {r[0]:.4f} = sqrt(1-pi/4) = {np.sqrt(1-np.pi/4):.4f}")
print(f"near pairs (s->1): Hel/D -> {r[-1]:.4f}  (Fisher-metric prediction 0.5)")
print(f"minimum over all overlaps: {r.min():.4f} at s={s[r.argmin()]:.2f}")
