import sys; sys.path.insert(0,"/home/claude")
import numpy as np, json, collections
exec(open('ph2.py').read().rsplit('if __name__',1)[0])
out=[]
def rep(tag,ok,msg): out.append(f"[{'PASS' if ok else 'FAIL'}] {tag}: {msg}"); print(out[-1],flush=True)
# chaos check
for n in (8,10):
    rs=[]
    for sd in (1,2,3):
        rng=np.random.default_rng(sd); H=generic_complex(n,rng); E=np.linalg.eigvalsh(H)
        m=len(E); E=E[m//4:3*m//4]; s=np.diff(E); r=np.minimum(s[1:],s[:-1])/np.maximum(s[1:],s[:-1]); rs.append(r.mean())
    rep(f"Chaos check n={n} (GUE 0.600, GOE 0.536, Poisson 0.386)",np.mean(rs)>0.56,f"mean level-spacing ratio <r> = {np.mean(rs):.3f}")
# ground-state bounce entanglement range for the tested fields (paper caption)
vals=[S(ground_bounce(n,g,None),n) for n in (3,4,5,6,7) for g in (3.0,2.0,1.5,1.2,1.0,0.8)]
rep("Ground-state bounce entanglement range (caption says 0.03-0.32)",abs(min(vals)-0.03)<0.01 and abs(max(vals)-0.32)<0.01,
    f"actual range {min(vals):.3f}-{max(vals):.3f} bits")
# Table 1 recomputed from raw data
def load(n):
    recs=[]
    for f in [f"ph_n{n}.jsonl", f"ph2_perturbed_n{n}.jsonl"]:
        try:
            for l in open(f): r=json.loads(l); r['par']=r.get('par',r.get('theta')); recs.append(r)
        except FileNotFoundError: pass
    return recs
paper={3:(0.00,0.00,0.02),4:(0.27,0.06,0.48),5:(0.32,0.19,0.74),6:(0.57,0.18,0.59),7:(0.77,0.67,0.78)}
for n in (3,4,5,6,7):
    recs=load(n); mids=[]
    for sd in sorted({r['seed'] for r in recs}):
        rs=sorted([r for r in recs if r['seed']==sd],key=lambda r:r['par']); last=0.0; fail=None
        for r in rs:
            if r['on_history']: last=r['S0']
            else: fail=r['S0']; break
        mids.append((last+fail)/2 if fail is not None else last)
    med,lo,hi=np.median(mids),min(mids),max(mids); pm=paper[n]
    rep(f"Table 1 perturbed n={n}",abs(med-pm[0])<0.006 and abs(lo-pm[1])<0.006 and abs(hi-pm[2])<0.006,f"recomputed {med:.2f} ({lo:.2f}-{hi:.2f}) vs paper {pm[0]:.2f} ({pm[1]:.2f}-{pm[2]:.2f})")
r8=[json.loads(l) for l in open('ph2_perturbed_n8.jsonl')]
fails8=[r for r in r8 if not r['on_history']]; maxS=max(r['S0'] for r in r8)
rep("Table 1 n=8 (no failure up to 0.82)",len(fails8)==0,f"{len(r8)} records, {len(fails8)} failures, largest bounce entanglement tested {maxS:.3f}")
gp={3:12,4:14,5:29,6:30,7:36}
for n in (3,4,5,6,7):
    rs=[json.loads(l) for l in open(f'ph2_ground_n{n}.jsonl')]; ok=sum(r['on_history'] for r in rs)
    rep(f"Table 1 ground n={n}",ok==gp[n] and len(rs)==36,f"{ok}/{len(rs)} vs paper {gp[n]}/36")
rs=[json.loads(l) for l in open('ph2_ground_n8.jsonl')]
rep("Table 1 ground n=8",sum(r['on_history'] for r in rs)==len(rs)==6,f"{sum(r['on_history'] for r in rs)}/{len(rs)}")
# Table 2 (MPS, ground) from raw data
tp={5:(6,9),6:(9,9),7:(12,12),8:(6,6)}
for n in (5,6,7,8):
    rs=[json.loads(l) for l in open(f'mps_n{n}_chi2.jsonl')]
    rs=[r for r in rs if r['g'] in (1.5,1.2,1.0)]
    ok=sum(r['on_history'] for r in rs)
    rep(f"Table 2 MPS ground n={n}",(ok,len(rs))==tp[n],f"{ok}/{len(rs)} vs paper {tp[n][0]}/{tp[n][1]}")
open('verify/log_part3.txt','w').write("\n".join(out)+"\n")
