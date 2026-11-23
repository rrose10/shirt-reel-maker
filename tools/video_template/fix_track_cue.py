#!/usr/bin/env python3
"""Cancel tracker drift in a window by following the rigid shirt cues (collar/shoulders) instead.

usage: fix_track_cue.py <template id> <first frame> <last frame> [taper=8]   (run from the repo root)
Inside the window the print centre and scale are corrected by the deviation cue_drift measures; the correction fades in/out over `taper` frames.
"""
import sys,json,numpy as np,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import cue_drift
from scipy.ndimage import gaussian_filter1d
name,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);taper=int(sys.argv[4]) if len(sys.argv)>4 else 8
p=cue_drift.T+name+'/track.json';t=json.load(open(p));H=np.array(t['H'],float).reshape(-1,3,3);n=len(H)
out,_=cue_drift.run(name)
dev=np.zeros((n,3))
for r in out:
    if r:dev[r[0]]=[r[1],r[2],r[3]]
# fill gaps (frames where no cue estimate) by interpolation, then smooth
for j in range(3):
    ok=np.array([bool(r) for r in [None]+out]);idx=np.arange(n)
    dev[:,j]=np.interp(idx,idx[ok],dev[ok,j]) if ok.any() else 0
dev=gaussian_filter1d(dev,2.0,axis=0)
w=np.zeros(n)
for k in range(n):
    if a<=k<=b:w[k]=min(1,(k-a+1)/taper,(b-k+1)/taper)
x,y,bw,bh=t['bbox'];c=np.array([x+bw/2,y+bh/2,1.0])
for k in range(n):
    if w[k]==0:continue
    ds,dx,dy=dev[k]*np.array([1,1,1])
    s=1/(1+w[k]*ds/100);M=H[k].copy()
    # scale about the print centre, then move the centre
    q=M@c;q=q[:2]/q[2]
    S=np.eye(3);S[:2,:2]*=s;S[:2,2]=q-s*q;S2=np.eye(3);S2[:2,2]=[-w[k]*dx,-w[k]*dy]
    H[k]=S2@S@M
t['H']=[[round(float(v),6) for v in M.reshape(9)] for M in H]
json.dump(t,open(p,'w'),separators=(',',':'))
print('corrected',name,a,'..',b,'max |dev| in window: scale %.1f%% dx %.1f dy %.1f'%tuple(np.abs(dev[a:b+1]).max(0)))
