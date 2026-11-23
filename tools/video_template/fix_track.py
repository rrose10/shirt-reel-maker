#!/usr/bin/env python3
"""Replace a stretch of a template's tracking by a straight bridge between its two ends.

Why: on plain white shirts the hand smooths the fabric, the wrinkles change, and the tracker can drift
(print gets smaller/bigger and slides up/down) while the shirt itself (collar, sleeves) stays put.
Check with ECC on the collar/sleeves; where they say "no motion" but track.json says otherwise, bridge it.

usage: fix_track.py <template dir> <first bad frame> <last bad frame>
Frames first..last are rebuilt from the similarity (scale, rotation, bbox-centre position) at first-1 and last+1.
"""
import json,sys,numpy as np
d,a,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
t=json.load(open(d+'/track.json'));H=np.array(t['H'],float).reshape(-1,3,3)
x,y,w,h=t['bbox'];c=np.array([x+w/2,y+h/2,1.0])
def par(M):
    s=np.hypot(M[0,0],M[1,0]);th=np.arctan2(M[1,0],M[0,0]);q=M@c;return np.array([np.log(s),th,q[0]/q[2],q[1]/q[2]])
def mat(p):
    s=np.exp(p[0]);R=s*np.array([[np.cos(p[1]),-np.sin(p[1])],[np.sin(p[1]),np.cos(p[1])]])
    M=np.eye(3);M[:2,:2]=R;M[:2,2]=p[2:4]-R@c[:2];return M
p0,p1=par(H[a-1]),par(H[b+1])
for k in range(a,b+1):
    u=(k-(a-1))/((b+1)-(a-1));u=u*u*(3-2*u)  # smoothstep: no kink at the ends
    H[k]=mat(p0*(1-u)+p1*u)
t['H']=[[round(float(v),6) for v in M.reshape(9)] for M in H]
json.dump(t,open(d+'/track.json','w'),separators=(',',':'))
print('bridged',a,'..',b)
