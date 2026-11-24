#!/usr/bin/env python3
"""Re-anchor a template's track on the collar (neckband) so the print rides on the shirt, not on whatever a tracker latched on to.

A patch around the collar of frame 0 is searched in every frame (scaled by the track's own zoom, normalised correlation, +-90 px around
the track's prediction). The track becomes a similarity: scale and angle of the old track (smoothed), shifted so the collar patch lands
where it was found. The shift is smoothed with a smoothing spline. The old track is kept as track_before_collar.json.

usage: collar_anchor.py <template dir> x,y,w,h   (patch in frame-0 plate pixels, e.g. the neckline: 300,520,120,70) [lam=100]
"""
import cv2,numpy as np,json,sys,shutil,os
from scipy.interpolate import make_smoothing_spline
d=sys.argv[1].rstrip('/')+'/';x0,y0,pw,ph=map(int,sys.argv[2].split(','));lam=float(sys.argv[3]) if len(sys.argv)>3 else 100
t=json.load(open(d+'track.json'));H=np.array(t['H'],float).reshape(-1,3,3);n=len(H)
c=cv2.VideoCapture(d+'template.mp4');fr=[]
while True:
    ok,x=c.read()
    if not ok:break
    fr.append(cv2.cvtColor(x[:t['h'],:t['w']],cv2.COLOR_BGR2GRAY))
P=fr[0][y0:y0+ph,x0:x0+pw];pc=np.array([x0+pw/2,y0+ph/2,1.]);S=90
sc=np.hypot(H[:,0,0],H[:,1,0]);an=np.arctan2(H[:,1,0],H[:,0,0]);pos=[];cor=[]
for k in range(n):
    s=sc[k];cx,cy=(H[k]@pc)[:2]/(H[k]@pc)[2]
    Ps=cv2.resize(P,None,fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
    ax0=max(int(cx-Ps.shape[1]/2-S),0);ay0=max(int(cy-Ps.shape[0]/2-S),0)
    reg=fr[k][ay0:ay0+Ps.shape[0]+2*S,ax0:ax0+Ps.shape[1]+2*S]
    r=cv2.matchTemplate(reg,Ps,cv2.TM_CCOEFF_NORMED);_,mv,_,ml=cv2.minMaxLoc(r);mx,my=ml
    dx=dy=0.
    if 0<mx<r.shape[1]-1:
        a,b,c_=r[my,mx-1],r[my,mx],r[my,mx+1];dx=(a-c_)/(2*(a-2*b+c_)+1e-9)
    if 0<my<r.shape[0]-1:
        a,b,c_=r[my-1,mx],r[my,mx],r[my+1,mx];dy=(a-c_)/(2*(a-2*b+c_)+1e-9)
    pos.append((ax0+mx+dx+Ps.shape[1]/2,ay0+my+dy+Ps.shape[0]/2));cor.append(mv)
pos=np.array(pos);cor=np.array(cor);print('match corr min %.3f mean %.3f'%(cor.min(),cor.mean()))
x=np.arange(n);Q=np.c_[sc,an,pos]
Qs=np.array([make_smoothing_spline(x,Q[:,j],lam=lam)(x) for j in range(4)]).T
if not os.path.exists(d+'track_before_collar.json'):shutil.copy(d+'track.json',d+'track_before_collar.json')
out=[]
for k in range(n):
    s,a,qx,qy=Qs[k];R=np.array([[s*np.cos(a),-s*np.sin(a)],[s*np.sin(a),s*np.cos(a)]]);tl=np.array([qx,qy])-R@pc[:2]
    out.append([round(float(v),6) for v in (R[0,0],R[0,1],tl[0],R[1,0],R[1,1],tl[1],0,0,1)])
t['H']=out;json.dump(t,open(d+'track.json','w'),separators=(',',':'))
print('moved the anchor by up to %.1f px'%np.abs(Qs[:,2:]-np.array([(H[k]@pc)[:2]/(H[k]@pc)[2] for k in range(n)])).max())
