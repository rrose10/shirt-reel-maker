#!/usr/bin/env python3
"""Re-derive a flat-lay template's rigid track from the SHIRT SILHOUETTE only.

Leaf shadows, the hands and their shadows fool trackers that look at the fabric texture, so the print drifts while the shirt
does not. The outline of the shirt against the table (low saturation + bright, hand masked) is not fooled by any of that.
For every frame the print transform is  H_k = G_k . B_k  where B_k is the old track (or frame 0's transform while the shirt is
not zooming: `still` frames) and G_k = (scale about the plate centre, shift) is found by chamfer matching: the frame-0 outline,
mapped through the transform, must lie on the observed outline of frame k (points near the hand or the frame border ignored).

usage: silhouette_track.py <template id> <still_until> [free_scale_from=still_until+1]
   frames 1..still_until:  B = H_0, only a shift is fitted (the camera is static, shirt may shift a little)
   later frames:           B = old track, ONLY a vertical shift is fitted
"""
import cv2,numpy as np,json,sys
from scipy.optimize import minimize
tid=sys.argv[1];still=int(sys.argv[2]);d='site/templates/%s/'%tid
t=json.load(open(d+'track.json'));H=np.array(t['H'],float).reshape(-1,3,3);n=len(H)
c=cv2.VideoCapture(d+'template.mp4');DT=[];HM=[];M0=None;SK0=None
while True:
    ok,x=c.read()
    if not ok:break
    p=cv2.resize(x[:, :720],(360,640),interpolation=cv2.INTER_AREA);hsv=cv2.cvtColor(p,cv2.COLOR_BGR2HSV)
    mb=((hsv[...,1]<45)&(hsv[...,2]>95)).astype(np.uint8)
    if M0 is None:M0=mb
    DT.append((cv2.distanceTransform(mb,cv2.DIST_L2,5)-cv2.distanceTransform(1-mb,cv2.DIST_L2,5)).astype(np.float32))
    sk=(cv2.cvtColor(x[:640,720:1080],cv2.COLOR_BGR2GRAY)>40).astype(np.uint8)
    if SK0 is None:SK0=sk     # frame 0: no hand yet - whatever the matte covers there (table...) is not a hand
    HM.append(cv2.dilate(((sk==1)&(SK0==0)).astype(np.uint8),np.ones((41,41),np.uint8)))
cs,_=cv2.findContours(M0,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);P0=max(cs,key=len)[:,0,:].astype(float)   # frame-0 outline, half-res
P0=P0[(P0[:,0]>3)&(P0[:,0]<356)&(P0[:,1]>3)&(P0[:,1]<636)]
ctr=np.array([360.,640.])          # plate centre (full-res)
def cost(par,k,B):
    s,dx,dy=par
    M=np.array([[s,0,ctr[0]*(1-s)+dx],[0,s,ctr[1]*(1-s)+dy],[0,0,1.]])@B@np.linalg.inv(H[0])
    q=(np.c_[P0*2,np.ones(len(P0))]@M.T);q=q[:,:2]/q[:,2:]/2
    ok=(q[:,0]>4)&(q[:,0]<355)&(q[:,1]>4)&(q[:,1]<635)
    if ok.sum()<25:return 1e3
    xi=np.clip(np.round(q[ok,0]).astype(int),0,359);yi=np.clip(np.round(q[ok,1]).astype(int),0,639)
    free=HM[k][yi,xi]==0
    if free.sum()<25:return 1e3
    v=DT[k][yi,xi][free]
    return float(np.mean(np.minimum(np.abs(v),12)))         # robust L1 (outliers = hand edge, shadow)
Hn=H.copy();prev=np.array([1.,0.,0.]);PAR=np.zeros((n,3));PAR[:,0]=1;REL=np.zeros(n,bool);REL[0]=True;FUN=np.zeros(n)
for k in range(1,n):
    B=H[0] if k<=still else H[k]
    free_scale=False
    best=None
    for start in (prev,np.array([1.,0.,0.]),np.array([1.,0.,8.]),np.array([1.,-8.,8.])):
        f=lambda q:cost(np.array([q[0],q[1],q[2]]) if free_scale else np.array([1.,q[1],q[2]]),k,B)
        r=minimize(f,start,method='Nelder-Mead',options={'xatol':.05,'fatol':1e-4,'maxiter':400})
        if best is None or r.fun<best.fun:best=r
    par=best.x if free_scale else np.array([1.,best.x[1],best.x[2]])
    PAR[k]=par;FUN[k]=best.fun;REL[k]=best.fun<5
    if REL[k]:prev=par
from scipy.ndimage import median_filter,gaussian_filter1d
for j in (1,2):      # reject single-frame outliers only (a median over many frames would turn the pull-down ramp into a step)
    v=PAR[1:still+1,j];ref=median_filter(v,size=7,mode='nearest');bad=np.abs(v-ref)>2.5;v[bad]=ref[bad];PAR[1:still+1,j]=gaussian_filter1d(v,0.7,mode='nearest')
# once the camera zooms the outline leaves the frame and single-frame fits get noisy: from there on use one robust vertical offset
# (median of the reliable fits), eased in from the last still-phase value, and no sideways shift.
idx=np.arange(n);late=idx>still
rel_late=REL&late
dy_end=float(np.median(PAR[rel_late,2])) if rel_late.sum()>=3 else float(PAR[still,2])
for k in range(still+1,n):
    w=min(1.,(k-still)/30.);PAR[k]=(1.,0.,PAR[still,2]*(1-w)+dy_end*w)
rows=[]
for k in range(1,n):
    B=H[0] if k<=still else H[k];par=PAR[k]
    G=np.array([[par[0],0,ctr[0]*(1-par[0])+par[1]],[0,par[0],ctr[1]*(1-par[0])+par[2]],[0,0,1.]])
    Hn[k]=G@B;rows.append((k,par,FUN[k]))
t['H']=[[round(float(v),6) for v in h.reshape(-1)] for h in Hn]
json.dump(t,open(d+'track.json','w'),separators=(',',':'))
for k,par,fun in rows[::8]:print(k,'scale %.4f shift (%.1f,%.1f) cost %.2f %s'%(par[0],par[1],par[2],fun,'' if REL[k] else '(interpolated)'))
