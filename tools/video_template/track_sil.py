"""Silhouette tracker for bright shirts on a coloured / dark background (white tees, hanger, flat lay, bust).
Finds, per frame, the similarity (scale, rotation, translation) that makes the REFERENCE frame's shirt silhouette overlap
the silhouette of frame k (hands masked out), upper part of the shirt only (ROI), chained from the neighbouring frame.
usage: track_sil.py REF   env: STH,VTH (shirt = saturation<STH & value>VTH), YMAX (ref y limit of the ROI), ROT=0 to forbid rotation
needs fr/NNN.png and skins_raw.npy (from track_gen.py); writes Ms_sil.npy, sil_cost.npy"""
import cv2,numpy as np,glob,sys,os
from scipy.optimize import minimize
ref=int(sys.argv[1]);STH=int(os.environ.get('STH','50'));VTH=int(os.environ.get('VTH','110'));YMAX=int(os.environ.get('YMAX','700'));YMIN=int(os.environ.get('YMIN','0'))
ROT=os.environ.get('ROT','1')!='0'
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy')
SC=.25;Wd,Hd=int(720*SC),int(1280*SC);c0=np.array([360.,640.])
def silmask(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=((hsv[...,1]<STH)&(hsv[...,2]>VTH)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    if n>1:m=(lab==1+np.argmax(st[1:,4])).astype(np.uint8)
    # fill holes
    cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);f=np.zeros_like(m);cv2.drawContours(f,cnt,-1,1,-1)
    return f
def prep(k):
    im=cv2.imread(fs[k]);m=silmask(im)
    sk=cv2.dilate(SK[k],np.ones((25,25),np.uint8))
    m=np.where(sk>0,0,m).astype(np.float32)   # hands never count as shirt ...
    E=cv2.GaussianBlur(cv2.resize(m,(Wd,Hd),interpolation=cv2.INTER_AREA),(0,0),1.6)
    valid=(cv2.resize(sk,(Wd,Hd),interpolation=cv2.INTER_NEAREST)==0).astype(np.float32)  # ... and are ignored
    return E,valid
E0,v0=prep(ref)
roi=np.zeros((Hd,Wd),np.float32);roi[int(YMIN*SC):int(YMAX*SC),:]=1
def Mof(p):
    ls,rot,tx,ty=p;s=np.exp(ls);cr,sr=np.cos(rot)*s,np.sin(rot)*s
    M=np.array([[cr,-sr,0],[sr,cr,0],[0,0,1.]]);M[:2,2]=c0+np.array([tx,ty])-M[:2,:2]@c0;return M
Sx=np.diag([SC,SC,1.])
def warp(M):
    A=(Sx@M@np.linalg.inv(Sx))[:2];return cv2.warpAffine(E0,A,(Wd,Hd),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
NREF=float((roi*E0).sum())+1.0
def cost_fn(z,Ek,vk):
    p=[z[0]/100.,z[1]/100.,z[2],z[3]]
    if not ROT:p[1]=0
    M=Mof(p);Wm=warp(M)
    A=(Sx@M@np.linalg.inv(Sx))[:2];r=cv2.warpAffine(roi,A,(Wd,Hd),flags=cv2.INTER_NEAREST)
    w=r*vk
    return float(((Wm-Ek)**2*w).sum()/NREF)
def fit(k,p0):
    Ek,vk=prep(k)
    z0=np.array([p0[0]*100,p0[1]*100,p0[2],p0[3]],float)
    lo=z0-np.array([8,8,40,40.]);hi=z0+np.array([8,8,40,40.])
    if not ROT:lo[1]=hi[1]=z0[1]
    sim=np.array([z0,z0+[2,0,0,0],z0+[0,2,0,0],z0+[0,0,4,0],z0+[0,0,0,4]])
    res=minimize(cost_fn,z0,args=(Ek,vk),method='Nelder-Mead',bounds=list(zip(lo,hi)),options={'initial_simplex':sim,'xatol':.02,'fatol':1e-7,'maxiter':500})
    z=res.x;return np.array([z[0]/100.,z[1]/100.,z[2],z[3]]),res.fun
H0=np.load('Ms_bg.npy') if os.path.exists('Ms_bg.npy') else None
def p_from_H(H):
    H=H/H[2,2];L=H[:2,:2];s=np.sqrt(abs(np.linalg.det(L)));rot=np.arctan2(L[1,0],L[0,0]);t=(H@np.array([c0[0],c0[1],1.]));t=t[:2]/t[2]
    return np.array([np.log(s),rot,t[0]-c0[0],t[1]-c0[1]])
P=[None]*N;C=np.zeros(N);P[ref]=np.array([0,0,0,0.]);C[ref]=0
for rng in (range(ref-1,-1,-1),range(ref+1,N)):
    prev=np.zeros(4)
    for k in rng:
        cands=[prev]+([p_from_H(H0[k])] if H0 is not None else [])
        bestp=None;bestc=1e9
        for p0 in cands:
            p,c=fit(k,p0)
            if c<bestc:bestc,bestp=c,p
        P[k]=bestp;C[k]=bestc;prev=bestp
        if k%20==0:print(k,'cost %.4f scale %.3f rot %.2f t(%.1f,%.1f)'%(bestc,np.exp(bestp[0]),np.degrees(bestp[1]),bestp[2],bestp[3]),flush=True)
Ms=np.array([Mof(p) for p in P]);np.save('Ms_sil.npy',Ms);np.save('sil_cost.npy',C);np.save('sil_P.npy',np.array(P))
