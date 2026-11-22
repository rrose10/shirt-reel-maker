"""Dense-flow chain on the SHIRT pixels only (no background -> no parallax error).  Relative similarity between neighbouring frames
by RANSAC on a DIS optical-flow field restricted to the shirt silhouette (hands excluded), chained outwards from REF.
usage: track_flow.py REF   env STH VTH (shirt = sat<STH & val>VTH)  writes Ms_flow.npy (ref->frame k, full-res 3x3 similarity), flow_n.npy (inliers)"""
import cv2,numpy as np,glob,sys,os
ref=int(sys.argv[1]);STH=int(os.environ.get('STH','50'));VTH=int(os.environ.get('VTH','110'))
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy')
SC=.5;Wd,Hd=int(720*SC),int(1280*SC)
def silmask(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=((hsv[...,1]<STH)&(hsv[...,2]>VTH)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    if n>1:m=(lab==1+np.argmax(st[1:,4])).astype(np.uint8)
    return m
def load(k):
    im=cv2.imread(fs[k]);g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    g=cv2.createCLAHE(3.0,(8,8)).apply(g)
    m=silmask(im);m[cv2.dilate(SK[k],np.ones((21,21),np.uint8))>0]=0
    m=cv2.erode(m,np.ones((9,9),np.uint8))
    return cv2.resize(g,(Wd,Hd),interpolation=cv2.INTER_AREA),cv2.resize(m,(Wd,Hd),interpolation=cv2.INTER_NEAREST)
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
def rel(ga,ma,gb,mb):
    """similarity mapping frame-a coords -> frame-b coords (full-res px)"""
    fl=dis.calc(ga,gb,None)                       # a -> b
    ys,xs=np.where((ma>0)&(mb>0))
    if len(xs)<400:return None,0
    sel=np.arange(0,len(xs),max(1,len(xs)//4000));xs,ys=xs[sel],ys[sel]
    pa=np.stack([xs,ys],1).astype(np.float32);pb=pa+fl[ys,xs]
    A,inl=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=.6,maxIters=2000,confidence=.999)
    if A is None:return None,0
    M=np.vstack([A,[0,0,1]]);S=np.diag([SC,SC,1.]);return np.linalg.inv(S)@M@S,int(inl.sum())
res=[None]*N;ns=np.zeros(N);res[ref]=np.eye(3)
for rng,step in ((range(ref-1,-1,-1),+1),(range(ref+1,N),-1)):
    prevk=ref;gp,mp=load(ref);M=np.eye(3)
    for k in rng:
        gk,mk=load(k)
        R,n=rel(gp,mp,gk,mk)                 # prev-frame coords -> frame k coords
        if R is None:R=np.eye(3)
        M=R@M;res[k]=M.copy();ns[k]=n;gp,mp=gk,mk
        if k%20==0:print(k,n,'scale %.3f'%np.sqrt(abs(np.linalg.det(M[:2,:2]))),flush=True)
np.save('Ms_flow.npy',np.array(res));np.save('flow_n.npy',ns)
