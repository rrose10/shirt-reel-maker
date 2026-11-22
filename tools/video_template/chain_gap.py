"""Fill gaps of a fused track by frame-to-frame registration with loop closure.
For every run of frames the direct tracker was not sure about (good=False) between two confident frames a and b:
register k -> k-1 by ECC on the shirt/non-skin pixels (small steps, so even a 3x zoom with hands is fine), chain forward from a,
compare the chained result at b with the confident M[b] and spread the difference linearly over the run.
usage: chain_gap.py FUSED.npy GOOD.npy OUT.npy [MAXRUN=60]   (needs fr/NNN.png, skins_raw.npy;  env STH VTH)"""
import cv2,numpy as np,glob,sys,os
M=np.load(sys.argv[1]);good=np.load(sys.argv[2]);out=sys.argv[3];MAXRUN=int(sys.argv[4]) if len(sys.argv)>4 else 60
STH=int(os.environ.get('STH','50'));VTH=int(os.environ.get('VTH','110'))
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy');c0=np.array([360.,640.])
S=np.diag([.5,.5,1.]);cl=cv2.createCLAHE(3.0,(8,8))
cache={}
def prep(k):
    if k in cache:return cache[k]
    im=cv2.imread(fs[k]);g=cl.apply(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)).astype(np.float32)
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=((hsv[...,1]<STH)&(hsv[...,2]>VTH)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    if n>1:m=(lab==1+np.argmax(st[1:,4])).astype(np.uint8)
    m=cv2.erode(m,np.ones((21,21),np.uint8));m[cv2.dilate(SK[k],np.ones((25,25),np.uint8))>0]=0
    g=cv2.GaussianBlur(cv2.resize(g,(360,640),interpolation=cv2.INTER_AREA),(0,0),1.2);m=cv2.resize(m,(360,640),interpolation=cv2.INTER_NEAREST)
    m[:3]=0;m[-3:]=0;m[:,:3]=0;m[:,-3:]=0
    if len(cache)>6:cache.pop(next(iter(cache)))
    cache[k]=(g,m);return g,m
def rel(k0,k1):
    """similarity frame k0 coords -> frame k1 coords (full-res)"""
    g0,m0=prep(k0);g1,m1=prep(k1)
    W=np.eye(2,3,dtype=np.float32)
    # init: bg-free guess = identity, (consecutive frames); the zoom per frame is small
    back=m1
    mm=(m0>0).astype(np.uint8)
    for sc,it in ((.5,80),(1,60)):
        Wc=W.copy();Wc[:,2]*=sc
        ta=cv2.resize(g0,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else g0
        tb=cv2.resize(g1,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else g1
        mk=cv2.resize(mm,None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST) if sc!=1 else mm
        try:cc,Wc=cv2.findTransformECC(ta,tb,Wc,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,it,1e-6),mk,5)
        except cv2.error:return None
        W=Wc.copy();W[:,2]/=sc
    A=np.linalg.inv(S)@np.vstack([W,[0,0,1]])@S
    # project to similarity about the image centre
    L=A[:2,:2];s=np.sqrt(abs(np.linalg.det(L)));U,_,Vt=np.linalg.svd(L);R=U@Vt
    if np.linalg.det(R)<0:R=U@np.diag([1,-1])@Vt
    if os.environ.get('NOROT'):R=np.eye(2)
    t=A[:2,:2]@c0+A[:2,2];o=np.eye(3);o[:2,:2]=s*R;o[:2,2]=t-s*R@c0;return o
def dec(m):
    s=np.sqrt(abs(np.linalg.det(m[:2,:2])));r=np.arctan2(m[1,0],m[0,0]);p=m[:2,:2]@c0+m[:2,2];return np.array([np.log(s),r,p[0],p[1]])
def enc(v):
    s=np.exp(v[0]);c,sn=np.cos(v[1])*s,np.sin(v[1])*s;L=np.array([[c,-sn],[sn,c]]);o=np.eye(3);o[:2,:2]=L;o[:2,2]=v[2:]-L@c0;return o
R=M.copy();k=0
while k<N:
    if good[k]:k+=1;continue
    a=k-1
    while k<N and not good[k]:k+=1
    b=k
    if a<0 or b>=N or b-a>MAXRUN+1:print('skip run',a+1,b-1);continue
    # chain a -> b
    Mc={a:M[a]};ok=True
    for j in range(a+1,b+1):
        r=rel(j-1,j)
        if r is None:ok=False;break
        Mc[j]=r@Mc[j-1]
    if not ok:print('chain failed',a,b);continue
    e=dec(M[b])-dec(Mc[b])             # difference at the far anchor (params: log s, rot, px, py)
    for j in range(a+1,b):
        w=(j-a)/(b-a);R[j]=enc(dec(Mc[j])+w*e)
    print('run',a+1,b-1,'closure error: scale %.1f%%  pos (%.1f,%.1f) px'%((np.exp(e[0])-1)*100,e[2],e[3]),flush=True)
np.save(out,R)
