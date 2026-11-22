"""Camera/shirt tracking for blank-shirt videos (no print to follow).
Usage (in a folder with fr/NNN.png, 720x1280):  python3 track_gen.py [REF_FRAME]
Sequential SIFT + ECC between neighbouring frames (skin/hands masked out), accumulated into
M[k] = homography ref->frame k.  Writes track_raw.json (H per frame, 9 floats, frame k = M[k]) and skins.npy."""
import cv2,numpy as np,glob,json,sys
fs=sorted(glob.glob('fr/*.png'));N=len(fs)
def skin(im):
    ycc=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    m=((ycc[...,1]>135)&(ycc[...,1]<185)&(ycc[...,2]>80)&(ycc[...,2]<135)&(hsv[...,1]>35)&(hsv[...,2]>70)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    return m
clahe=cv2.createCLAHE(8,(8,8))
G=[];SK=[]
for f in fs:
    im=cv2.imread(f);G.append(clahe.apply(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)));SK.append(skin(im))
sift=cv2.SIFT_create(4000,contrastThreshold=.008)
bf=cv2.BFMatcher()
def feats(k):
    m=(cv2.dilate(SK[k],np.ones((41,41),np.uint8))==0).astype(np.uint8)*255
    return sift.detectAndCompute(G[k],m)
KP=[feats(k) for k in range(N)]
Ms=[np.eye(3)]
bad=0
for k in range(1,N):
    kp1,d1=KP[k];kp0,d0=KP[k-1]
    Hk=None
    if d1 is not None and d0 is not None and len(kp1)>8 and len(kp0)>8:
        mt=bf.knnMatch(d1,d0,k=2);good=[m for m,n in (x for x in mt if len(x)==2) if m.distance<.78*n.distance]
        if len(good)>=8:
            a=np.float32([kp1[m.queryIdx].pt for m in good]);b=np.float32([kp0[m.trainIdx].pt for m in good])
            Hk,inl=cv2.findHomography(a,b,cv2.RANSAC,2.5)
            if Hk is not None and inl.sum()<8:Hk=None
    if Hk is None:Hk=np.eye(3);bad+=1
    # ECC refine (template = frame k, input = frame k-1: warp maps k -> k-1)
    msk=((cv2.dilate(SK[k],np.ones((31,31),np.uint8))==0)&(cv2.dilate(SK[k-1],np.ones((31,31),np.uint8))==0)).astype(np.uint8)
    W=Hk.astype(np.float32)
    try:
        _,W=cv2.findTransformECC(cv2.GaussianBlur(G[k],(0,0),1.5),cv2.GaussianBlur(G[k-1],(0,0),1.5),W,cv2.MOTION_HOMOGRAPHY,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-6),msk,5)
    except cv2.error:pass
    Ms.append(np.linalg.inv(W.astype(np.float64))@Ms[-1]);Ms[-1]/=Ms[-1][2,2]
print('frames',N,'sift fallbacks',bad)
ref=int(sys.argv[1]) if len(sys.argv)>1 else 0
Rinv=np.linalg.inv(Ms[ref])
out=[(M@Rinv) for M in Ms]
np.save('Ms.npy',np.array(out));np.save('skins_raw.npy',np.array(SK))
for k in (0,N//4,N//2,3*N//4,N-1):print(k,np.round(out[k],3).tolist())
