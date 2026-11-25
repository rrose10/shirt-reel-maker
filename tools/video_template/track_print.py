"""Track the PRINT itself (high contrast, glued to the fabric) from a reference frame: SIFT + RANSAC homography per frame,
then ECC refinement on pixels that agree with the reference. Writes Mp.npy (ref -> frame k) and ok.npy."""
import cv2,numpy as np,glob,sys
fs=sorted(glob.glob('fr/*.png'));N=len(fs);REF=0
x0,y0,x1,y1=[int(v) for v in sys.argv[1].split(',')]
clahe=cv2.createCLAHE(3,(8,8))
G=[cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY) for f in fs]
box=np.zeros((1280,720),np.uint8);box[y0:y1,x0:x1]=255
sift=cv2.SIFT_create(3000,contrastThreshold=.01)
kr,dr=sift.detectAndCompute(G[REF],box)
bf=cv2.BFMatcher()
Ms=[None]*N;ninl=[0]*N
for k in range(N):
    if k==REF:Ms[k]=np.eye(3);ninl[k]=999;continue
    kk,dk=sift.detectAndCompute(G[k],box)
    H=None
    if dk is not None and len(kk)>10:
        mt=bf.knnMatch(dr,dk,k=2);good=[m for m,n in (x for x in mt if len(x)==2) if m.distance<.75*n.distance]
        if len(good)>=10:
            a=np.float32([kr[m.queryIdx].pt for m in good]);b=np.float32([kk[m.trainIdx].pt for m in good])
            H,inl=cv2.findHomography(a,b,cv2.RANSAC,2.0)
            if H is not None:
                ninl[k]=int(inl.sum())
                # sanity: near-identity (the camera barely moves)
                c=np.array([[x0,y0,1],[x1,y0,1],[x0,y1,1],[x1,y1,1]],float).T;q=H@c;q=q[:2]/q[2]
                if np.abs(q-c[:2]).max()>60 or ninl[k]<12:H=None
    Ms[k]=H
good=[k for k in range(N) if Ms[k] is not None]
print('tracked',len(good),'of',N)
# fill gaps by linear interpolation of the 8 params
P=np.array([np.zeros(8) if M is None else (M/M[2,2]).flatten()[:8] for M in Ms])
idx=np.array(good)
for j in range(8):P[:,j]=np.interp(np.arange(N),idx,P[idx,j])
Mp=[np.append(p,1).reshape(3,3) for p in P]
np.save('Mp.npy',np.array(Mp));np.save('ninl.npy',np.array(ninl))
print('weak frames (<12 inliers):',[k for k in range(N) if Ms[k] is None])
