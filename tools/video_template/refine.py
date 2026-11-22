import cv2,numpy as np,glob
src=open('track.py').read().split('ims=')[0]
exec(src)
ims=[cv2.imread(f) for f in fs]
P=[printmask(i) for i in ims];S=[skin(i) for i in ims]
Ws=np.load('Ws.npy');ious=np.load('ious.npy')
R0=P[0].astype(np.float32)
crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,300,1e-7)
def iou(t,Wt):
    msk=(1-cv2.dilate(S[t],np.ones((25,25),np.uint8)))>0
    wp=cv2.warpPerspective(P[0],Wt,(W,H),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)
    i=((wp>0)&(P[t]>0)&msk).sum();u=(((wp>0)|(P[t]>0))&msk).sum();return i/max(u,1)
def ecc(t,init,sig):
    T=cv2.GaussianBlur(P[t].astype(np.float32),(0,0),sig);R=cv2.GaussianBlur(R0,(0,0),sig)
    msk=(1-cv2.dilate(S[t],np.ones((25,25),np.uint8))).astype(np.uint8)
    _,Wt=cv2.findTransformECC(T,R,init.copy(),cv2.MOTION_HOMOGRAPHY,crit,msk,5);return Wt.astype(np.float32)
N=len(Ws)
for it in range(3):
    for t in range(1,N-1):
        if ious[t]>0.85:continue
        cands=[Ws[t],(Ws[t-1]+Ws[t+1])/2,Ws[t-1],Ws[t+1]]
        best=(ious[t],Ws[t])
        for c in cands:
            try:
                w=ecc(t,c.astype(np.float32),10);w=ecc(t,w,4);w=ecc(t,w,2)
            except cv2.error:continue
            v=iou(t,w)
            if v>best[0]:best=(v,w)
        ious[t]=best[0];Ws[t]=best[1]
    print(it,'mean',ious.mean(),'min',ious.min(),[(i,round(float(ious[i]),2)) for i in np.argsort(ious)[:6]])
np.save('Ws2.npy',Ws)
