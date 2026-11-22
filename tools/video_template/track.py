import cv2,numpy as np,glob,json
fs=sorted(glob.glob('fr/*.png'));N=len(fs);H,W=1280,720
def skin(im):
    ycc=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    m=((ycc[...,1]>135)&(ycc[...,1]<185)&(ycc[...,2]>80)&(ycc[...,2]<135)&(hsv[...,1]>35)&(hsv[...,2]>70)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    return m
def printmask(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    m=((hsv[...,2]>195)&(hsv[...,1]<55)).astype(np.uint8)
    # keep the main cluster (drop press / table specks): dilate, biggest components covering most area
    d=cv2.dilate(m,np.ones((31,31),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(d)
    best=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    keep=(lab==best).astype(np.uint8)
    return m*keep
ims=[cv2.imread(f) for f in fs]
P=[printmask(i) for i in ims];S=[skin(i) for i in ims]
ref=0
R=cv2.GaussianBlur(P[ref].astype(np.float32),(0,0),3)
Ws=[np.eye(3,dtype=np.float32)]
crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6)
ious=[1.0]
for t in range(1,N):
    T=cv2.GaussianBlur(P[t].astype(np.float32),(0,0),3)
    msk=(1-cv2.dilate(S[t],np.ones((25,25),np.uint8))).astype(np.uint8)
    # also ignore the image border (print cut by frame edge)
    W0=Ws[-1].copy()
    ok=False
    for sc in (4,2,1):
        pass
    try:
        _,Wt=cv2.findTransformECC(T,R,W0,cv2.MOTION_HOMOGRAPHY,crit,msk,5)
        ok=True
    except cv2.error:
        Wt=W0
    Ws.append(Wt.astype(np.float32))
    # iou: warp ref print to frame t (WARP_INVERSE_MAP: dst(x)=src(W x))
    wp=cv2.warpPerspective(P[ref],Wt,(W,H),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)
    vis=(msk>0)
    inter=((wp>0)&(P[t]>0)&vis).sum();uni=(((wp>0)|(P[t]>0))&vis).sum()
    ious.append(float(inter/max(uni,1)))
    if not ok: print('ECC fail',t)
ious=np.array(ious);print('mean IoU',ious.mean(),'min',ious.min(),'worst',[(i,round(float(ious[i]),2)) for i in np.argsort(ious)[:10]])
np.save('Ws.npy',np.array(Ws));np.save('ious.npy',ious)
