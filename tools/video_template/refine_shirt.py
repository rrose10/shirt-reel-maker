"""Second tracking stage: make the print follow the SHIRT itself, not the room/background.
Each frame's shirt silhouette (dark, non-skin, largest blob) is aligned straight to the reference frame's silhouette
with ECC (no chaining, so no drift), starting from the Ms.npy of track_gen.py.  Keeps Ms_bg.npy as backup.
Usage: python3 refine_shirt.py REF_FRAME [VMAX=50] [homography|affine]"""
import cv2,numpy as np,glob,sys
ref=int(sys.argv[1]);VMAX=int(sys.argv[2]) if len(sys.argv)>2 else 50
MODE=cv2.MOTION_AFFINE if (len(sys.argv)>3 and sys.argv[3]=='affine') else cv2.MOTION_HOMOGRAPHY
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy');Ms=np.load('Ms.npy');np.save('Ms_bg.npy',Ms)
S=np.diag([.5,.5,1.0])
def shape(k):
    im=cv2.imread(fs[k]);v=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[...,2]
    d=((v<VMAX)&(cv2.dilate(SK[k],np.ones((7,7),np.uint8))==0)).astype(np.uint8)
    d=cv2.morphologyEx(d,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(d)
    if n>1:d=(lab==1+np.argmax(st[1:,cv2.CC_STAT_AREA])).astype(np.uint8)
    valid=(cv2.dilate(SK[k],np.ones((25,25),np.uint8))==0).astype(np.uint8)
    valid[:6]=0;valid[-6:]=0;valid[:,:6]=0;valid[:,-6:]=0
    f=cv2.GaussianBlur(cv2.resize(d.astype(np.float32),(360,640),interpolation=cv2.INTER_AREA),(0,0),4)
    return f,cv2.resize(valid,(360,640),interpolation=cv2.INTER_NEAREST),d
Fr,_,Dr=shape(ref);new=[];ious=[];old_ious=[]
def iou(M,d,valid):
    w=cv2.warpPerspective(d,np.linalg.inv(M),(720,1280),flags=cv2.INTER_NEAREST)  # frame -> ref
    v=cv2.warpPerspective(valid,np.linalg.inv(M),(720,1280),flags=cv2.INTER_NEAREST)>0
    a=(w>0)&v;b=(Dr>0)&v
    return (a&b).sum()/max(1,(a|b).sum())
for k in range(N):
    Fk,vk,Dk=shape(k);M0=Ms[k]
    W0=(S@np.linalg.inv(M0)@np.linalg.inv(S)).astype(np.float32)   # half-res: frame k -> ref
    W=W0.copy()
    try:
        _,W=cv2.findTransformECC(Fk,Fr,W,MODE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,1e-6),vk,5)
    except cv2.error:pass
    if MODE==cv2.MOTION_AFFINE:W=np.vstack([W[:2],[0,0,1]]).astype(np.float32)
    M1=np.linalg.inv(np.linalg.inv(S)@W.astype(np.float64)@S);M1/=M1[2,2]
    vfull=cv2.resize(vk,(720,1280),interpolation=cv2.INTER_NEAREST)
    i0,i1=iou(M0,Dk,vfull),iou(M1,Dk,vfull);old_ious.append(i0);ious.append(i1)
    new.append(M1 if i1>=i0-0.01 else M0)   # never accept a worse alignment
print('shirt-silhouette IoU  before: mean %.3f min %.3f   after: mean %.3f min %.3f'%(np.mean(old_ious),np.min(old_ious),np.mean(ious),np.min(ious)))
np.save('Ms.npy',np.array(new));np.save('iou_before.npy',np.array(old_ious));np.save('iou_after.npy',np.array(ious))
bad=[k for k in range(N) if old_ious[k]<.9];print('frames with weak alignment before:',len(bad),bad[:40])
