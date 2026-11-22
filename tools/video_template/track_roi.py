import cv2,numpy as np,glob,sys,json
REF=int(sys.argv[1]);ROI=[int(v) for v in sys.argv[2].split(',')];THR=int(sys.argv[3]) if len(sys.argv)>3 else 150
fs=sorted(glob.glob('fr/*.png'));N=len(fs);H,W=1280,720
ims=[cv2.imread(f) for f in fs]
def skin(im):
    ycc=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    m=((ycc[...,1]>135)&(ycc[...,1]<185)&(ycc[...,2]>80)&(ycc[...,2]<135)&(hsv[...,1]>35)&(hsv[...,2]>70)).astype(np.uint8)
    return cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
def bright(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);mx=im.max(axis=2).astype(np.int16);mn=im.min(axis=2).astype(np.int16)
    return ((hsv[...,2]>THR)&((mx-mn)<30)).astype(np.uint8)
S=[skin(i) for i in ims];B=[bright(i) for i in ims]
roi0=np.zeros((H,W),np.uint8);x0,y0,x1,y1=ROI;roi0[y0:y1,x0:x1]=1
Pref=B[REF]*roi0
n,lab,st,_=cv2.connectedComponentsWithStats(cv2.dilate(Pref,np.ones((31,31),np.uint8)))
best=1+int(np.argmax(st[1:,4]));Pref=Pref*(lab==best)
ys,xs=np.where(Pref>0);bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())];print('ref bbox',bbox)
Rb=cv2.GaussianBlur(Pref.astype(np.float32),(0,0),3);regref=cv2.dilate(Pref,np.ones((61,61),np.uint8))
crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,300,1e-7)
Ws=[None]*N;Ws[REF]=np.eye(3,dtype=np.float32);P=[None]*N;P[REF]=Pref;ious=np.zeros(N);ious[REF]=1
def step(t,prev):
    roi=cv2.warpPerspective(regref,prev,(W,H),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)   # ref region -> frame t (approx. via previous W)
    skd=cv2.dilate(S[t],np.ones((15,15),np.uint8))
    Pt=(B[t]*roi*(1-skd)).astype(np.uint8)
    T=cv2.GaussianBlur(Pt.astype(np.float32),(0,0),3);msk=(1-skd).astype(np.uint8)
    best=(-1,prev)
    for init in (prev,):
        for sig in (8,3):
            Tt=cv2.GaussianBlur(Pt.astype(np.float32),(0,0),sig);Rr=cv2.GaussianBlur(Pref.astype(np.float32),(0,0),sig)
            try:
                _,w=cv2.findTransformECC(Tt,Rr,init.copy(),cv2.MOTION_HOMOGRAPHY,crit,msk,5);init=w.astype(np.float32)
            except cv2.error:pass
        wp=cv2.warpPerspective(Pref,init,(W,H),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)
        vis=msk>0;i=((wp>0)&(Pt>0)&vis).sum();u=(((wp>0)|(Pt>0))&vis).sum();best=(i/max(u,1),init)
    return Pt,best
for rng in (range(REF+1,N),range(REF-1,-1,-1)):
    prev=np.eye(3,dtype=np.float32)
    for t in rng:
        Pt,(iou,w)=step(t,prev);Ws[t]=w;P[t]=Pt;ious[t]=iou;prev=w
print('mean IoU',ious.mean(),'worst',[(int(i),round(float(ious[i]),2)) for i in np.argsort(ious)[:8]])
np.save('P.npy',np.array(P));np.save('S.npy',np.array(S));np.save('Ws.npy',np.array(Ws));np.save('ious.npy',ious)
json.dump({'ref':REF,'bbox':bbox,'thr':THR},open('meta.json','w'))
