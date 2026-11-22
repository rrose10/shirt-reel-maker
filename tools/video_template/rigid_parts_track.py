# NOTE: copy of the working script; see docstring. Env REG=regions (ref coords) of rigid parts, NOROT=1 to forbid rotation.
"""Rigid-part direct tracker: ref->frame similarity using the white mannequin head/neck + base (never covered by the hands).
usage: rigid.py REF  (regions in ref coords given by env REG='x0,y0,x1,y1;...')"""
import cv2,numpy as np,glob,sys,os
ref=int(sys.argv[1]);REG=[[int(v) for v in r.split(',')] for r in os.environ['REG'].split(';')]
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy');H0=np.load('Ms_bg.npy')
S=np.diag([.5,.5,1.0])
def prep(k):
    im=cv2.imread(fs[k]);v=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[...,2].astype(np.float32)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    E=cv2.GaussianBlur(cv2.resize(g,(360,640),interpolation=cv2.INTER_AREA),(0,0),1.0)
    sk=cv2.resize(cv2.dilate(SK[k],np.ones((25,25),np.uint8)),(360,640),interpolation=cv2.INTER_NEAREST)
    valid=(sk==0).astype(np.uint8);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
    return E,valid
E0,_=prep(ref)
band=np.zeros((1280,720),np.uint8)
for x0,y0,x1,y1 in REG:band[y0:y1,x0:x1]=1
band=cv2.resize(band,(360,640),interpolation=cv2.INTER_NEAREST)
def sim(Mf,c):
    L=Mf[:2,:2];s=np.sqrt(abs(np.linalg.det(L)));U,_,Vt=np.linalg.svd(L);R=U@Vt
    if np.linalg.det(R)<0:R=U@np.diag([1,-1])@Vt
    if os.environ.get('NOROT'):R=np.eye(2)
    tc=(Mf@c)[:2];M=np.eye(3);M[:2,:2]=s*R;M[:2,2]=tc-s*R@c[:2];return M
def reg(k,Mi):
    Ek,vk=prep(k);W0=S@Mi@np.linalg.inv(S)
    bk=cv2.warpPerspective(band,W0,(360,640),flags=cv2.INTER_NEAREST);m=((bk>0)&(vk>0)).astype(np.uint8)
    if m.sum()<400:return None,0
    W=W0[:2].astype(np.float32).copy();cc=0;done=False
    for levels in (((.5,80),(1,60)),((1,100),)):
        W=W0[:2].astype(np.float32).copy()
        try:
            for sc,it in levels:
                Wc=W.copy();Wc[:,2]*=sc
                ta=cv2.resize(E0,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else E0
                tb=cv2.resize(Ek,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else Ek
                mm=cv2.resize(m,None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST) if sc!=1 else m
                cc,Wc=cv2.findTransformECC(ta,tb,Wc,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,it,1e-6),mm,5)
                W=Wc.copy();W[:,2]/=sc
            done=True;break
        except cv2.error as e:continue
    if not done:return None,0
    Mf=np.linalg.inv(S)@np.vstack([W,[0,0,1]])@S
    return Mf,cc
c0=np.array([360.,640.,1.])
res=[None]*N;ccs=np.zeros(N)
res[ref]=np.eye(3);ccs[ref]=1
for rng in (range(ref-1,-1,-1),):
    prev=np.eye(3)
    for k in rng:
        Mf,cc=reg(k,prev)
        if Mf is None or cc<.9:
            M2,c2=reg(k,H0[k]/H0[k][2,2])
            if M2 is not None and c2>cc:Mf,cc=M2,c2
        if Mf is None or cc<.6:
            res[k]=None;continue
        Mf=sim(Mf,c0);res[k]=Mf;ccs[k]=cc;prev=Mf
out=np.array([np.eye(3) if r is None else r for r in res]);ok=np.array([r is not None for r in res])
np.save('Ms_rigid.npy',out);np.save('rigid_ok.npy',ok);np.save('rigid_cc.npy',ccs)
def s(h):return np.sqrt(abs(np.linalg.det(h[:2,:2])/h[2,2]**2))
for k in range(0,N,6):print(k,ok[k],'cc %.2f'%ccs[k],'rigid %.3f bg %.3f'%(s(out[k]),s(H0[k])))
