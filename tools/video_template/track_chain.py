"""Keyframe-chained direct tracker (collar / hanger / bust band).  M[k] = similarity ref -> frame k.
Every frame is registered by ECC to the NEAREST KEYFRAME (a frame of similar zoom already registered to the ref), so a 2.5x zoom is
never matched in one step; keyframes are renewed when the relative scale exceeds ~15 % or after KFLEN frames.
usage: track_chain.py REF   env: REG='x0,y0,x1,y1;...' (ref coords, areas that move rigidly with the shirt: collar/shoulders/hanger/bust), NOROT=1, KFLEN
needs fr/NNN.png, skins_raw.npy, Ms_bg.npy (bg track, only used as init).  writes Ms_chain.npy, chain_cc.npy"""
import cv2,numpy as np,glob,sys,os
ref=int(sys.argv[1]);REG=[[int(v) for v in r.split(',')] for r in os.environ['REG'].split(';')]
KFLEN=int(os.environ.get('KFLEN','14'));CCMIN=float(os.environ.get('CCMIN','.9'))
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy');H0=np.load('Ms_bg.npy')
S=np.diag([.5,.5,1.0]);c0=np.array([360.,640.,1.])
cache={}
def prep(k):
    if k in cache:return cache[k]
    im=cv2.imread(fs[k]);g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    E=cv2.GaussianBlur(cv2.resize(g,(360,640),interpolation=cv2.INTER_AREA),(0,0),1.0)
    sk=cv2.resize(cv2.dilate(SK[k],np.ones((25,25),np.uint8)),(360,640),interpolation=cv2.INTER_NEAREST)
    valid=(sk==0).astype(np.uint8);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
    if len(cache)>40:cache.pop(next(iter(cache)))
    cache[k]=(E,valid);return E,valid
band0=np.zeros((1280,720),np.uint8)
for x0,y0,x1,y1 in REG:band0[y0:y1,x0:x1]=1
def sim(M,c):
    L=M[:2,:2];s=np.sqrt(abs(np.linalg.det(L)));U,_,Vt=np.linalg.svd(L);R=U@Vt
    if np.linalg.det(R)<0:R=U@np.diag([1,-1])@Vt
    if os.environ.get('NOROT'):R=np.eye(2)
    tc=(M@c)[:2];o=np.eye(3);o[:2,:2]=s*R;o[:2,2]=tc-s*R@c[:2];return o
def reg(T,MT,k,Wi):
    """ECC from keyframe T (ref->T = MT) to frame k, init Wi (T coords -> k coords, full-res 3x3). returns (W, cc)"""
    ET,vT=prep(T);Ek,vk=prep(k)
    bandT=cv2.warpPerspective(cv2.resize(band0,(360,640),interpolation=cv2.INTER_NEAREST),S@MT@np.linalg.inv(S),(360,640),flags=cv2.INTER_NEAREST)
    W0=S@Wi@np.linalg.inv(S)
    bk=cv2.warpPerspective(bandT,W0,(360,640),flags=cv2.INTER_NEAREST)
    m=((bandT>0)&(vT>0)).astype(np.uint8)          # template-side mask (ECC mask applies to the template)
    m2=((bk>0)&(vk>0)).astype(np.uint8)
    # ECC only takes one mask (template coords): bring the frame-side validity back into template coords
    back=cv2.warpPerspective(m2,np.linalg.inv(W0),(360,640),flags=cv2.INTER_NEAREST)
    m=(m&back).astype(np.uint8)
    if m.sum()<500:return None,0
    for levels in (((.5,80),(1,60)),((1,100),)):
        W=W0[:2].astype(np.float32).copy()
        try:
            for sc,it in levels:
                Wc=W.copy();Wc[:,2]*=sc
                ta=cv2.resize(ET,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else ET
                tb=cv2.resize(Ek,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else Ek
                mm=cv2.resize(m,None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST) if sc!=1 else m
                cc,Wc=cv2.findTransformECC(ta,tb,Wc,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,it,1e-6),mm,5)
                W=Wc.copy();W[:,2]/=sc
            return np.linalg.inv(S)@np.vstack([W,[0,0,1]])@S,cc
        except cv2.error:continue
    return None,0
Hn=np.array([h/h[2,2] for h in H0]);Hn=np.array([h@np.linalg.inv(Hn[ref]) for h in Hn])
res=[None]*N;ccs=np.zeros(N);res[ref]=np.eye(3);ccs[ref]=1
for rng in (range(ref-1,-1,-1),range(ref+1,N)):
    T=ref;MT=np.eye(3);last=None;lastk=ref;vel=None;since=0
    for k in rng:
        step=1 if k>ref else -1
        # init candidates (T coords -> k coords)
        cands=[]
        if last is not None:cands.append(last@np.linalg.inv(MT)) if False else None
        Mprev=res[lastk] if res[lastk] is not None else MT
        cands.append(Mprev@np.linalg.inv(MT))                                  # no motion since last good frame
        cands.append(Mprev@np.linalg.inv(Hn[lastk])@Hn[k]@np.linalg.inv(MT))   # bg relative motion carried on
        if vel is not None:cands.append(vel@Mprev@np.linalg.inv(MT))            # constant velocity
        best=(None,0)
        for Wi in cands:
            Wk,cc=reg(T,MT,k,Wi)
            if Wk is not None and cc>best[1]:best=(sim(Wk,c0),cc)
            if best[1]>.985:break
        Wk,cc=best
        if Wk is None or cc<CCMIN:res[k]=None;continue
        Mk=sim(Wk@MT,c0);
        if res[lastk] is not None:vel=Mk@np.linalg.inv(res[lastk])
        res[k]=Mk;ccs[k]=cc;lastk=k;since+=1
        rel=np.sqrt(abs(np.linalg.det(Wk[:2,:2])))
        if since>=KFLEN or abs(np.log(rel))>np.log(1.15):T=k;MT=Mk;since=0
    print('dir done',flush=True)
ok=np.array([r is not None for r in res])
out=np.array([np.eye(3) if r is None else r for r in res])
np.save('Ms_chain.npy',out);np.save('chain_ok.npy',ok);np.save('chain_cc.npy',ccs)
def s_(h):return np.sqrt(abs(np.linalg.det(h[:2,:2])/h[2,2]**2))
for k in range(0,N,6):print(k,ok[k],'cc %.2f'%ccs[k],'chain %.3f bg %.3f'%(s_(out[k]),s_(Hn[k])))
