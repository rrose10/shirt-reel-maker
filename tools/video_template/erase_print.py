"""Erase a coloured print from a WHITE shirt (print tracked by track_print.py -> Mp.npy). Hands stay.
Writes cl/NNN.png (clean plates) and skins.npy (hand mattes, 0-255)."""
import cv2,numpy as np,glob,os,sys
fs=sorted(glob.glob('fr/*.png'));N=len(fs);H,W=1280,720
Mp=np.load('Mp.npy')
x0,y0,x1,y1=[int(v) for v in sys.argv[1].split(',')]
ref=cv2.imread(fs[0]);hs=cv2.cvtColor(ref,cv2.COLOR_BGR2HSV)
# print mask in the reference frame: everything that is not plain shirt white inside the box, merged into blobs
m=(((hs[...,1]>50)|(hs[...,2]<160))).astype(np.uint8)
box=np.zeros((H,W),np.uint8);box[y0:y1,x0:x1]=1;m&=box
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
pm=np.zeros_like(m)
for c in cnt:
    if cv2.contourArea(c)>800:cv2.drawContours(pm,[c],-1,1,-1)
pm=cv2.dilate(pm,np.ones((13,13),np.uint8))
cv2.imwrite('printmask_ref.png',pm*255)
def skin_raw(im):
    """skin = a narrow chroma band in Lab (a~136, b~145 in this footage). The print's orange/rust/mustard have a,b = 152/174, 161/160, 134/177: well outside it."""
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    a=cv2.GaussianBlur(lab[...,1],(0,0),1.0);b=cv2.GaussianBlur(lab[...,2],(0,0),1.0)
    return ((a>125)&(a<148)&(b>135)&(b<156)&(hsv[...,1]>42)&(hsv[...,2]>60)).astype(np.uint8)
def skin_clean(s):
    s=cv2.morphologyEx(s,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    n,lab_,st,_=cv2.connectedComponentsWithStats(s,connectivity=8);keep=np.zeros(n,np.uint8);keep[1:]=(st[1:,cv2.CC_STAT_AREA]>=1500)
    s=keep[lab_]
    s=cv2.morphologyEx(s,cv2.MORPH_CLOSE,np.ones((13,13),np.uint8))
    cts,hier=cv2.findContours(s,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for c,hh in zip(cts,hier[0]):
            if hh[3]>=0 and cv2.contourArea(c)<6000:cv2.drawContours(s,[c],-1,1,-1)
    return s
def skin(im):return skin_clean(skin_raw(im))
def fill_hole(im,hole):
    """smooth fabric for a big hole: relax (Laplace) from coarse to fine with the known pixels held fixed"""
    unk=hole>0;val=(~unk).astype(np.float32)
    cur=None
    for sc in (48,24,12,6):
        h,w=H//sc,W//sc
        sm=cv2.resize(im*val[...,None],(w,h),interpolation=cv2.INTER_AREA);vm=cv2.resize(val,(w,h),interpolation=cv2.INTER_AREA)
        known=vm>0.999;hm=~known
        base=sm/np.maximum(vm[...,None],1e-4)
        if cur is None:
            cur=base.copy();cur[hm]=np.median(base[known],axis=0)
            for sg in (4,2,1):
                n=cv2.GaussianBlur(sm,(0,0),sg);d=cv2.GaussianBlur(vm,(0,0),sg);est=n/np.maximum(d[...,None],1e-4);u=hm&(d>0.02);cur[u]=est[u]
        else:
            up=cv2.resize(cur,(w,h),interpolation=cv2.INTER_LINEAR);cur=base.copy();cur[hm]=up[hm]
        for _ in range(500 if sc>=24 else 200):
            b2=cv2.blur(cur,(3,3));cur[hm]=b2[hm]
    return cv2.resize(cur,(W,H),interpolation=cv2.INTER_CUBIC)
THR=float(os.environ.get('THR','45'))
DIS=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);DIS.setPatchSize(12);DIS.setPatchStride(4);DIS.setVariationalRefinementIterations(10)
rng=np.random.default_rng(3)
def mkfold():
    r=np.random.default_rng(11);acc=np.zeros((H,W),np.float32)
    for sg,amp,ax in ((70,1.0,2.2),(34,.55,1.8),(16,.25,1.4)):
        n=r.normal(0,1,(H//4,W//4)).astype(np.float32)
        n=cv2.GaussianBlur(n,(0,0),sg/4/ax,sigmaY=sg/4*ax*0.8)   # long, mostly vertical creases
        n=cv2.resize(n,(W,H),interpolation=cv2.INTER_CUBIC);n/=max(n.std(),1e-6);acc+=n*amp
    return acc/acc.std()
FOLD=mkfold()
SKM=[(np.array([170,150,150],np.float32),np.eye(3,dtype=np.float32)*200)]
os.makedirs('cl',exist_ok=True);skins=[]
KS=[int(v) for v in os.environ['KS'].split(',')] if os.environ.get('KS') else None
for k,f in enumerate(fs):
    if KS is not None and k not in KS:
        skins.append(np.zeros((H,W),np.uint8));continue
    im=cv2.imread(f);f32=im.astype(np.float32)
    Mk=Mp[k]
    wm=cv2.warpPerspective(pm,Mk,(W,H),flags=cv2.INTER_NEAREST)
    sk_out=skin(im)
    wref=cv2.warpPerspective(ref,Mk,(W,H),flags=cv2.INTER_LINEAR)
    # hands: near the print compare every pixel with the print as it was (and with the skin colour of this very frame)
    zone=cv2.dilate(wm,np.ones((31,31),np.uint8))
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32);labr=cv2.cvtColor(wref,cv2.COLOR_BGR2LAB).astype(np.float32)
    labb=cv2.GaussianBlur(lab,(0,0),1.2);labrb=cv2.GaussianBlur(labr,(0,0),1.2)
    # the fabric moves a little where the hands push it: re-align the old print with optical flow before comparing
    g1=cv2.cvtColor(wref,cv2.COLOR_BGR2GRAY);g2=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    fl=DIS.calc(g2,g1,None)
    gx,gy=np.meshgrid(np.arange(W),np.arange(H))
    wref2=cv2.remap(wref,(gx+fl[...,0]).astype(np.float32),(gy+fl[...,1]).astype(np.float32),cv2.INTER_LINEAR)
    labr2=cv2.GaussianBlur(cv2.cvtColor(wref2,cv2.COLOR_BGR2LAB).astype(np.float32),(0,0),1.2)
    dp=np.sqrt(((labb-labr2)**2).sum(2))
    raw=skin_raw(im)
    labz=cv2.cvtColor(im,cv2.COLOR_BGR2LAB);hsz=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    strict=((labz[...,1]>134)&(hsz[...,1]>70)&(raw>0)).astype(np.uint8)   # true skin; the pale cast shadow of a hand on the fabric is not
    core=cv2.morphologyEx(strict&(zone>0),cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    cand=(raw&cv2.dilate(core,np.ones((9,9),np.uint8))&(zone>0)).astype(np.uint8)
    cand=cv2.morphologyEx(cand,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    seed=((sk_out>0)&(zone==0)).astype(np.uint8)
    allc=np.maximum(cand,seed).astype(np.uint8)
    n,lb,st,_=cv2.connectedComponentsWithStats(allc,connectivity=8)
    has=np.zeros(n,np.uint8);has[np.unique(lb[seed>0])]=1;has[0]=0
    hand=((has[lb]>0)&(zone>0)).astype(np.uint8)
    hand=cv2.morphologyEx(hand,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    labk=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32),(0,0),1.0)
    pc=((labk[...,2]>160)|(labk[...,1]>152)|((labk[...,2]<118)&(labk[...,0]<120))|((labk[...,1]<118)&(labk[...,0]<170))).astype(np.uint8)   # print colours (mustard / orange / rust / navy / teal), never skin
    hand=hand&(1-cv2.dilate(pc,np.ones((3,3),np.uint8)))
    hand=cv2.morphologyEx(hand,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    hand=(cv2.GaussianBlur(hand.astype(np.float32),(0,0),2.5)>0.5).astype(np.uint8)
    # print still visible (matches the re-aligned old print) but outside the tracked mask: tracking slack near stretched fabric
    hsr=cv2.cvtColor(wref2,cv2.COLOR_BGR2HSV)
    prlike=(((hsr[...,1]>50)|(hsr[...,2]<165))&(dp<THR*0.8)&(zone>0)).astype(np.uint8)
    wm2=cv2.remap(wm,(gx+fl[...,0]).astype(np.float32),(gy+fl[...,1]).astype(np.float32),cv2.INTER_NEAREST)
    wm=np.maximum(wm,wm2)
    extra=cv2.dilate(prlike,np.ones((13,13),np.uint8))
    sk=np.where(zone>0,hand,sk_out).astype(np.uint8)
    hsvi=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    rel=((lab[...,1]>121)&(lab[...,2]>127)&(hsvi[...,1]>28)&(hsvi[...,2]>50)&(zone>0)).astype(np.uint8)
    rel=cv2.morphologyEx(rel,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    skd=np.maximum(cv2.dilate(sk,np.ones((9,9),np.uint8)),cv2.dilate(rel,np.ones((9,9),np.uint8)))
    hole=(((cv2.dilate(wm,np.ones((21,21),np.uint8))>0)|(extra>0))).astype(np.uint8)
    hole=(cv2.GaussianBlur(cv2.morphologyEx(hole,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8)).astype(np.float32),(0,0),6)>0.5).astype(np.uint8)
    # fill uses only pixels that are neither print nor hand
    known=1-np.maximum(np.maximum(cv2.dilate(wm,np.ones((3,3),np.uint8)),extra),skd)
    base=f32.copy();base[known==0]=np.nan
    tmp=im.copy()
    # replace unknown with local mean so the relaxation sees valid data only
    fillsrc=f32.copy();unk=(known==0)
    fillsrc[unk]=0
    up=fill_hole(fillsrc,unk.astype(np.uint8))
    # make the fill meet its surroundings exactly: diffuse the mismatch measured on a thin ring of known pixels inwards
    ring=((cv2.dilate(hole,np.ones((31,31),np.uint8))>0)&(hole==0)&(known>0)).astype(np.float32)
    res=(f32-up)*ring[...,None]
    num=cv2.GaussianBlur(res,(0,0),14);den=cv2.GaussianBlur(ring,(0,0),14)
    corr=num/np.maximum(den[...,None],1e-3);corr[den<0.01]=0
    dt=cv2.distanceTransform((hole>0).astype(np.uint8),cv2.DIST_L2,5)
    up=up+corr*np.exp(-dt/110.0)[...,None]
    fold=cv2.warpPerspective(FOLD,Mk,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    lumref=cv2.GaussianBlur(up.mean(2),(0,0),20)
    up=up*(1+fold[...,None]*0.018*np.clip((lumref[...,None]-90)/120,0.3,1.0))
    g=rng.normal(0,1.0,(H,W)).astype(np.float32);g=cv2.GaussianBlur(g,(0,0),0.8)*2.4
    up=up+g[...,None]
    dtw=cv2.distanceTransform((wm==0).astype(np.uint8),cv2.DIST_L2,5)
    wcore=np.maximum((dtw<=7).astype(np.float32),(extra>0).astype(np.float32))
    wsoft=np.maximum(wcore,np.clip((26-dtw)/19.0,0,1)*(hole>0))
    soft=cv2.GaussianBlur(wsoft,(0,0),3.0)[...,None]
    out=f32*(1-soft)+up*soft
    hm=cv2.GaussianBlur(sk.astype(np.float32),(0,0),1.2)[...,None]
    out=out*(1-hm)+f32*hm
    if os.environ.get('DUMP')==str(k):np.savez('dump.npz',sk=sk,wm=wm,soft=soft[...,0],hm=hm[...,0],hole=hole,extra=extra,skd=skd,known=known)
    if os.environ.get('DBG'):
        dbg=im.copy();dbg[sk>0]=(0.5*dbg[sk>0]+[0,0,127]).astype(np.uint8);dbg[(hole>0)]=(0.5*dbg[(hole>0)]+[0,127,0]).astype(np.uint8);cv2.imwrite('dbg_%03d.png'%k,dbg)
    cv2.imwrite('cl/%03d.png'%k,np.clip(out,0,255).astype(np.uint8))
    skins.append(sk)
np.save('skins_bin.npy',np.array(skins))
print('done',N)
