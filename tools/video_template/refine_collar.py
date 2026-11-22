"""Better tracking for dark shirts (run after track_gen.py, before build_blank.py): re-registers every frame DIRECTLY to the reference frame (no chaining, so no drift) on the collar and shoulders only, hands masked out, with a pure similarity (no stretching or shear). Keeps Ms_bg.npy. Writes Ms_direct.npy, copy it to Ms.npy. usage: python3 refine_collar.py REF x,y,w,h sim BAND_BELOW_PRINT_TOP(40) [BAND_TOP_Y]
import cv2,numpy as np,glob,sys
ref=int(sys.argv[1]);bx,by,bw,bh=[int(v) for v in sys.argv[2].split(',')];mode=sys.argv[3] if len(sys.argv)>3 else 'affine'
MODE=cv2.MOTION_AFFINE
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy');H0=np.load('Ms_bg.npy') if __import__('os').path.exists('Ms_bg.npy') else np.load('Ms.npy')
if not __import__('os').path.exists('Ms_bg.npy'):np.save('Ms_bg.npy',H0)
S=np.diag([.5,.5,1.0])
def prep(k):
    im=cv2.imread(fs[k]);v=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[...,2].astype(np.float32)
    E=cv2.GaussianBlur(cv2.resize(np.clip(v*3.5,0,255),(360,640),interpolation=cv2.INTER_AREA),(0,0),1.2)
    sk=cv2.resize(cv2.dilate(SK[k],np.ones((21,21),np.uint8)),(360,640),interpolation=cv2.INTER_NEAREST)
    valid=(sk==0).astype(np.uint8);valid[:4]=0;valid[-4:]=0;valid[:,:4]=0;valid[:,-4:]=0
    return E,valid
E0,v0=prep(ref)
# ref-space band: shirt top .. print bottom + margin
r=cv2.imread(fs[ref]);vr=cv2.cvtColor(r,cv2.COLOR_BGR2HSV)[...,2]
ys=np.where((vr<60).sum(1)>30)[0];top=max(0,ys.min()-40)
if len(sys.argv)>5:top=int(sys.argv[5])
band=np.zeros((1280,720),np.uint8);band[top:by+int(sys.argv[4]) if len(sys.argv)>4 else by+80,:]=1
band=cv2.resize(band,(360,640),interpolation=cv2.INTER_NEAREST)
def refine(k,Hk,Cprev):
    """Hk: bg init; Cprev: 3x3 correction carried from the neighbouring frame (full-res). returns (M or None)"""
    Ek,vk=prep(k)
    Mi=Cprev@(Hk/Hk[2,2]);W0=(S@Mi@np.linalg.inv(S))
    bk=cv2.warpPerspective(band,W0,(360,640),flags=cv2.INTER_NEAREST)
    m=(bk>0)&(vk>0)
    W=W0[:2].astype(np.float32).copy()
    try:
        for sc,it in ((.25,50),(.5,50),(1,40)):
            Wc=W.copy();Wc[:,2]*=sc
            ta=cv2.resize(E0,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else E0
            tb=cv2.resize(Ek,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc!=1 else Ek
            mm=cv2.resize(m.astype(np.uint8),None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST) if sc!=1 else m.astype(np.uint8)
            cc,Wc=cv2.findTransformECC(ta,tb,Wc,MODE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,it,1e-6),mm,5)
            W=Wc.copy();W[:,2]/=sc
    except cv2.error:return None,Mi
    Wf=np.vstack([W,[0,0,1]]);Mf=np.linalg.inv(S)@Wf@S
    if mode=='sim':
        cc_=np.array([bx+bw/2,by+bh/2,1.]);L=Mf[:2,:2];s_=np.sqrt(abs(np.linalg.det(L)));U,_,Vt=np.linalg.svd(L);R=U@Vt
        if np.linalg.det(R)<0:R=U@np.diag([1,-1])@Vt
        tc=(Mf@cc_)[:2];Mf=np.eye(3);Mf[:2,:2]=s_*R;Mf[:2,2]=tc-s_*R@cc_[:2]
    C=np.float32([[bx,by],[bx+bw,by],[bx+bw,by+bh],[bx,by+bh]]).reshape(-1,1,2)
    a=cv2.perspectiveTransform(C,Mi).reshape(-1,2);b=cv2.perspectiveTransform(C,Mf).reshape(-1,2)
    dev=np.abs(a-b).max();sw=np.hypot(*(b[1]-b[0]))/np.hypot(*(a[1]-a[0]))
    if dev>45 or not(.88<sw<1.12):return None,Mi
    return Mf,Mi
res=[None]*N;acc=0
for rng in (range(ref,N),range(ref-1,-1,-1)):
    Cprev=np.eye(3)
    for k in rng:
        Hk=H0[k];Mf,Mi=refine(k,Hk,Cprev)
        if Mf is not None:
            res[k]=Mf;acc+=1;Cprev=Mf@np.linalg.inv(Hk/Hk[2,2])
        else:res[k]=Mi
    if rng.start==ref:pass
out=res;info=[]
print('accepted',acc,'/',N)
Ms=np.array(out);np.save('Ms_direct.npy',Ms)

