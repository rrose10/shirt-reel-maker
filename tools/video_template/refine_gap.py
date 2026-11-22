"""(refine-around-a-guess variant) Scale-scan direct tracker (collar / hanger / bust band) for shirts that zoom a lot and get touched by hands.
For every frame it tries a range of zoom values around the neighbour's, aligns the reference band by ECC-translation at each,
keeps the best correlation, then refines (affine->similarity).  M[k] = similarity ref -> frame k.
usage: track_scan.py REF  env: REG='x0,y0,x1,y1;...'  NOROT=1  SPAN (log-scale half range around the neighbour, default .10)
needs fr/NNN.png, skins_raw.npy, Ms_bg.npy; writes Ms_scan.npy scan_cc.npy"""
import cv2,numpy as np,glob,sys,os
ref=int(sys.argv[1]);GUESS=np.load(sys.argv[2]);FRAMES=[int(v) for v in sys.argv[3].split(',')] if len(sys.argv)>3 else list(range(len(GUESS)));REG=[[int(v) for v in r.split(',')] for r in os.environ['REG'].split(';')]
SPAN=float(os.environ.get('SPAN','.10'));NOROT=bool(os.environ.get('NOROT'))
fs=sorted(glob.glob('fr/*.png'));N=len(fs);SK=np.load('skins_raw.npy');H0=np.load('Ms_bg.npy')
S=np.diag([.5,.5,1.0]);c0=np.array([360.,640.,1.]);Wd,Hd=360,640
def prep(k):
    im=cv2.imread(fs[k]);g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    g=cv2.resize(g,(Wd,Hd),interpolation=cv2.INTER_AREA)
    sk=cv2.resize(cv2.dilate(SK[k],np.ones((25,25),np.uint8)),(Wd,Hd),interpolation=cv2.INTER_NEAREST)
    valid=(sk==0).astype(np.uint8);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
    return g,valid
G0,v0=prep(ref)
band=np.zeros((1280,720),np.uint8)
for x0,y0,x1,y1 in REG:band[y0:y1,x0:x1]=1
band=cv2.resize(band,(Wd,Hd),interpolation=cv2.INTER_NEAREST)&v0
ys,xs=np.where(band>0);cb=np.array([xs.mean(),ys.mean()])      # band centre (half-res ref coords)
def sim_full(M,c):
    L=M[:2,:2];s=np.sqrt(abs(np.linalg.det(L)));U,_,Vt=np.linalg.svd(L);R=U@Vt
    if np.linalg.det(R)<0:R=U@np.diag([1,-1])@Vt
    if NOROT:R=np.eye(2)
    tc=(M@c)[:2];o=np.eye(3);o[:2,:2]=s*R;o[:2,2]=tc-s*R@c[:2];return o
def blur_for(g,s):
    sg=max(0,(s-1)*.9) if s>1 else 0
    return cv2.GaussianBlur(g,(0,0),sg+1.0)
def try_scale(Gk,vk,s,tpred):
    """ref band scaled by s about its centre and shifted to tpred -> ECC translation. returns (cc, M_half (2x3 ref->frame at half-res))"""
    A=np.array([[s,0,tpred[0]-s*cb[0]],[0,s,tpred[1]-s*cb[1]]],np.float32)
    T=cv2.warpAffine(blur_for(G0,1.0),A,(Wd,Hd),flags=cv2.INTER_LINEAR,borderValue=0)
    mb=cv2.warpAffine(band.astype(np.uint8),A,(Wd,Hd),flags=cv2.INTER_NEAREST)
    m=(mb>0)&(vk>0)
    if m.sum()<600:return 0,None
    F=blur_for(Gk,s)
    Wt=np.eye(2,3,dtype=np.float32)
    try:
        cc,Wt=cv2.findTransformECC(T,F,Wt,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-5),m.astype(np.uint8),5)
    except cv2.error:return 0,None
    M=np.vstack([A,[0,0,1]]);M[:2,2]+=Wt[:,2]
    return cc,M
def refine(Gk,vk,M):
    A=M[:2].astype(np.float32).copy()
    mb=cv2.warpAffine(band.astype(np.uint8),A,(Wd,Hd),flags=cv2.INTER_NEAREST);m=((mb>0)&(vk>0)).astype(np.uint8)
    s=np.sqrt(abs(np.linalg.det(M[:2,:2])))
    # template = ref blurred, in REF coords; ECC warp maps template->input
    Wf=A.copy()
    try:
        cc,Wf=cv2.findTransformECC(blur_for(G0,1.0),blur_for(Gk,s),Wf,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,50,1e-6),cv2.warpAffine(m,cv2.invertAffineTransform(Wf),(Wd,Hd),flags=cv2.INTER_NEAREST)&band.astype(np.uint8),5)
    except cv2.error:return None,0
    return np.vstack([Wf,[0,0,1]]),cc

out=GUESS.copy();ccs=np.zeros(len(GUESS))
for k in FRAMES:
    Gk,vk=prep(k)
    Mg=S@GUESS[k]@np.linalg.inv(S)                      # half-res ref->frame
    sg=np.sqrt(abs(np.linalg.det(Mg[:2,:2])));pt=Mg[:2,:2]@cb+Mg[:2,2]
    best=(0,None)
    for d in np.linspace(-SPAN,SPAN,int(SPAN*200)+1):
        for dx,dy in ((0,0),(6,0),(-6,0),(0,6),(0,-6)):
            cc,M=try_scale(Gk,vk,sg*np.exp(d),pt+np.array([dx,dy]))
            if cc>best[0]:best=(cc,M)
    cc,M=best
    if M is None:continue
    M2,cc2=refine(Gk,vk,M)
    if M2 is not None and cc2>cc:M,cc=M2,cc2
    Mf=sim_full(np.linalg.inv(S)@M@S,c0);out[k]=Mf;ccs[k]=cc
    print(k,'cc %.3f  scale %.3f (guess %.3f)'%(cc,np.sqrt(abs(np.linalg.det(Mf[:2,:2]))),sg),flush=True)
np.save('Ms_gap.npy',out);np.save('gap_cc.npy',ccs)
