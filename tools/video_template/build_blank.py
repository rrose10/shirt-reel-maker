"""Template from a video of a BLANK shirt filmed with a fixed camera (nothing to remove or track).
Usage (in a work folder that has fr/NNN.png at 720x1280):  python3 build_blank.py FPS x,y,w,h
 x,y,w,h = the print area in frame pixels. Then run pack.py."""
import cv2,numpy as np,glob,json,sys,os
WHITE=bool(os.environ.get('WHITE'));SMED=int(os.environ.get('SMED','90'));WSAT=int(os.environ.get('WSAT','48'));STH=int(os.environ.get('STH','60'));VTH=int(os.environ.get('VTH','120'))   # WHITE=1: white/light shirt (shirt = low saturation & bright)
FPS=int(sys.argv[1]);bx,by,bw,bh=[int(v) for v in sys.argv[2].split(',')]
fs=sorted(glob.glob('fr/*.png'));N=len(fs);H,W=1280,720
def skin(im):
    ycc=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    m=((ycc[...,1]>135)&(ycc[...,1]<185)&(ycc[...,2]>80)&(ycc[...,2]<135)&(hsv[...,1]>(WSAT if WHITE else 35))&(hsv[...,2]>70)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    if WHITE:   # drop tiny specks (warm-tinted fabric folds) - real hands/fingers are far larger; specks would punch holes in the print
        n,lab,st,_=cv2.connectedComponentsWithStats(m,connectivity=8);keep=np.zeros(n,np.uint8);keep[1:]=(st[1:,cv2.CC_STAT_AREA]>=1500)
        for i in range(1,n):   # warm fabric shadows next to a hand are weakly saturated; skin is strongly saturated
            if keep[i] and np.median(hsv[...,1][lab==i])<SMED:keep[i]=0
        m=keep[lab].astype(np.uint8)
    # no dilation: a grown mask lets the dark shirt around the fingers show up as a black rim over the print
    if WHITE:   # light shirt: a slightly grown mask only shows plain white fabric around the fingers (invisible); a tight one would leave a ring of print colour
        m=cv2.dilate(m,np.ones((5,5),np.uint8));return cv2.GaussianBlur(m.astype(np.float32),(0,0),1.4)
    m=cv2.erode(m,np.ones((3,3),np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32),(0,0),1.0)
# two-sided fabric lighting: code = 128 + (ratio-1)*254, ratio = local shirt brightness / its heavy blur
# (<128 = fold in shadow, >128 = fold catching light). Skin / background are excluded from the reference.
skins=[];ratios=[]
for t,f in enumerate(fs):
    im=cv2.imread(f);v=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[...,2].astype(np.float32)
    sk=skin(im);sm=cv2.GaussianBlur(v,(0,0),6)
    sat=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[...,1]
    M=(((v>VTH)&(sat<STH)) if WHITE else (v<70))&(cv2.dilate(sk,np.ones((31,31),np.uint8))<.05)
    M=M.astype(np.float32)
    M=cv2.erode(M,np.ones((9,9),np.uint8))
    num=cv2.GaussianBlur(sm*M,(0,0),28);den=cv2.GaussianBlur(M,(0,0),28)
    ref=num/np.maximum(den,1e-3);r=np.where(M>.5,sm/np.maximum(ref,4),1.0)
    ratios.append(r);skins.append((np.clip(sk,0,1)*255).astype(np.uint8))
ratios=np.array(ratios,np.float32)
sm_t=ratios.copy()
K=[1,2,3,2,1]
for t in range(len(ratios)):
    acc=0;w=0
    for k,wt in zip(range(-2,3),K):
        j=min(max(t+k,0),len(ratios)-1);acc=acc+ratios[j]*wt;w+=wt
    sm_t[t]=acc/w
shades=np.clip(128+(sm_t-1)*254,1,255).astype(np.uint8)
import os;os.makedirs('cl',exist_ok=True)
for t,f in enumerate(fs):os.system(f'cp {f} cl/{t:03d}.png')
np.save('skins.npy',np.array(skins));np.save('shades.npy',np.array(shades))
import os
if os.path.exists('Ms.npy'):
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from smooth_h import smooth_h
    Hs=smooth_h(np.load('Ms.npy'))
else:Hs=[[1,0,0,0,1,0,0,0,1]]*N
json.dump({'N':N,'fps':FPS,'white':255,'bi':1,'bbox':[bx,by,bx+bw,by+bh],'H':Hs},open('track_raw.json','w'))
print('ok',N,'frames')
