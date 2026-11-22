"""Template from a video of a BLANK shirt filmed with a fixed camera (nothing to remove or track).
Usage (in a work folder that has fr/NNN.png at 720x1280):  python3 build_blank.py FPS x,y,w,h
 x,y,w,h = the print area in frame pixels. Then run pack.py."""
import cv2,numpy as np,glob,json,sys
FPS=int(sys.argv[1]);bx,by,bw,bh=[int(v) for v in sys.argv[2].split(',')]
fs=sorted(glob.glob('fr/*.png'));N=len(fs);H,W=1280,720
def skin(im):
    ycc=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    m=((ycc[...,1]>135)&(ycc[...,1]<185)&(ycc[...,2]>80)&(ycc[...,2]<135)&(hsv[...,1]>35)&(hsv[...,2]>70)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    return cv2.GaussianBlur(cv2.dilate(m,np.ones((5,5),np.uint8)).astype(np.float32),(0,0),1.5)
ref=cv2.cvtColor(cv2.imread(fs[0]),cv2.COLOR_BGR2HSV)[...,2].astype(np.float32)
med=float(np.median(ref[by:by+bh,bx:bx+bw]))
skins=[];shades=[]
for t,f in enumerate(fs):
    im=cv2.imread(f);v=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[...,2].astype(np.float32)
    vb=cv2.GaussianBlur(v,(0,0),5)
    shades.append(np.clip(246+5.0*(vb-med),150,255).astype(np.uint8))
    skins.append((np.clip(skin(im),0,1)*255).astype(np.uint8))
import os;os.makedirs('cl',exist_ok=True)
for t,f in enumerate(fs):os.system(f'cp {f} cl/{t:03d}.png')
np.save('skins.npy',np.array(skins));np.save('shades.npy',np.array(shades))
json.dump({'N':N,'fps':FPS,'white':255,'bbox':[bx,by,bx+bw,by+bh],'H':[[1,0,0,0,1,0,0,0,1]]*N},open('track_raw.json','w'))
print('ok',N,'frames, median shirt V',med)
