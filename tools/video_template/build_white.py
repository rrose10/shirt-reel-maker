"""Packing inputs for the print-erased white-shirt template: hand mattes, two-sided shade map, smoothed tracking.
Usage: python3 build_white.py FPS x,y,w,h   (after erase_print.py; then pack.py)"""
import cv2,numpy as np,glob,json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from smooth_h import smooth_h
FPS=int(sys.argv[1]);bx,by,bw,bh=[int(v) for v in sys.argv[2].split(',')]
fs=sorted(glob.glob('cl/*.png'));N=len(fs);SK=np.load('skins_bin.npy')
skins=[];ratios=[]
for t,f in enumerate(fs):
    im=cv2.imread(f);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);v=hsv[...,2].astype(np.float32);sat=hsv[...,1]
    sk=SK[t].astype(np.float32)
    sk=cv2.dilate(sk,np.ones((5,5),np.uint8));skf=cv2.GaussianBlur(sk,(0,0),1.4)   # a slightly grown matte only shows plain white fabric
    sm=cv2.GaussianBlur(v,(0,0),6)
    M=((v>120)&(sat<60)&(cv2.dilate(sk,np.ones((31,31),np.uint8))<.05)).astype(np.float32);M=cv2.erode(M,np.ones((9,9),np.uint8))
    num=cv2.GaussianBlur(sm*M,(0,0),28);den=cv2.GaussianBlur(M,(0,0),28)
    ref=num/np.maximum(den,1e-3);r=np.where(M>.5,sm/np.maximum(ref,4),1.0)
    ratios.append(r);skins.append((np.clip(skf,0,1)*255).astype(np.uint8))
ratios=np.array(ratios,np.float32);sm_t=ratios.copy();K=[1,2,3,2,1]
for t in range(N):
    acc=0;w=0
    for k,wt in zip(range(-2,3),K):
        j=min(max(t+k,0),N-1);acc=acc+ratios[j]*wt;w+=wt
    sm_t[t]=acc/w
shades=np.clip(128+(sm_t-1)*254,1,255).astype(np.uint8)
np.save('skins.npy',np.array(skins));np.save('shades.npy',np.array(shades))
Hs=smooth_h(np.load('Mp.npy'))[:N]
json.dump({'N':N,'fps':FPS,'white':255,'bi':1,'bbox':[bx,by,bx+bw,by+bh],'H':Hs},open('track_raw.json','w'))
print('ok',N)
