#!/usr/bin/env python3
"""Local bending/sliding of the fabric under the print, for templates with creases that move (hand smoothing the shirt).

usage: fabric_field.py <template dir> [freeze frame]     (writes <dir>/fabric.json)
The cloth creases are mostly vertical, so only the sideways (x) shift is measurable (the aperture problem hides the rest): 1-D Lucas-Kanade
on the plate between consecutive frames, with warping, hand + hand shadow masked out and filled from the surroundings. A coarse grid of
cloth points (frame 0 positions) is carried through the flows; the part that is just the rigid track (similarity fit) is removed.
Stored as int8 in 0.1 px units of the 720-wide plate: frames x rows x cols (x shift only).
"""
import cv2,numpy as np,json,sys,base64
d=sys.argv[1].rstrip('/');name=d.split('/')[-1]
FREEZE=int(sys.argv[2]) if len(sys.argv)>2 else None   # after this frame the shirt is at rest (camera push-in only): hold the field, do not keep integrating flow
t=json.load(open(d+'/track.json'));H=np.array(t['H'],float).reshape(-1,3,3)
cap=cv2.VideoCapture(d+'/template.mp4');G=[];SK=[]
while True:
    ok,x=cap.read()
    if not ok:break
    g=cv2.cvtColor(x[:, :720],cv2.COLOR_BGR2GRAY).astype(np.float32);G.append(cv2.GaussianBlur(cv2.resize(g,(360,640),interpolation=cv2.INTER_AREA),(0,0),1.5))
    SK.append(cv2.cvtColor(x[:640,720:1080],cv2.COLOR_BGR2GRAY))
n=min(len(H),len(G))
bx,by,bw,bh=t['bbox'];x0=max(8,int(bx/2-bw*0.2));x1=min(352,int((bx+bw)/2+bw*0.2));y0=max(8,int(by/2-bh*0.2));y1=min(632,int((by+bh)/2+bh*0.25))
STEP=16;gx=np.arange(x0,x1+1,STEP,dtype=np.float32);gy=np.arange(y0,y1+1,STEP,dtype=np.float32);GX,GY=np.meshgrid(gx,gy);P0=np.stack([GX,GY],-1).reshape(-1,2)
yy,xx=np.mgrid[0:640,0:360].astype(np.float32)
def lk_x(A,B,bad):
    """horizontal shift field u such that A(x) ~ B(x+u); bad = pixels not to trust"""
    u=np.zeros_like(A);w_ok=(1-bad).astype(np.float32)
    for it in range(4):
        Bw=cv2.remap(B,xx+u,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        Ix=0.5*(cv2.Sobel(A,cv2.CV_32F,1,0,ksize=3)/8+cv2.Sobel(Bw,cv2.CV_32F,1,0,ksize=3)/8)
        It=Bw-A
        # window sums
        a=cv2.GaussianBlur(Ix*Ix*w_ok,(0,0),9);b=cv2.GaussianBlur(-Ix*It*w_ok,(0,0),9)
        du=b/(a+4.0)          # +4: damping, where there is no gradient the shift stays 0
        u=u+np.clip(du,-3,3)
        u=cv2.GaussianBlur(u,(0,0),3)
    conf=cv2.GaussianBlur(Ix*Ix*w_ok,(0,0),9)
    return u,conf
pos=P0.copy();track=[P0.copy()]
for k in range(1,n):
    hand=((SK[k-1]>50)|(SK[k]>50)).astype(np.uint8);bad=cv2.dilate(hand,np.ones((61,61),np.uint8))   # hand + its shadow
    # remove global brightness change (hand shadow / exposure) before matching
    A=G[k-1];B=G[k]
    u,conf=lk_x(A,B,bad)
    ok=(conf>8)&(bad==0)
    num=cv2.GaussianBlur(u*ok,(0,0),16);den=cv2.GaussianBlur(ok.astype(np.float32),(0,0),16)
    u=np.where(ok,u,num/np.maximum(den,1e-3))
    ix=np.clip(pos[:,0],0,358.99);iy=np.clip(pos[:,1],0,638.99);xi=ix.astype(int);yi=iy.astype(int);ax=(ix-xi)
    f=u[yi,xi]*(1-ax)+u[yi,xi+1]*ax
    pos=pos.copy();pos[:,0]+=f;track.append(pos.copy())
track=np.array(track)
out=np.zeros((n,len(P0)),np.float32);H0i=np.linalg.inv(H[0])
for k in range(n):
    M=H[k]@H0i;p=np.c_[P0*2,np.ones(len(P0))]@M.T;rig=(p[:,:2]/p[:,2:])/2
    r=track[k,:,0]-rig[:,0]                          # x residual, half-res
    c=P0-P0.mean(0);A=np.c_[np.ones(len(P0)),c[:,0],c[:,1]]   # remove x-shift + x-scale + shear (rigid-ish part)
    sol=np.linalg.lstsq(A,r,rcond=None)[0];out[k]=(r-A@sol)*2
if FREEZE is not None:
    for k in range(FREEZE+1,n):out[k]=out[FREEZE]
lim=12.0;out=np.clip(out,-lim,lim)
q=np.round(out/0.1).astype(np.int8)
res={'step':STEP*2,'x0':float(gx[0]*2),'y0':float(gy[0]*2),'cols':len(gx),'rows':len(gy),'n':n,'unit':0.1,'d':base64.b64encode(q.reshape(n,len(gy),len(gx)).tobytes()).decode()}
json.dump(res,open(d+'/fabric.json','w'),separators=(',',':'))
print('wrote',d+'/fabric.json','grid',len(gx),'x',len(gy),'frames',n,'size KB',len(res['d'])//1024)
print('rms by frame',[(k,round(float(np.sqrt((out[k]**2).mean())),2)) for k in range(0,n,20)])
np.save('/tmp/w/fld/%s_res.npy'%name,out);np.save('/tmp/w/fld/%s_P0.npy'%name,P0)
