#!/usr/bin/env python3
"""Add the swing of a hanging shirt to a template's track (scale+shift only until now).

usage: collar_rotation.py <template dir> x0,y0,x1,y1      (box around hanger + collar in the reference frame, 720x1280 plate px)
The plate is warped back with the existing track (cancels the camera zoom), then a rigid fit (rotation + shift, ECC) of that box against frame 0
gives how far the shirt has swung on its hanger. H[f] becomes H[f] @ R[f] @ R[0]^-1. The old track is kept as track_norot.json.
"""
import cv2,numpy as np,json,sys,shutil
from scipy.ndimage import gaussian_filter1d
d=sys.argv[1].rstrip('/');x0,y0,x1,y1=[int(v) for v in sys.argv[2].split(',')]
tr=json.load(open(d+'/track.json'));H=np.array(tr['H'],float).reshape(-1,3,3)
cap=cv2.VideoCapture(d+'/template.mp4');fr=[]
while True:
    ok,x=cap.read()
    if not ok:break
    fr.append(cv2.cvtColor(x[:1280,:720],cv2.COLOR_BGR2GRAY).astype(np.float32))
blur=lambda a:cv2.GaussianBlur(a,(0,0),1.5)
g0=blur(fr[0]);mask=np.zeros((1280,720),np.uint8);mask[y0:y1,x0:x1]=255
P=[];prev=np.eye(2,3,dtype=np.float32)
for f in range(len(fr)):
    Hi=np.linalg.inv(H[f]);w=blur(cv2.warpPerspective(fr[f],Hi,(720,1280)));valid=cv2.warpPerspective(np.full((1280,720),255,np.uint8),Hi,(720,1280))
    m=((mask>0)&(valid>250)).astype(np.uint8);W=prev.copy();ok=False
    if m.sum()>4000:
        try:
            cc,W=cv2.findTransformECC(g0,w,W,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),m,5);ok=cc>0.9
        except Exception:pass
    if ok:prev=W.copy();P.append([np.arctan2(W[1,0],W[0,0]),W[0,2],W[1,2]])
    else:P.append([np.nan]*3)
P=np.array(P)
for j in range(3):                                     # frames that did not fit: interpolate
    g=np.isfinite(P[:,j]);P[:,j]=np.interp(np.arange(len(P)),np.where(g)[0],P[g,j])
P=gaussian_filter1d(P,1.5,axis=0,mode='nearest')
def mat(a,tx,ty):return np.array([[np.cos(a),-np.sin(a),tx],[np.sin(a),np.cos(a),ty],[0,0,1]])
W0i=np.linalg.inv(mat(*P[0]))
shutil.copy(d+'/track.json',d+'/track_norot.json')
Hn=[H[f]@mat(*P[f])@W0i for f in range(len(H))]
tr['H']=[[round(float(v),6) for v in h.reshape(-1)] for h in Hn]
json.dump(tr,open(d+'/track.json','w'),separators=(',',':'))
print('max swing %.2f deg at frame %d'%(np.degrees(np.abs(P[:,0]-P[0,0]).max()),int(np.abs(P[:,0]-P[0,0]).argmax())))
