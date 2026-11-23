"""Compare a template's tracking with the motion of the rigid part of the shirt above the print (collar, shoulders, hanger).
usage: cue_drift.py <template id>[:<track.json>]   (run from the repo root; TPL_ROOT overrides site/templates/)
Prints how far the tracked print centre / scale deviates from what the collar region says. Real zoom shows up in both and cancels;
a stretch where this grows and comes back is the tracker drifting while the hand works on the fabric (see fix_track.py --cue).
"""
import cv2,numpy as np,json,sys
import os
T=os.environ.get('TPL_ROOT','site/templates/')
if not T.endswith('/'):T+='/'
def load(name):
    c=cv2.VideoCapture(T+name+'/template.mp4');g=[];sk=[]
    while True:
        ok,x=c.read()
        if not ok:break
        g.append(cv2.cvtColor(x[:, :720],cv2.COLOR_BGR2GRAY))
        sk.append(cv2.resize(cv2.cvtColor(x[:640,720:1080],cv2.COLOR_BGR2GRAY),(720,1280),interpolation=cv2.INTER_LINEAR))
    return g,sk
def run(name,track=None,band=(0.1,0.7,0.02),cache={}):
    if name not in cache:cache[name]=load(name)
    g,sk=cache[name];t=json.load(open(track or T+name+'/track.json'));H=np.array(t['H']).reshape(-1,3,3);n=len(g)
    bx,by,bw,bh=t['bbox'];cref=np.array([bx+bw/2,by+bh/2,1.0])
    bref=np.array([[bx-band[0]*bw,by-band[1]*bh,1],[bx+bw*(1+band[0]),by-band[1]*bh,1],[bx+bw*(1+band[0]),by-band[2]*bh,1],[bx-band[0]*bw,by-band[2]*bh,1]])
    c0=H[0]@cref;c0=c0/c0[2];Tm=np.eye(3);out=[];nf=[]
    for k in range(1,n):
        q=bref@H[k-1].T;q=q[:,:2]/q[:,2:]
        pm=np.zeros((1280,720),np.uint8);cv2.fillPoly(pm,[q.astype(np.int32)],255)
        hm=cv2.dilate((sk[k-1]>60).astype(np.uint8),np.ones((41,41),np.uint8));hm2=cv2.dilate((sk[k]>60).astype(np.uint8),np.ones((41,41),np.uint8))
        pm[(hm>0)|(hm2>0)]=0
        p0=cv2.goodFeaturesToTrack(g[k-1],400,0.005,6,mask=pm,blockSize=7)
        if p0 is None or len(p0)<8:out.append(None);continue
        p1,st,er=cv2.calcOpticalFlowPyrLK(g[k-1],g[k],p0,None,winSize=(21,21),maxLevel=3)
        ok=st.ravel()==1;a,b=p0[ok],p1[ok]
        if len(a)<8:out.append(None);continue
        W,inl=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.0)
        if W is None:out.append(None);continue
        Wl=np.eye(3);Wl[:2]=W;Tm=Wl@Tm;nf.append(int(inl.sum()))
        ph=H[k]@cref;ph=ph[:2]/ph[2];pb=Tm@c0;pb=pb[:2]/pb[2]
        sH=np.hypot(*(H[k]@np.linalg.inv(H[0]))[:2,0]);sB=np.hypot(*Tm[:2,0])
        out.append((k,100*(sH/sB-1),ph[0]-pb[0],ph[1]-pb[1]))
    return out,np.median(nf) if nf else 0
if __name__=='__main__':
    for a in sys.argv[1:]:
        name,_,tr=a.partition(':');r,nf=run(name,tr or None);r=[x for x in r if x]
        ds=np.array([x[1:] for x in r]);i=np.abs(ds).argmax(0)
        print('%-24s inliers~%d max dev: scale %+.1f%% (k%d)  dx %+.1f (k%d)  dy %+.1f (k%d)'%(name,nf,ds[i[0],0],r[i[0]][0],ds[i[1],1],r[i[1]][0],ds[i[2],2],r[i[2]][0]))
        print('    (k,scale%,dx,dy):',[(x[0],round(x[1],1),round(x[2]),round(x[3])) for x in r[::10]])
