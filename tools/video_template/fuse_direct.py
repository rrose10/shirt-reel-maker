"""Robust fusion of direct-tracking results: keep confident frames (cc>=CCMIN, not an outlier vs. the running median),
fill the rest with monotone-cubic (PCHIP) interpolation of log-scale / rotation / position, light Gaussian smoothing.
usage: fuse_direct.py RUN.npy CC.npy OUT.npy [CCMIN=.93] [KEEP=a:b,c:d ...]   (KEEP = frame ranges that must be interpolated even if cc is high)
Writes OUT (N x 3 x 3 similarities ref -> frame, built about the image centre)."""
import numpy as np,sys
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d,median_filter
M=np.load(sys.argv[1]);cc=np.load(sys.argv[2]);out=sys.argv[3];CCMIN=float(sys.argv[4]) if len(sys.argv)>4 else .93
DROP=[tuple(int(v) for v in a.split(':')) for a in sys.argv[5:]]
N=len(M);c0=np.array([360.,640.])
def dec(m):
    s=np.sqrt(abs(np.linalg.det(m[:2,:2])));r=np.arctan2(m[1,0],m[0,0]);p=m[:2,:2]@c0+m[:2,2];return np.log(s),r,p[0],p[1]
P=np.array([dec(m) for m in M])
good=cc>=CCMIN
for a,b in DROP:good[a:b+1]=False
for it in range(3):
    gi=np.where(good)[0]
    cur=np.stack([np.interp(np.arange(N),gi,P[gi,j]) for j in range(4)],1)
    med=median_filter(cur,size=(9,1),mode='nearest')
    bad=(np.abs(P[:,0]-med[:,0])>.025)|(np.abs(P[:,2]-med[:,2])>12)|(np.abs(P[:,3]-med[:,3])>12)
    good&=~bad
gi=np.where(good)[0]
F=np.stack([PchipInterpolator(gi,P[gi,j],extrapolate=True)(np.arange(N)) for j in range(4)],1)
F=gaussian_filter1d(F,1.5,axis=0,mode='nearest')
R=np.zeros((N,3,3))
for k in range(N):
    s=np.exp(F[k,0]);c,sn=np.cos(F[k,1])*s,np.sin(F[k,1])*s
    L=np.array([[c,-sn],[sn,c]]);R[k]=np.eye(3);R[k,:2,:2]=L;R[k,:2,2]=F[k,2:]-L@c0
np.save(out,R);np.save(out.replace('.npy','_good.npy'),good)
print('kept',good.sum(),'/',N,' interpolated:',[k for k in range(N) if not good[k]])
for k in range(0,N,8):print(k,'%.3f'%np.exp(F[k,0]),'rot %.2f'%np.degrees(F[k,1]),'pos(%.0f,%.0f)'%(F[k,2],F[k,3]),'' if good[k] else '*')
