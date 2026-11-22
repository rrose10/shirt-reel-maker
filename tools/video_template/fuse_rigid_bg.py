"""fuse2: scale+y from the rigid parts, x from bg (hand window interpolated), bg before K0. usage: fuse2.py K0 K1 XA XB [cx cy]
 XA..XB = frames where bg x is corrupted (linear interpolation)."""
import numpy as np,sys
from scipy.ndimage import gaussian_filter1d
k0,k1,xa,xb=[int(v) for v in sys.argv[1:5]];c=np.array([360.,520.,1.])
B=np.load('Ms_bg.npy');B=np.array([b/b[2,2] for b in B]);R=np.load('Ms_rigid.npy');cc=np.load('rigid_cc.npy');ok=np.load('rigid_ok.npy');N=len(B)
def pos(h):p=h@c;return p[:2]/p[2]
def sc(h):return np.sqrt(abs(np.linalg.det(h[:2,:2])))
ps=np.array([pos(h) for h in R]);pb=np.array([pos(h) for h in B]);ss=np.log([sc(h) for h in R])
good=ok&(cc>=.99);good[:k0]=False
# gate vs running median trend
from scipy.signal import medfilt
for it in range(3):
    gi=np.where(good)[0]
    ys=np.interp(np.arange(N),gi,ps[gi,1]);sl=np.interp(np.arange(N),gi,ss[gi])
    ym=medfilt(ys,15);sm=medfilt(sl,15)
    bad=(np.abs(ps[:,1]-ym)>4)|(np.abs(ss-sm)>.012)
    good&=~bad
gi=np.where(good)[0]
y_f=gaussian_filter1d(np.interp(np.arange(N),gi,ps[gi,1]),2.5,mode='nearest')
s_f=np.exp(gaussian_filter1d(np.interp(np.arange(N),gi,ss[gi]),2.5,mode='nearest'))
x_b=pb[:,0].copy()
x_f=x_b.copy()
x_f[xa:xb]=np.interp(np.arange(xa,xb),[xa-1,xb],[x_b[xa-1],x_b[xb]])
x_f=gaussian_filter1d(x_f,2.0,mode='nearest')
print('rejected rigid frames',[k for k in range(k0,N) if not good[k]])
out=B.copy()
for k in range(N):
    M=np.eye(3);M[0,0]=M[1,1]=s_f[k];M[:2,2]=np.array([x_f[k],y_f[k]])-s_f[k]*c[:2]
    if k<k0:out[k]=B[k]
    elif k<k1:w=(k-k0)/(k1-k0);out[k]=B[k]*(1-w)+M*w
    else:out[k]=M
np.save('Ms.npy',out)
for k in range(56,240,6):print(k,'%.3f'%s_f[k],'x %.1f y %.1f'%(x_f[k],y_f[k]),'| bg x %.1f y %.1f s %.3f'%(pb[k,0],pb[k,1],sc(B[k])))
