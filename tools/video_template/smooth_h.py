"""Remove tracking spikes and jitter from per-frame homographies (Ms.npy: ref -> frame k).
The four corners of the frame are followed over time (median filter against spikes, then Gaussian), and the
homography is rebuilt from the smoothed corners.   smooth_h(Ms) -> list of 9-float lists."""
import numpy as np,cv2
from scipy.ndimage import gaussian_filter1d,median_filter
def smooth_h(Ms,W=720,H=1280,med=9,sigma=2.6):
    Ms=np.asarray(Ms,float);Ms=Ms/Ms[:,2:3,2:3];C=np.float32([[0,0],[W,0],[0,H],[W,H]])
    P=np.array([cv2.perspectiveTransform(C.reshape(-1,1,2),M).reshape(-1,2) for M in Ms])
    P=median_filter(P,size=(med,1,1),mode='nearest');P=gaussian_filter1d(P,sigma,axis=0,mode='nearest')
    out=[]
    for p in P:
        M=cv2.getPerspectiveTransform(C,np.float32(p));M=M/M[2,2];out.append([float(v) for v in M.ravel()])
    return out
