"""For a PERSON wearing the shirt (the body is not rigid): reduce the tracked homographies in Ms.npy to
scale + rotation + translation measured over the torso, so the print never skews.  Run after track_gen.py,
then build_blank.py.   python3 similarity.py [x0,y0,x1,y1 of the torso in the reference frame]"""
import numpy as np,cv2,sys
x0,y0,x1,y1=[float(v) for v in (sys.argv[1] if len(sys.argv)>1 else "120,560,600,1100").split(",")]
Ms=np.load("Ms.npy");np.save("Ms_homography.npy",Ms)
gx,gy=np.meshgrid(np.linspace(x0,x1,7),np.linspace(y0,y1,7));P0=np.float32(np.c_[gx.ravel(),gy.ravel()]).reshape(-1,1,2)
out=[]
for M in Ms:
    P1=cv2.perspectiveTransform(P0,M);A,_=cv2.estimateAffinePartial2D(P0,P1,method=cv2.LMEDS);out.append(np.vstack([A,[0,0,1]]))
np.save("Ms.npy",np.array(out))
