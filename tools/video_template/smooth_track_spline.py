#!/usr/bin/env python3
"""Remove the slow wobble from a template's track (print speeds up / freezes while the shirt moves steadily).

A texture tracker that follows leaf shadows or cloth gives H a ~15-20 frame oscillation (the print stops for several frames, then
jumps). The camera move itself is steady, so scale, rotation and shift are smoothed with a smoothing spline (lam ~3000 keeps moves
longer than ~40 frames and removes the wobble, no bias at the ends). The original is saved as track_before_spline.json.
Check: collar/hem patch velocities from template.mp4 agree with the new H within ~0.1-0.2 px/frame (the old one was off by up to 0.8).

usage: smooth_track_spline.py <template dir> [lam=3000]
"""
import json,sys,shutil,os,numpy as np
from scipy.interpolate import make_smoothing_spline
d=sys.argv[1].rstrip('/')+'/';lam=float(sys.argv[2]) if len(sys.argv)>2 else 3000
t=json.load(open(d+'track.json'));H=np.array(t['H'],float).reshape(-1,3,3);x=np.arange(len(H))
P=np.c_[np.hypot(H[:,0,0],H[:,1,0]),np.arctan2(H[:,1,0],H[:,0,0]),H[:,0,2],H[:,1,2]]
Ps=np.array([make_smoothing_spline(x,P[:,j],lam=lam)(x) for j in range(4)]).T
if not os.path.exists(d+'track_before_spline.json'):shutil.copy(d+'track.json',d+'track_before_spline.json')
out=[]
for k in range(len(H)):
    s,a,tx,ty=Ps[k];out.append([round(float(v),6) for v in (s*np.cos(a),-s*np.sin(a),tx,s*np.sin(a),s*np.cos(a),ty,0,0,1)])
t['H']=out;json.dump(t,open(d+'track.json','w'),separators=(',',':'))
print('max shift change %.2f px, scale %.4f'%(np.abs(Ps[:,2:]-P[:,2:]).max(),np.abs(Ps[:,0]-P[:,0]).max()))
