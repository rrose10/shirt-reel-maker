import cv2,numpy as np,glob,json,os,subprocess
exec(open('track.py').read().split('ims=')[0])
ims=[cv2.imread(f) for f in fs];N=len(ims)
P=[printmask(i) for i in ims];S=[skin(i) for i in ims]
Ws=np.load('Ws2.npy').astype(np.float64)
# --- outlier removal + smoothing of the tracked homographies
Hs=[np.linalg.inv(w) for w in Ws];Hs=[h/h[2,2] for h in Hs]
Pm=np.array([h.flatten()[:8] for h in Hs])
def corners_shift(h):
    c=np.array([[100,200,1],[600,200,1],[100,900,1],[600,900,1]],float).T;q=h@c;q=(q[:2]/q[2]).T;return q
Q=np.array([corners_shift(h).flatten() for h in Hs])
med=np.array([np.median(Q[max(0,i-3):i+4],axis=0) for i in range(N)])
bad=np.where(np.abs(Q-med).max(axis=1)>12)[0];print('outliers',bad)
for i in bad:
    a=i-1
    while a in bad:a-=1
    b=i+1
    while b in bad and b<N-1:b+=1
    wgt=(i-a)/(b-a);Pm[i]=Pm[a]*(1-wgt)+Pm[b]*wgt
# gentle temporal smoothing (keeps duplicate-frame stepping)
k=np.exp(-0.5*(np.arange(-3,4)/1.0)**2);k/=k.sum()
pad=np.pad(Pm,((3,3),(0,0)),mode='edge');Ps=np.stack([np.convolve(pad[:,j],k,mode='valid') for j in range(8)],1)
Hs=[np.append(p,1).reshape(3,3) for p in Ps]
# --- reference print bbox (frame 0 cluster)
ys,xs=np.where(P[0]>0);x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max();print('print bbox',x0,y0,x1,y1)
# --- clean plate + masks
def fabric_fill(im,m):
    valid=(1-m).astype(np.float32)
    f=im.astype(np.float32)
    out=f.copy()
    num=np.zeros_like(f);den=np.zeros_like(valid)
    for s in (6,18):
        n=cv2.GaussianBlur(f*valid[...,None],(0,0),s);d=cv2.GaussianBlur(valid,(0,0),s)
        fill=n/np.maximum(d[...,None],1e-3)
        w=(d>0.05).astype(np.float32)[...,None]
        out=np.where((m[...,None]>0)&(w>0),fill,out) if s==6 else np.where((m[...,None]>0)&(out_prev_unfilled[...,None]),fill,out)
        out_prev_unfilled=(m>0)&(d<=0.05)
    return out
rng=np.random.default_rng(1)
os.makedirs('cl',exist_ok=True)
skins=[];shades=[]
for t in range(N):
    im=ims[t];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    sk=S[t]
    skd=cv2.dilate(sk,np.ones((5,5),np.uint8))
    mx=im.max(axis=2).astype(np.int16);mn=im.min(axis=2).astype(np.int16);neutral=((mx-mn)<24)
    # region where the old print lives this frame
    reg=cv2.dilate(P[t],np.ones((31,31),np.uint8))
    wr=cv2.warpPerspective(cv2.dilate(P[0],np.ones((45,45),np.uint8)),np.linalg.inv(Hs[t]),(W,H),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)
    reg=((reg>0)|(wr>0)).astype(np.uint8)
    bright=((hsv[...,2]>52)&neutral).astype(np.uint8)  # print incl. anti-aliased edge (fabric is < ~60)
    ring=(cv2.dilate(sk,np.ones((23,23),np.uint8))&(1-cv2.dilate(sk,np.ones((3,3),np.uint8))))
    near=((hsv[...,2]>72)&((mx-mn)<48)&(ring>0)&(reg>0)).astype(np.uint8)
    strict=((hsv[...,2]>165)&((mx-mn)<16)&(reg>0)).astype(np.uint8);strict=cv2.dilate(strict,np.ones((5,5),np.uint8))
    pm=((bright&reg&(1-skd))|near|strict).astype(np.uint8)
    pm=cv2.dilate(pm,np.ones((5,5),np.uint8))&(1-skd)
    # shade of the original print (brightness of print pixels), spread over the print area
    core=((hsv[...,2]>195)&(hsv[...,1]<55)&(reg>0)&(skd==0)).astype(np.float32)
    v=hsv[...,2].astype(np.float32)
    n=cv2.GaussianBlur(v*core,(0,0),25);d=cv2.GaussianBlur(core,(0,0),25)
    sh=np.where(d>0.002,n/np.maximum(d,1e-4),235)
    sh=np.clip(sh,150,255)
    shades.append(sh.astype(np.uint8))
    # fabric fill
    f=im.astype(np.float32);skb=cv2.dilate(sk,np.ones((17,17),np.uint8));valid=((1-pm)*(1-skb)).astype(np.float32)
    n1=cv2.GaussianBlur(f*valid[...,None],(0,0),5);d1=cv2.GaussianBlur(valid,(0,0),5)
    fill=n1/np.maximum(d1[...,None],1e-3)
    n2=cv2.GaussianBlur(f*valid[...,None],(0,0),20);d2=cv2.GaussianBlur(valid,(0,0),20)
    fill2=n2/np.maximum(d2[...,None],1e-3)
    fill=np.where((d1>0.15)[...,None],fill,fill2)
    # fabric grain
    g=rng.normal(0,2.2,(H,W)).astype(np.float32);g=cv2.GaussianBlur(g,(0,0),0.7)*1.6
    fill=fill+g[...,None]
    soft=cv2.GaussianBlur(pm.astype(np.float32),(0,0),1.0)[...,None]
    out=f*(1-soft)+fill*soft
    cv2.imwrite(f'cl/{t:03d}.png',np.clip(out,0,255).astype(np.uint8))
    # soft skin occluder (hands): closing + feather
    s2=cv2.morphologyEx(sk,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8));s2=cv2.dilate(s2,np.ones((5,5),np.uint8))
    s2=cv2.GaussianBlur(s2.astype(np.float32),(0,0),1.5)
    skins.append((np.clip(s2*1.0,0,1)*255).astype(np.uint8))
np.save('skins.npy',np.array(skins));np.save('shades.npy',np.array(shades))
json.dump({'N':N,'bbox':[int(x0),int(y0),int(x1),int(y1)],'H':[[float(v) for v in h.flatten()] for h in Hs]},open('track_raw.json','w'))
print('done')
