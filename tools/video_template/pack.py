"""Pack the clean plates (cl/), skin masks and shade maps into template.mp4 + track.json.
Usage: python3 pack.py OUTDIR "Template name"   (run after build_tpl.py in the same work folder)"""
import cv2,numpy as np,json,os,subprocess,sys
out=sys.argv[1];name=sys.argv[2] if len(sys.argv)>2 else "Template"
sk=np.load('skins.npy');sh=np.load('shades.npy');N=len(sk);os.makedirs('pk',exist_ok=True);os.makedirs(out,exist_ok=True)
for t in range(N):
    c=cv2.imread(f'cl/{t:03d}.png')
    s=cv2.resize(sk[t],(360,640),interpolation=cv2.INTER_AREA);h=cv2.resize(sh[t],(360,640),interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'pk/{t:03d}.png',np.hstack([c,cv2.cvtColor(np.vstack([s,h]),cv2.COLOR_GRAY2BGR)]))
subprocess.check_call(['ffmpeg','-v','error','-y','-framerate','30','-i','pk/%03d.png','-c:v','libx264','-preset','slow','-crf','17','-g','8','-pix_fmt','yuv420p','-movflags','+faststart','-an',out+'/template.mp4'])
t=json.load(open('track_raw.json'));x0,y0,x1,y1=t['bbox']
json.dump({'v':1,'name':name,'w':720,'h':1280,'fps':30,'n':t['N'],'white':round(t.get('white',232)),'bbox':[x0,y0,x1-x0,y1-y0],'layout':{'skin':[720,0,360,640],'shade':[720,640,360,640]},'H':[[round(v,6) for v in h] for h in t['H']]},open(out+'/track.json','w'),separators=(',',':'))
print('wrote',out)
