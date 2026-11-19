"""Build src/hands.js from the three photo-real hand cut-outs in tools/hands_src/ (hand on the left, forearm to the right).
The photographed arm is cut by the image edge, so its last columns fade out and the page continues the forearm with a smooth
skin-coloured fan (see armPath in the HTML), widening like a real forearm."""
import json, base64, io, numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter1d
FADE = 120
out = {}
for name in ['open', 'point1', 'point2']:
    im = Image.open(f'tools/hands_src/{name}.png').convert('RGBA')
    a = np.array(im).astype(float)
    H, W = a.shape[:2]
    alpha = a[:, :, 3]
    ys, xs = np.where(alpha > 20)
    x0c, y0c, y1c = int(xs.min()) - 3, int(ys.min()) - 3, int(ys.max()) + 4
    # trim the last columns (edge artefacts) -> arm photo ends at ax
    ax = W - 6
    crop = a[y0c:y1c, x0c:ax, :].copy()
    h, w = crop.shape[:2]
    # fade the end of the photo
    ramp = np.ones(w); ramp[w - FADE:] = np.linspace(1, 0, FADE)
    crop[:, :, 3] *= ramp[None, :]
    al = alpha[y0c:y1c, x0c:ax]
    def edge(x):
        rows = np.where(al[:, x] > 128)[0]
        return int(rows[0]), int(rows[-1])
    xe = w - 1
    yt, yb = edge(xe - 2)
    # slopes of the forearm edges over the last 220 px
    xsamp = np.arange(w - 220, w - 4, 6)
    tops = np.array([edge(x)[0] for x in xsamp]); bots = np.array([edge(x)[1] for x in xsamp])
    sT = float(np.polyfit(xsamp, tops, 1)[0]); sB = float(np.polyfit(xsamp, bots, 1)[0])
    # colour profile across the arm at the end of the photo (use the un-faded pixels just before the fade)
    xa, xb = w - 36, w - 6
    seg = a[y0c:y1c, x0c + xa:x0c + xb, :3]
    wgt = (al[:, xa:xb] > 200)[:, :, None]
    prof = (seg * wgt).sum(axis=1) / np.maximum(wgt.sum(axis=1), 1)
    t0, b0 = edge(xa)[0], edge(xb - 1)[1]
    t1, b1 = edge(xa)[0], edge(xb - 1)[1]
    prof = gaussian_filter1d(prof, 3, axis=0, mode='nearest')
    # rows of the arm at the far end (yt..yb) mapped to the profile rows taken at the fade start
    pt, pb = edge(xa)
    pt2, pb2 = edge(xb - 1)
    pt, pb = min(pt, pt2), max(pb, pb2)
    stops = []
    n = 14
    for i in range(n + 1):
        t = i / n
        r = prof[int(round(pt + t * (pb - pt)))]
        stops.append([round(t, 3), int(r[0]), int(r[1]), int(r[2])])
    # anchor: open hand = centre of the palm/fingers area; pointing = fingertip
    cols = np.where((al > 128).any(axis=0))[0]
    if name == 'open':
        hand = al[:, :int(w * 0.52)] > 128
        yy, xx = np.where(hand)
        cx, cy = float(xx.mean()), float(yy.mean())
    else:
        xm = int(cols.min())
        rows = np.where(al[:, xm:xm + 6].max(axis=1) > 128)[0]
        cx, cy = float(xm + 6), float(rows.mean())
    img = Image.fromarray(crop.astype(np.uint8), 'RGBA')
    buf = io.BytesIO(); img.save(buf, 'WEBP', quality=92, alpha_quality=100, method=4)
    out[name] = {'b': base64.b64encode(buf.getvalue()).decode(), 'w': w, 'h': h, 'cx': round(cx, 1), 'cy': round(cy, 1),
                 'ax': w, 'fd': FADE, 'yt': yt, 'yb': yb, 'yt0': edge(w - FADE)[0], 'yb0': edge(w - FADE)[1], 'sT': round(sT, 5), 'sB': round(sB, 5), 'prof': stops}
    print(name, (w, h), 'anchor', (round(cx), round(cy)), 'arm rows', yt, yb, 'slopes', round(sT, 3), round(sB, 3), 'bytes', len(buf.getvalue()))
open('src/hands.js', 'w').write('window.HANDS=' + json.dumps(out, separators=(',', ':')) + ';\n')
