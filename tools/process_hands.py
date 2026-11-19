"""Rebuild src/hands.js from tools/hands.orig.js.
The original hand sprites have a forearm that was stretched from a single pixel column (streaky, and too short when the camera zooms).
This keeps only the real photographed part (hand + wrist), fades its end, and stores the skin colours across the arm so the
page can draw a clean, endless forearm (slightly widening towards the elbow) behind it."""
import json, base64, io, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
src = open('tools/hands.orig.js').read()
data = json.loads(src[src.index('{'):src.rindex('}') + 1])
FADE = 110
out = {}
for name, c in data.items():
    im = Image.open(io.BytesIO(base64.b64decode(c['b']))).convert('RGBA')
    a = np.array(im).astype(float)
    H, W = a.shape[:2]
    diff = np.abs(np.diff(a[:, :, :3], axis=1)).mean(axis=(0, 2))
    d2 = diff.copy(); d2[W - 12:] = 0
    ma = np.convolve(d2, np.ones(30) / 30, mode='same')
    big = np.where(ma > 0.25)[0]            # last stretch of columns that still change = end of the real photo
    xs = int(big[-1]) + 15
    xs = int(xs)
    alpha = a[:, :, 3]
    def ext(x):
        rows = np.where(alpha[:, x] > 128)[0]
        return int(rows[0]), int(rows[-1])
    yt, yb = ext(xs - 1)
    yt0, yb0 = ext(xs - FADE)
    # colour profile across the arm, from the last columns of real photo, blurred so it is smooth
    cols = a[:, xs - 60:xs - 1, :3]
    w = (alpha[:, xs - 60:xs - 1] > 200)[:, :, None]
    prof = (cols * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1)
    valid = np.where(w.sum(axis=1)[:, 0] > 5)[0]
    prof = prof[valid[0]:valid[-1] + 1]
    prof = gaussian_filter1d(prof, 7, axis=0, mode='nearest')
    stops = []
    n = 14
    for i in range(n + 1):
        t = i / n
        r = prof[int(round(t * (len(prof) - 1)))]
        stops.append([round(t, 3), int(r[0]), int(r[1]), int(r[2])])
    # offset between valid rows and yt..yb (valid rows are inside the opaque region)
    vt, vb = int(valid[0]), int(valid[-1])
    # crop to the real photo and fade its end
    crop = a[:, :xs, :].copy()
    ramp = np.ones(xs)
    ramp[xs - FADE:] = np.linspace(1, 0, FADE)
    crop[:, :, 3] *= ramp[None, :]
    img = Image.fromarray(crop.astype(np.uint8), 'RGBA')
    buf = io.BytesIO()
    img.save(buf, 'WEBP', quality=92, alpha_quality=100, method=4)
    out[name] = {'b': base64.b64encode(buf.getvalue()).decode(), 'w': xs, 'h': H, 'cx': c['cx'], 'cy': c['cy'],
                 'ax': xs, 'fd': FADE, 'yt0': yt0 + 0, 'yb0': yb0, 'yt': vt, 'yb': vb, 'prof': stops}
    print(name, 'photo ends at', xs, 'of', W, 'arm rows', vt, vb, 'bytes', len(buf.getvalue()))
open('src/hands.js', 'w').write('window.HANDS=' + json.dumps(out, separators=(',', ':')) + ';\n')
