"""Tiny proxy between the Shirt Reel Maker page and the Google Gemini API.

Why it exists: the API key must never be inside the web page (anyone could read it with
"View source"). The key lives only here, in the GEMINI_API_KEY environment variable.

Routes (nginx forwards /ai/ to this server):
  GET  /ai/status            -> {"configured": bool, "videosToday": n, "dailyLimit": n}
  POST /ai/image             -> {"mime","data"}  one image (shirt photo with the design on it)
  POST /ai/video             -> {"job": id}      starts a video, returns at once
  GET  /ai/job/<id>          -> {"state": "running|done|error", "error"?}
  GET  /ai/job/<id>/file     -> the finished MP4
Only python3 standard library, no packages to install.
"""
import base64, json, os, re, threading, time, urllib.request, urllib.error, uuid, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = os.environ.get("GEMINI_API_KEY", "").strip()
API = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
IMAGE_MODEL = os.environ.get("AI_IMAGE_MODEL", "gemini-nano-banana-2.1")
VIDEO_MODEL = os.environ.get("AI_VIDEO_MODEL", "gemini-omni-1.1-flash")
DAILY_LIMIT = int(os.environ.get("AI_DAILY_LIMIT", "30"))      # max videos per day (cost guard)
IMAGE_LIMIT = int(os.environ.get("AI_DAILY_IMAGE_LIMIT", "100"))
MAX_BODY = 25 * 1024 * 1024
JOBS, LOCK = {}, threading.Lock()
COUNT = {"day": "", "video": 0, "image": 0}


def today():
    return datetime.date.today().isoformat()


def bump(kind, limit):
    with LOCK:
        if COUNT["day"] != today():
            COUNT.update(day=today(), video=0, image=0)
        if COUNT[kind] >= limit:
            raise ApiError("Daily limit reached (%d %ss). Change AI_DAILY_LIMIT in .env if you need more." % (limit, kind))
        COUNT[kind] += 1


class ApiError(Exception):
    pass


def call(method, url, body=None, timeout=600, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = r.read()
            return out if raw else json.loads(out)
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf8", "replace")
        try:
            msg = json.loads(txt).get("error", {}).get("message", txt)
        except Exception:
            msg = txt
        raise ApiError("Google said %d: %s" % (e.code, str(msg)[:400]))
    except urllib.error.URLError as e:
        raise ApiError("Could not reach Google (%s). Does the NAS have internet?" % e.reason)


def find_media(obj, kind):
    """walk the interaction JSON and return the first content item of this kind (image/video)."""
    if isinstance(obj, dict):
        if obj.get("type") == kind and (obj.get("data") or obj.get("uri")):
            return obj
        for v in obj.values():
            r = find_media(v, kind)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_media(v, kind)
            if r:
                return r
    return None


def clean_img(im):
    if not isinstance(im, dict) or not im.get("data"):
        raise ApiError("Missing image")
    mime = im.get("mime") or "image/png"
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise ApiError("Image must be PNG, JPEG or WebP")
    return {"type": "image", "mime_type": mime, "data": im["data"]}


def make_image(p):
    bump("image", IMAGE_LIMIT)
    prompt = str(p.get("prompt", ""))[:4000]
    imgs = [clean_img(i) for i in (p.get("images") or [])][:4]
    body = {"model": IMAGE_MODEL, "input": [{"type": "text", "text": prompt}] + imgs,
            "response_format": {"type": "image", "mime_type": "image/jpeg",
                                "aspect_ratio": p.get("aspect_ratio") if p.get("aspect_ratio") in ("9:16", "16:9", "1:1", "3:4", "4:3") else "9:16",
                                "image_size": "2K"}}
    res = call("POST", API + "/interactions", body)
    m = find_media(res, "image")
    if not m or not m.get("data"):
        raise ApiError("Google returned no image. Try a different prompt.")
    return {"mime": m.get("mime_type", "image/jpeg"), "data": m["data"]}


def file_id(uri):
    m = re.search(r"files/([A-Za-z0-9_\-]+)", uri or "")
    if not m:
        raise ApiError("Unexpected video link from Google")
    return m.group(1)


def run_video(job, p):
    try:
        prompt = str(p.get("prompt", ""))[:4000]
        content = []
        if p.get("image"):
            content.append(clean_img(p["image"]))
        content.append({"type": "text", "text": prompt})
        body = {"model": VIDEO_MODEL, "input": content if p.get("image") else prompt,
                "response_format": {"type": "video",
                                    "aspect_ratio": "16:9" if p.get("aspect_ratio") == "16:9" else "9:16",
                                    "resolution": p.get("resolution") if p.get("resolution") in ("720p", "1080p") else "720p",
                                    "delivery": "uri"}}
        if p.get("image"):
            body["generation_config"] = {"video_config": {"task": "image_to_video"}}
        res = call("POST", API + "/interactions", body)
        m = find_media(res, "video")
        if not m:
            raise ApiError("Google returned no video. Try a different prompt.")
        if m.get("data"):
            job["bytes"] = base64.b64decode(m["data"])
        else:
            fid = file_id(m["uri"])
            for _ in range(240):                                   # up to ~20 minutes
                st = call("GET", "%s/files/%s" % (API, fid), timeout=60).get("state")
                if st == "ACTIVE":
                    break
                if st == "FAILED":
                    raise ApiError("Google could not finish the video.")
                time.sleep(5)
            else:
                raise ApiError("Timed out waiting for the video.")
            job["bytes"] = call("GET", "%s/files/%s:download?alt=media" % (API, fid), raw=True)
        job["state"] = "done"
    except ApiError as e:
        job["state"], job["error"] = "error", str(e)
    except Exception as e:                                         # never leak internals / key
        job["state"], job["error"] = "error", "Unexpected error: %s" % type(e).__name__


def start_video(p):
    bump("video", DAILY_LIMIT)
    jid = uuid.uuid4().hex[:12]
    job = {"state": "running", "t": time.time()}
    with LOCK:
        for k in [k for k, v in JOBS.items() if time.time() - v["t"] > 3600]:
            JOBS.pop(k)
        JOBS[jid] = job
    threading.Thread(target=run_video, args=(job, p), daemon=True).start()
    return {"job": jid}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/ai/status":
            with LOCK:
                v = COUNT["video"] if COUNT["day"] == today() else 0
            return self.send(200, {"configured": bool(KEY), "videosToday": v, "dailyLimit": DAILY_LIMIT})
        m = re.fullmatch(r"/ai/job/([0-9a-f]+)(/file)?", path)
        if m and m.group(1) in JOBS:
            job = JOBS[m.group(1)]
            if m.group(2):
                if job["state"] != "done":
                    return self.send(409, {"error": "not ready"})
                return self.send(200, job["bytes"], "video/mp4")
            out = {"state": job["state"]}
            if job.get("error"):
                out["error"] = job["error"]
            return self.send(200, out)
        self.send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if not KEY:
                raise ApiError("No API key on the NAS yet. Put GEMINI_API_KEY in the .env file and restart.")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise ApiError("Request too large (max 25 MB)")
            p = json.loads(self.rfile.read(n))
            path = self.path.split("?")[0]
            if path == "/ai/image":
                return self.send(200, make_image(p))
            if path == "/ai/video":
                return self.send(200, start_video(p))
            self.send(404, {"error": "not found"})
        except ApiError as e:
            self.send(400, {"error": str(e)})
        except Exception as e:
            self.send(500, {"error": "Server error: %s" % type(e).__name__})


if __name__ == "__main__":
    print("ai-proxy on :8090, key %s" % ("set" if KEY else "MISSING"), flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8090), H).serve_forever()
