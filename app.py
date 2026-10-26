import base64
import datetime
import io
import ipaddress
import math
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from werkzeug.security import safe_join

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from flask import Flask, Response, abort, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
PHOTO_DIR = BASE_DIR / "photos"
CERT_DIR = BASE_DIR / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"
PORT = 8443
SHEET_DPI = 600
_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
_cv_lock = threading.Lock()
LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights
ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def cert_matches(ip):
    if not (CERT_FILE.exists() and KEY_FILE.exists()):
        return False
    try:
        cert = x509.load_pem_x509_certificate(CERT_FILE.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        ips = san.get_values_for_type(x509.IPAddress)
        expires = cert.not_valid_after_utc
        soon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
        return ipaddress.ip_address(ip) in ips and expires > soon
    except Exception:
        return False


def ensure_cert(ip):
    """Self-signed cert with the LAN IP in the SAN. Chrome needs HTTPS for camera access."""
    if cert_matches(ip):
        return
    CERT_DIR.mkdir(exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "camera-app")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address(ip)),
    ]
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def new_photo_name(ext):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:6]}{ext}"


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/single")
def single_page():
    return send_from_directory(BASE_DIR, "single.html")


@app.get("/camera")
def camera_page():
    return send_from_directory(BASE_DIR, "camera.html")


@app.get("/view")
def view_page():
    return send_from_directory(BASE_DIR, "view.html")


@app.get("/assets/<name>")
def assets(name):
    if name not in ("app.css", "guide.js"):
        abort(404)
    return send_from_directory(BASE_DIR, name)


@app.get("/api/photos")
def list_photos():
    files = sorted(
        (p for p in PHOTO_DIR.iterdir() if p.suffix.lower() in ALLOWED_TYPES.values()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return jsonify([p.name for p in files])


@app.post("/api/photos")
def upload_photo():
    f = request.files.get("photo")
    if f is None:
        abort(400, "No photo field")
    ext = ALLOWED_TYPES.get(f.mimetype)
    if ext is None:
        abort(415, "Unsupported image type")
    smooth, slim = retouch_amounts(request.form)
    data, applied = retouch_bytes(f.read(), smooth, slim)
    if applied:
        ext = ".jpg"
    name = new_photo_name(ext)
    (PHOTO_DIR / name).write_bytes(data)
    return jsonify({"name": name, "retouched": applied}), 201


@app.get("/photos/<path:name>")
def get_photo(name):
    return send_from_directory(PHOTO_DIR, name)


def other_ips(primary):
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    found = {i[4][0] for i in infos}
    return sorted(x for x in found if x != primary and not x.startswith(("127.", "169.254.")))


def background_stats(img):
    """Sample the left, right and top margins of the 35 x 45 mm frame (where the wall should be)."""
    h, w = img.shape[:2]
    xs, ys, yt = int(w * 0.12), int(h * 0.67), int(h * 0.08)
    left, right, top = img[:ys, :xs], img[:ys, w - xs:], img[:yt, xs:w - xs]
    px = np.concatenate([a.reshape(-1, 3) for a in (left, right, top)]).astype(np.float32)
    b, g, r = px.mean(axis=0)
    lum = px @ LUMA
    lum_l = float((left.reshape(-1, 3).astype(np.float32) @ LUMA).mean())
    lum_r = float((right.reshape(-1, 3).astype(np.float32) @ LUMA).mean())
    return {
        "r": round(float(r), 1), "g": round(float(g), 1), "b": round(float(b), 1),
        "lum": round(float(lum.mean()), 1), "std": round(float(lum.std()), 1),
        "lr": round(abs(lum_l - lum_r), 1),
    }


def decode_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        abort(400, "Bad image")
    return img


# ---- Optional subtle retouching (off unless the user asks for it) ----
MAX_SMOOTH_ALPHA = 0.6   # at slider 100: blend this much of the smoothed layer into skin
MAX_SLIM = 0.07          # at slider 100: cheeks and jaw pulled in by up to 7 %


def clamp_pct(v):
    return max(0, min(100, int(v or 0)))


def retouch_amounts(src):
    """Read smooth and slim (0-100) from request.args or request.form."""
    return clamp_pct(src.get("smooth", 0, type=int)), clamp_pct(src.get("slim", 0, type=int))


def find_face(img):
    """Largest face as (x, y, w, h) in full-resolution pixels, or None."""
    h, w = img.shape[:2]
    scale = 560 / w if w > 560 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    with _cv_lock:
        faces = _face.detectMultiScale(gray, 1.1, 6, minSize=(int(small.shape[1] * 0.2),) * 2)
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)


def smooth_skin(img, face, amount):
    """Soften fine wrinkles: edge preserving blur, blended in only on skin coloured pixels of the face."""
    x, y, fw, fh = face
    h, w = img.shape[:2]
    d = max(5, int(fw * 0.035) | 1)
    low = cv2.bilateralFilter(img, d, 28, d * 1.5)
    low = cv2.bilateralFilter(low, d, 20, d * 1.5)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (40, 133, 77), (255, 173, 127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    zone = np.zeros((h, w), np.uint8)
    cv2.ellipse(zone, (x + fw // 2, y + int(fh * 0.55)), (int(fw * 0.5), int(fh * 0.62)), 0, 0, 360, 255, -1)
    # keep eyes, brows and mouth crisp
    cv2.rectangle(zone, (x, y + int(fh * 0.22)), (x + fw, y + int(fh * 0.42)), 0, -1)
    cv2.ellipse(zone, (x + fw // 2, y + int(fh * 0.82)), (int(fw * 0.26), int(fh * 0.09)), 0, 0, 360, 0, -1)
    mask = cv2.bitwise_and(skin, zone)
    k = max(3, int(fw * 0.06) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0

    # under-eye and forehead lines sit in the excluded band, so allow a gentler pass there
    band = np.zeros((h, w), np.float32)
    cv2.rectangle(band, (x + int(fw * 0.1), y + int(fh * 0.08)), (x + int(fw * 0.9), y + int(fh * 0.2)), 1.0, -1)
    cv2.rectangle(band, (x + int(fw * 0.12), y + int(fh * 0.42)), (x + int(fw * 0.88), y + int(fh * 0.52)), 1.0, -1)
    band = cv2.GaussianBlur(band, (k, k), 0)
    soft_skin = cv2.GaussianBlur(skin, (k, k), 0).astype(np.float32) / 255.0
    mask = np.maximum(mask, band * soft_skin * 0.8)

    alpha = (mask * amount * MAX_SMOOTH_ALPHA)[..., None]
    return (img * (1 - alpha) + low * alpha).astype(np.uint8)


def slim_face(img, face, amount):
    """Pull the cheeks and jaw slightly towards the centre line. Smooth falloff, plain background stays plain."""
    x, y, fw, fh = face
    h, w = img.shape[:2]
    cx, cy = x + fw / 2, y + fh * 0.72
    sx, sy = fw * 0.75, fh * 0.38
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    weight = np.exp(-(((xs - cx) ** 2) / (2 * sx ** 2) + ((ys - cy) ** 2) / (2 * sy ** 2)))
    map_x = cx + (xs - cx) * (1 + MAX_SLIM * amount * weight)
    return cv2.remap(img, map_x.astype(np.float32), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def retouch_image(img, smooth, slim):
    """Returns (image, applied). No face found means the image comes back untouched."""
    if smooth <= 0 and slim <= 0:
        return img, False
    face = find_face(img)
    if face is None:
        return img, False
    if smooth > 0:
        img = smooth_skin(img, face, smooth / 100.0)
    if slim > 0:
        img = slim_face(img, face, slim / 100.0)
    return img, True


def retouch_bytes(data, smooth, slim):
    """Encoded image in, encoded JPEG out. Returns (bytes, applied)."""
    if smooth <= 0 and slim <= 0:
        return data, False
    out, applied = retouch_image(decode_image(data), smooth, slim)
    if not applied:
        return data, False
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return (buf.tobytes(), True) if ok else (data, False)


@app.post("/api/retouch")
def retouch_preview():
    smooth, slim = retouch_amounts(request.args)
    data, applied = retouch_bytes(request.get_data(), smooth, slim)
    resp = no_store(Response(data, mimetype="image/jpeg" if applied else (request.mimetype or "image/jpeg")))
    resp.headers["X-Retouched"] = "1" if applied else "0"
    return resp


@app.post("/api/analyze")
def analyze():
    return jsonify(measure(decode_image(request.get_data())))


def measure(img):
    """Measure one preview frame (already cropped to 7:9). Returns raw numbers, fractions of the frame."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = {"face": None, "eyes": [], "tilt": None, "bg": background_stats(img)}
    with _cv_lock:
        faces = _face.detectMultiScale(gray, 1.1, 6, minSize=(int(w * 0.2), int(w * 0.2)))
        if len(faces) == 0:
            return out
        x, y, fw, fh = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
        roi = gray[y:y + fh, x:x + fw]
        eyes = _eye.detectMultiScale(roi[: int(fh * 0.6)], 1.1, 5, minSize=(int(fw * 0.12), int(fw * 0.12)))
    out["face"] = {"x": x / w, "y": y / h, "w": fw / w, "h": fh / h}

    left = [e for e in eyes if e[0] + e[2] / 2 < fw * 0.5]
    right = [e for e in eyes if e[0] + e[2] / 2 >= fw * 0.5]
    if left and right:
        pts = []
        for e in (max(left, key=lambda e: e[2] * e[3]), max(right, key=lambda e: e[2] * e[3])):
            pts.append((x + e[0] + e[2] / 2, y + e[1] + e[3] / 2))
        out["eyes"] = [[px / w, py / h] for px, py in pts]
        out["tilt"] = round(math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])), 2)

    inner = roi[int(fh * 0.2): int(fh * 0.95), int(fw * 0.1): int(fw * 0.9)]
    mean = float(inner.mean())
    half = inner.shape[1] // 2
    out["sharp"] = round(float(cv2.Laplacian(cv2.resize(roi, (160, 160)), cv2.CV_64F).var()), 1)
    out["lum"] = round(mean, 1)
    out["clip_hi"] = round(float((inner >= 250).mean()), 4)
    out["clip_lo"] = round(float((inner <= 8).mean()), 4)
    out["lr"] = round(abs(float(inner[:, :half].mean()) - float(inner[:, half:].mean())) / max(mean, 1.0), 3)
    return out


# ---- Remote camera relay: one device streams frames, another views them and triggers the shutter ----
relay_lock = threading.Lock()
relay = {"seq": 0, "frame": None, "analysis": None, "ts": 0.0, "viewer_ts": 0.0,
         "cmd": 0, "pending": None, "pending_info": None, "pending_n": 0}
ONLINE_SECS = 3


def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/relay/frame")
def relay_frame():
    """Camera device pushes a 7:9 preview frame. The reply carries the latest shutter command."""
    data = request.get_data()
    img = decode_image(data)
    small = img if img.shape[1] == 420 else cv2.resize(img, (420, 540), interpolation=cv2.INTER_AREA)
    analysis = measure(small)
    with relay_lock:
        relay.update(seq=relay["seq"] + 1, frame=data, analysis=analysis, ts=time.time())
        viewer = time.time() - relay["viewer_ts"] < ONLINE_SECS
        return jsonify({"cmd": relay["cmd"], "viewer": viewer})


@app.get("/api/relay/state")
def relay_state():
    after = request.args.get("after", 0, type=int)
    now = time.time()
    with relay_lock:
        relay["viewer_ts"] = now
        online = now - relay["ts"] < ONLINE_SECS
        out = {"online": online, "seq": relay["seq"], "cmd": relay["cmd"], "frame": None,
               "analysis": relay["analysis"] if online else None, "pending": relay["pending_info"]}
        if online and relay["seq"] > after:
            out["frame"] = base64.b64encode(relay["frame"]).decode()
    return no_store(jsonify(out))


@app.post("/api/relay/shoot")
def relay_shoot():
    with relay_lock:
        if time.time() - relay["ts"] >= ONLINE_SECS:
            abort(409, "Camera is offline")
        relay["pending"] = None
        relay["pending_info"] = None
        relay["cmd"] += 1
        return jsonify({"cmd": relay["cmd"]})


@app.post("/api/relay/photo")
def relay_photo():
    """Camera device uploads the full resolution photo after a shutter command."""
    data = request.get_data()
    try:
        w, h = Image.open(io.BytesIO(data)).size
    except Exception:
        abort(400, "Bad image")
    with relay_lock:
        relay["pending_n"] += 1
        relay["pending"] = data
        relay["pending_info"] = {"n": relay["pending_n"], "w": w, "h": h}
    return jsonify({"ok": True})


@app.get("/api/relay/pending.jpg")
def relay_pending():
    with relay_lock:
        data = relay["pending"]
    if data is None:
        abort(404)
    smooth, slim = retouch_amounts(request.args)
    data, _ = retouch_bytes(data, smooth, slim)
    return no_store(Response(data, mimetype="image/jpeg"))


@app.post("/api/relay/save")
def relay_save():
    smooth, slim = retouch_amounts(request.args)
    with relay_lock:
        data = relay["pending"]
        if data is None:
            abort(404)
        relay["pending"] = None
        relay["pending_info"] = None
    data, applied = retouch_bytes(data, smooth, slim)
    name = new_photo_name(".jpg")
    (PHOTO_DIR / name).write_bytes(data)
    return jsonify({"name": name, "retouched": applied}), 201


@app.post("/api/relay/discard")
def relay_discard():
    with relay_lock:
        relay["pending"] = None
        relay["pending_info"] = None
    return jsonify({"ok": True})


@app.get("/api/sheet/<path:name>")
def sheet(name):
    """10 x 15 cm print sheet with 8 photos of 35 x 45 mm at 600 dpi."""
    path = safe_join(str(PHOTO_DIR), name)
    if path is None or not Path(path).is_file():
        abort(404)
    mm = lambda v: round(v / 25.4 * SHEET_DPI)
    pw, ph, sw, sh, gap = mm(35), mm(45), mm(150), mm(100), mm(2)
    img = Image.open(path).convert("RGB")
    iw, ih = img.size
    if iw * 9 > ih * 7:  # too wide: centre crop to 7:9
        nw = round(ih * 7 / 9)
        img = img.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
    else:
        nh = round(iw * 9 / 7)
        img = img.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
    photo = img.resize((pw, ph), Image.LANCZOS)
    page = Image.new("RGB", (sw, sh), "white")
    cols, rows = 4, 2
    mx = (sw - cols * pw - (cols - 1) * gap) // 2
    my = (sh - rows * ph - (rows - 1) * gap) // 2
    for r in range(rows):
        for c in range(cols):
            page.paste(photo, (mx + c * (pw + gap), my + r * (ph + gap)))
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=95, dpi=(SHEET_DPI, SHEET_DPI))
    return Response(buf.getvalue(), mimetype="image/jpeg")


def print_banner(ip):
    url = f"https://{ip}:{PORT}"
    extra = other_ips(ip)
    lines = [
        "CAMERA WEB APP IS RUNNING",
        "",
        "Open the menu on your phone or laptop (same Wi-Fi):",
        url,
        "",
        "Direct links:",
        "1 One device:      " + url + "/single",
        "2 Camera device:   " + url + "/camera",
        "3 Viewer device:   " + url + "/view",
        "",
        "Chrome will warn about the certificate:",
        "tap Advanced, then Proceed to " + ip,
        "",
        *(["If that fails, try: " + ", ".join(f"https://{x}:{PORT}" for x in extra), ""] if extra else []),
        "Photos are saved in: " + str(PHOTO_DIR),
        "Press Ctrl+C to stop",
    ]
    width = max(len(x) for x in lines) + 4
    print()
    print("+" + "-" * width + "+")
    for line in lines:
        print("|  " + line.ljust(width - 2) + "|")
    print("+" + "-" * width + "+")
    print()


def main():
    missing = [f for f in ("index.html", "single.html", "view.html", "camera.html", "app.css", "guide.js") if not (BASE_DIR / f).exists()]
    if missing:
        print("Missing next to app.py: " + ", ".join(missing))
        sys.exit(1)
    PHOTO_DIR.mkdir(exist_ok=True)
    ip = get_lan_ip()
    if ip.startswith("127."):
        print("No network connection found. Connect to Wi-Fi and try again.")
        sys.exit(1)
    ensure_cert(ip)
    print_banner(ip)
    app.run(host="0.0.0.0", port=PORT, ssl_context=(str(CERT_FILE), str(KEY_FILE)), debug=False)


if __name__ == "__main__":
    main()
