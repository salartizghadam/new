"""Crop the largest face from an image (or a video frame) to 480 px wide for face QC.
usage: facecrop.py IN OUT [--t SECONDS]"""
import sys, cv2

def largest_face(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(g, 1.1, 6, minSize=(60, 60))
    return max(faces, key=lambda f: f[2] * f[3]) if len(faces) else None

def crop(img, pad=0.6, width=480):
    f = largest_face(img)
    if f is None:
        return None
    x, y, w, h = f
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad)))
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (width, int(c.shape[0] * width / c.shape[1])), interpolation=cv2.INTER_CUBIC)

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    if "--t" in sys.argv:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sys.argv[sys.argv.index("--t") + 1]) * 1000)
        ok, img = cap.read()
    else:
        img = cv2.imread(src)
    out = crop(img)
    if out is None:
        sys.exit(f"no face: {src}")
    cv2.imwrite(dst, out)
