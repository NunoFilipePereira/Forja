"""Processamento de imagem (sobre OpenCV) e reconstrução por silhuetas (visual hull)."""
import numpy as np
import cv2
from scipy import ndimage


def to_gray(img):
    a = np.asarray(img)
    if a.ndim == 3:
        return cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return a


def to_u8(img):
    a = np.asarray(img, float)
    if a.dtype == np.uint8:
        return a
    f = np.isfinite(a)
    if not f.any():
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.nanmin(a[f]), np.nanmax(a[f])
    out = (np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-12) * 255
    return out.astype(np.uint8)


def load(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raw = np.fromfile(path, np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Não foi possível ler {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return img


def load_raw(path, width, height, dtype):
    return np.fromfile(path, dtype=dtype).reshape(height, width)


def save(path, img):
    a = np.asarray(img)
    if a.dtype != np.uint8 and not path.lower().endswith((".tif", ".tiff")):
        a = to_u8(a)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if path.lower().endswith((".s8", ".u8", ".s16", ".u16", ".f32", ".raw")):
        a.tofile(path)
        return
    ok, buf = cv2.imencode("." + path.rsplit(".", 1)[-1], a)
    if not ok:
        raise ValueError("Formato de imagem não suportado")
    buf.tofile(path)


def colorize(img, cmap="viridis"):
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    a = np.asarray(img, float)
    f = np.isfinite(a)
    lo, hi = (np.nanmin(a[f]), np.nanmax(a[f])) if f.any() else (0, 1)
    n = (np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-12)
    rgb = (cm(n)[..., :3] * 255).astype(np.uint8)
    rgb[~f] = 0
    return rgb, (lo, hi)


# --- Limiarização ------------------------------------------------------------
def threshold(img, mode="otsu", value=128, block=31, c=5, invert=False):
    g = to_u8(to_gray(img))
    t = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if mode == "fixo":
        _, out = cv2.threshold(g, value, 255, t)
    elif mode == "otsu":
        _, out = cv2.threshold(g, 0, 255, t | cv2.THRESH_OTSU)
    else:
        out = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, t, block | 1, c)
    return out


# --- Morfologia --------------------------------------------------------------
def morphology(img, op="abrir", size=5, shape="elipse", iterations=1):
    k = cv2.getStructuringElement({"elipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT,
                                   "cruz": cv2.MORPH_CROSS}[shape], (size, size))
    ops = {"erodir": cv2.MORPH_ERODE, "dilatar": cv2.MORPH_DILATE, "abrir": cv2.MORPH_OPEN,
           "fechar": cv2.MORPH_CLOSE, "gradiente": cv2.MORPH_GRADIENT, "tophat": cv2.MORPH_TOPHAT,
           "blackhat": cv2.MORPH_BLACKHAT}
    return cv2.morphologyEx(np.asarray(img), ops[op], k, iterations=iterations)


def fill_holes(img):
    return (ndimage.binary_fill_holes(np.asarray(img) > 0) * 255).astype(np.uint8)


def skeleton(img):
    from skimage.morphology import skeletonize
    return (skeletonize(np.asarray(img) > 0) * 255).astype(np.uint8)


# --- Filtros -----------------------------------------------------------------
def filter_(img, kind="gaussiano", size=5, sigma=1.5):
    a = np.asarray(img)
    size = int(size) | 1
    if kind == "gaussiano":
        return cv2.GaussianBlur(a, (size, size), sigma)
    if kind == "mediana":
        return cv2.medianBlur(to_u8(a) if a.dtype != np.uint8 else a, size)
    if kind == "bilateral":
        return cv2.bilateralFilter(to_u8(a) if a.dtype != np.uint8 else a, size, sigma * 25, sigma * 25)
    if kind == "média":
        return cv2.blur(a, (size, size))
    if kind == "nitidez":
        blur = cv2.GaussianBlur(a, (size, size), sigma)
        return cv2.addWeighted(a, 1.8, blur, -0.8, 0)
    raise ValueError(kind)


def equalize(img):
    a = to_u8(np.asarray(img))
    if a.ndim == 3:
        y = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)
        y[..., 0] = cv2.equalizeHist(y[..., 0])
        return cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)
    return cv2.equalizeHist(a)


# --- Arestas -----------------------------------------------------------------
def edges(img, kind="canny", low=50, high=150):
    g = to_u8(to_gray(img))
    if kind == "canny":
        return cv2.Canny(g, low, high)
    if kind == "sobel":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        return to_u8(np.hypot(gx, gy))
    if kind == "laplaciano":
        return to_u8(np.abs(cv2.Laplacian(g, cv2.CV_32F)))
    raise ValueError(kind)


# --- Características ---------------------------------------------------------
def contours(binary, min_area=20):
    cs, _ = cv2.findContours((np.asarray(binary) > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for i, c in enumerate(cs):
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        M = cv2.moments(c)
        cx, cy = M["m10"] / max(M["m00"], 1e-9), M["m01"] / max(M["m00"], 1e-9)
        (rx, ry), (rw, rh), ang = cv2.minAreaRect(c)
        out.append({"id": len(out) + 1, "área": a, "perímetro": cv2.arcLength(c, True),
                    "centro": (cx, cy), "rect_rodado": (rx, ry, rw, rh, ang), "pts": c[:, 0, :]})
    return out


def circles(img, dp=1.2, min_dist=20, p1=100, p2=30, rmin=5, rmax=0):
    g = to_u8(to_gray(img))
    c = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp, min_dist, param1=p1, param2=p2,
                         minRadius=rmin, maxRadius=rmax)
    return [] if c is None else c[0].tolist()


def corners(img, n=200, quality=0.01, min_dist=10):
    g = to_u8(to_gray(img))
    c = cv2.goodFeaturesToTrack(g, n, quality, min_dist)
    return [] if c is None else c[:, 0, :].tolist()


def draw_overlay(img, cnts=(), circs=(), pts=()):
    base = np.asarray(img)
    base = to_u8(base)
    out = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) if base.ndim == 2 else base.copy()
    for c in cnts:
        cv2.polylines(out, [c["pts"].astype(np.int32)], True, (255, 60, 60), 1)
        cv2.putText(out, str(c["id"]), tuple(int(v) for v in c["centro"]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1)
    for x, y, r in circs:
        cv2.circle(out, (int(x), int(y)), int(r), (60, 255, 60), 1)
    for x, y in pts:
        cv2.drawMarker(out, (int(x), int(y)), (60, 160, 255), cv2.MARKER_CROSS, 6)
    return out


# --- Aritmética --------------------------------------------------------------
def arithmetic(a, b, op):
    A, B = np.asarray(a, float), np.asarray(b, float)
    if A.ndim != B.ndim:  # cor com cinzento: compara em intensidade
        A = A.mean(2) if A.ndim == 3 else A
        B = B.mean(2) if B.ndim == 3 else B
    if A.shape != B.shape:
        B = cv2.resize(B, (A.shape[1], A.shape[0]))
    r = {"somar": A + B, "subtrair": A - B, "diferença abs.": np.abs(A - B), "multiplicar": A * B,
         "dividir": A / np.where(B == 0, 1e-9, B), "média": (A + B) / 2, "máximo": np.maximum(A, B),
         "mínimo": np.minimum(A, B)}[op]
    return r


# --- Geometria de imagem -----------------------------------------------------
def rotate(img, angle):
    a = np.asarray(img)
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(a, M, (w, h))


def resize(img, factor):
    a = np.asarray(img)
    return cv2.resize(a, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)


def crop(img, x, y, w, h):
    return np.asarray(img)[y:y + h, x:x + w]


# --- Calibração de câmara ----------------------------------------------------
def calibrate_chessboard(images, cols=9, rows=6, square=10.0):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    objs, imgs = [], []
    size = None
    for im in images:
        g = to_u8(to_gray(im))
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, (cols, rows))
        if ok:
            c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            objs.append(objp); imgs.append(c)
    if len(objs) < 3:
        raise ValueError(f"Tabuleiro encontrado em {len(objs)} imagens; são precisas pelo menos 3.")
    err, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    return {"erro_rms_px": err, "K": K, "distorção": dist.ravel(), "imagens_usadas": len(objs)}


def undistort(img, K, dist):
    return cv2.undistort(np.asarray(img), K, np.asarray(dist))


# --- Aquisição ---------------------------------------------------------------
def capture(camera_index=0, frames=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmara (verifique as permissões de Câmara do macOS).")
    out = []
    for _ in range(5):
        cap.read()  # aquecimento / exposição automática
    for _ in range(frames):
        ok, f = cap.read()
        if ok:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise RuntimeError("A câmara não devolveu imagens.")
    return out


# --- Visual hull (silhuetas em mesa rotativa) -------------------------------
def visual_hull(silhouettes, angles_deg, px_mm=1.0, resolution=96):
    """Escultura de voxels: mantém os voxels cuja projeção cai dentro de todas as silhuetas.
    Assume câmara ortográfica, eixo de rotação vertical no centro da imagem."""
    from skimage import measure
    import trimesh
    masks = [np.asarray(s) > 0 for s in silhouettes]
    h, w = masks[0].shape
    R = w / 2 * px_mm
    H = h * px_mm
    n = resolution
    xs = np.linspace(-R, R, n); zs = np.linspace(0, H, int(n * H / (2 * R)) + 1)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    occ = np.ones(X.shape, bool)
    for m, a in zip(masks, np.radians(angles_deg)):
        u = (X * np.cos(a) + Y * np.sin(a)) / px_mm + w / 2
        v = h - 1 - Z / px_mm
        ui = np.clip(np.round(u).astype(int), 0, w - 1)
        vi = np.clip(np.round(v).astype(int), 0, h - 1)
        occ &= m[vi, ui]
    if not occ.any():
        raise ValueError("Nenhum voxel sobreviveu: verifique silhuetas/ângulos.")
    if occ.all():
        raise ValueError("As silhuetas cobrem a imagem toda: use imagens binárias com o objeto a branco e o fundo a preto.")
    pad = np.pad(occ, 1).astype(float)
    pad = ndimage.gaussian_filter(pad, 0.8)
    if pad.max() <= 0.5:
        raise ValueError("As silhuetas não se intersetam num volume sólido: verifique os ângulos, a ordem das imagens "
                         "e se o objeto está a branco sobre fundo preto.")
    verts, faces, _, _ = measure.marching_cubes(pad, 0.5)
    step = np.array([xs[1] - xs[0], xs[1] - xs[0], zs[1] - zs[0]])
    verts = (verts - 1) * step + np.array([-R, -R, 0])
    m = trimesh.Trimesh(verts, faces, process=True)
    if m.volume < 0:
        m.invert()
    return m


def demo_silhouettes(n=12, w=200, h=200):
    """Silhuetas sintéticas de um objeto (para testar o visual hull)."""
    angs = np.linspace(0, 360, n, endpoint=False)
    out = []
    for a in np.radians(angs):
        img = np.zeros((h, w), np.uint8)
        # cilindro elíptico + esfera em cima
        rx, ry = 50, 25
        half = np.sqrt((rx * np.cos(a)) ** 2 + (ry * np.sin(a)) ** 2)
        img[h - 110:h, int(w / 2 - half):int(w / 2 + half)] = 255
        cv2.circle(img, (w // 2, h - 140), 35, 255, -1)
        out.append(img)
    return out, angs
