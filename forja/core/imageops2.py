"""Mais operações de imagem: lotes, matemática por pixel, frequência, segmentação, geometria,
formatos adicionais e geradores."""
import os
import numpy as np
import cv2
from scipy import ndimage, signal
from .imageops import to_gray, to_u8


def _f(a):
    a = np.asarray(a)
    return a.astype(float) if a.ndim == 2 else to_gray(a).astype(float) if a.dtype == np.uint8 else a.mean(2)


# --------------------------------------------------------------------------
# Lotes (várias imagens do mesmo tamanho)
# --------------------------------------------------------------------------
def _stack(imgs):
    shp = np.asarray(imgs[0]).shape
    ok = [np.asarray(i, float) for i in imgs if np.asarray(i).shape == shp]
    if len(ok) < len(imgs):
        raise ValueError(f"{len(imgs) - len(ok)} imagem(ns) com tamanho diferente da primeira.")
    return np.stack(ok)


def batch_stat(imgs, kind):
    S = _stack(imgs)
    return {"média": S.mean(0), "mínimo": S.min(0), "máximo": S.max(0), "desvio-padrão": S.std(0),
            "mediana": np.median(S, 0), "soma": S.sum(0), "amplitude (máx−mín)": S.max(0) - S.min(0)}[kind]


def progressive_mean(imgs):
    S = _stack(imgs)
    return list(np.cumsum(S, 0) / np.arange(1, len(S) + 1)[:, None, None] if S.ndim == 3 else
                np.cumsum(S, 0) / np.arange(1, len(S) + 1)[:, None, None, None])


def batch_pairwise(b1, b2, op):
    from .imageops import arithmetic
    if len(b1) != len(b2):
        raise ValueError("Os dois lotes têm de ter o mesmo número de imagens.")
    return [arithmetic(a, b, op) for a, b in zip(b1, b2)]


def join(imgs, direction="horizontal"):
    arrs = [np.asarray(i) for i in imgs]
    if direction == "horizontal":
        h = min(a.shape[0] for a in arrs)
        return np.hstack([a[:h] for a in arrs])
    w = min(a.shape[1] for a in arrs)
    return np.vstack([a[:, :w] for a in arrs])


def collage(imgs, cols=0, gap=4, bg=255):
    arrs = [to_u8(np.asarray(i)) for i in imgs]
    arrs = [cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) if a.ndim == 2 else a for a in arrs]
    n = len(arrs)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    ch = max(a.shape[0] for a in arrs); cw = max(a.shape[1] for a in arrs)
    out = np.full((rows * ch + (rows + 1) * gap, cols * cw + (cols + 1) * gap, 3), bg, np.uint8)
    for k, a in enumerate(arrs):
        r, c = divmod(k, cols)
        y, x = gap + r * (ch + gap), gap + c * (cw + gap)
        out[y:y + a.shape[0], x:x + a.shape[1]] = a
    return out


def stitch(imgs, mode="panorama"):
    arrs = [to_u8(np.asarray(i)) for i in imgs]
    arrs = [cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2BGR) for a in arrs]
    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS)
    status, pano = st.stitch(arrs)
    if status != 0:
        raise ValueError("Não foi possível juntar as imagens (sobreposição insuficiente?).")
    return cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)


def mosaic(img, block=12):
    a = np.asarray(img)
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# Matemática por pixel
# --------------------------------------------------------------------------
def pixel_math(img, op, value=1.0):
    a = np.asarray(img, float)
    with np.errstate(all="ignore"):
        r = {"+ valor": a + value, "× valor": a * value, "− valor": a - value, "÷ valor": a / (value or 1e-12),
             "potência": np.power(np.abs(a), value), "raiz quadrada": np.sqrt(np.abs(a)), "ln": np.log(np.maximum(a, 1e-12)),
             "exp": np.exp(np.clip(a, -50, 50)), "seno": np.sin(a), "cosseno": np.cos(a), "valor absoluto": np.abs(a),
             "negativo": -a}[op]
    return r


def atan2_img(a, b):
    return np.arctan2(np.asarray(a, float), np.asarray(b, float))


def complex_norm(a, b):
    return np.hypot(np.asarray(a, float), np.asarray(b, float))


def gamma(img, g=1.0):
    a = to_u8(np.asarray(img))
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(a, lut)


def stretch(img, low_pct=1.0, high_pct=99.0):
    a = np.asarray(img, float)
    lo, hi = np.nanpercentile(a, [low_pct, high_pct])
    return (np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)


def normalize(img):
    a = np.asarray(img, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / max(hi - lo, 1e-12)


def linear(img, gain=1.0, offset=0.0):
    return np.asarray(img, float) * gain + offset


def clamp(img, lo, hi):
    return np.clip(np.asarray(img, float), lo, hi)


def replace_value(img, old, new, tol=0.0):
    a = np.asarray(img, float).copy()
    a[np.abs(a - old) <= tol] = new
    return a


def set_region(img, x, y, w, h, value):
    a = np.asarray(img).copy()
    a[y:y + h, x:x + w] = value
    return a


def apply_mask(img, mask, outside=0):
    a = np.asarray(img).astype(float).copy()
    m = np.asarray(mask) > 0
    if m.shape != a.shape[:2]:
        raise ValueError("A máscara tem de ter o mesmo tamanho da imagem.")
    a[~m] = outside
    return a


def add_noise(img, kind="gaussiano", amount=10.0, rng=None):
    rng = rng or np.random.default_rng()
    a = np.asarray(img, float)
    if kind == "gaussiano":
        return a + rng.normal(0, amount, a.shape)
    if kind == "sal e pimenta":
        out = a.copy(); r = rng.random(a.shape[:2])
        p = amount / 100
        out[r < p / 2] = a.min(); out[r > 1 - p / 2] = a.max()
        return out
    return a * (1 + rng.normal(0, amount / 100, a.shape))  # speckle (multiplicativo)


# --------------------------------------------------------------------------
# Frequência
# --------------------------------------------------------------------------
def fft(img):
    F = np.fft.fftshift(np.fft.fft2(_f(img)))
    return {"potência (log)": np.log1p(np.abs(F) ** 2), "real": F.real, "imaginária": F.imag,
            "fase": np.angle(F), "_complex": F}


def ifft(F):
    f = np.fft.ifft2(np.fft.ifftshift(F))
    return {"real": f.real, "imaginária": f.imag, "fase": np.angle(f), "módulo": np.abs(f)}


def fft_filter(img, kind="passa-baixo", r1=20.0, r2=60.0):
    a = _f(img)
    F = np.fft.fftshift(np.fft.fft2(a))
    h, w = a.shape
    Y, X = np.ogrid[:h, :w]
    R = np.hypot(X - w / 2, Y - h / 2)
    M = {"passa-baixo": R <= r1, "passa-alto": R >= r1, "passa-banda": (R >= r1) & (R <= r2),
         "rejeita-banda": (R < r1) | (R > r2)}[kind]
    M = ndimage.gaussian_filter(M.astype(float), 2)
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * M)))


def hilbert_x(img):
    h = signal.hilbert(_f(img) - _f(img).mean(1, keepdims=True), axis=1)
    return {"real": h.real, "imaginária": h.imag, "potência": np.abs(h), "fase": np.angle(h)}


def wiener(img, size=5):
    return signal.wiener(_f(img), size)


# --------------------------------------------------------------------------
# Segmentação / análise
# --------------------------------------------------------------------------
def gradient(img, kind="magnitude", distance=1):
    a = _f(img)
    gy, gx = np.gradient(ndimage.uniform_filter(a, distance) if distance > 1 else a)
    return {"magnitude": np.hypot(gx, gy), "X": gx, "Y": gy, "direção": np.degrees(np.arctan2(gy, gx))}[kind]


def distance_transform(binary):
    return ndimage.distance_transform_edt(np.asarray(binary) > 0)


def labeling(binary):
    lab, n = ndimage.label(np.asarray(binary) > 0)
    return lab, n


def watershed(img, min_distance=10):
    from skimage.segmentation import watershed as ws
    from skimage.feature import peak_local_max
    b = np.asarray(img) > 0
    if not b.any() or b.all():
        from .imageops import threshold
        b = threshold(img, "otsu") > 0
    d = ndimage.distance_transform_edt(b)
    coords = peak_local_max(d, min_distance=min_distance, labels=b)
    markers = np.zeros(d.shape, int)
    for k, (r, c) in enumerate(coords, 1):
        markers[r, c] = k
    return ws(-d, markers, mask=b)


def label_stats(lab):
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == i
        cy, cx = ndimage.center_of_mass(m)
        out.append({"id": i, "área": int(m.sum()), "centro": (cx + sl[1].start, cy + sl[0].start),
                    "caixa": (sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start)})
    return out


def local_extrema(img, size=15, kind="máximos", threshold=None):
    a = _f(img)
    f = ndimage.maximum_filter(a, size) if kind == "máximos" else ndimage.minimum_filter(a, size)
    m = a == f
    if threshold is not None:
        m &= (a >= threshold) if kind == "máximos" else (a <= threshold)
    ys, xs = np.nonzero(m)
    return list(zip(xs.tolist(), ys.tolist()))


def harris(img, block=2, ksize=3, k=0.04, rel=0.01):
    g = np.float32(_f(img))
    R = cv2.cornerHarris(g, block, ksize, k)
    ys, xs = np.nonzero(R > rel * R.max())
    return R, list(zip(xs.tolist(), ys.tolist()))


def isef_edges(img, strength=0.5):
    """Filtro exponencial simétrico infinito (Shen–Castan) seguido de deteção de passagens por zero."""
    a = _f(img)
    b = np.clip(strength, 0.01, 0.99)
    def smooth(x, axis):
        y = signal.lfilter([1 - b], [1, -b], x, axis=axis)
        y = np.flip(signal.lfilter([1 - b], [1, -b], np.flip(y, axis), axis=axis), axis)
        return y
    s = smooth(smooth(a, 0), 1)
    bli = s - a
    zc = np.zeros(a.shape, bool)
    zc[:, 1:] |= np.sign(bli[:, 1:]) != np.sign(bli[:, :-1])
    zc[1:, :] |= np.sign(bli[1:, :]) != np.sign(bli[:-1, :])
    g = np.hypot(*np.gradient(s))
    return ((zc & (g > g.mean() + g.std())) * 255).astype(np.uint8)


def hough_lines(img, thr=80, min_len=30, max_gap=10):
    from .imageops import edges
    e = edges(img, "canny") if np.asarray(img).dtype != np.uint8 or len(np.unique(img)) > 2 else np.asarray(img)
    L = cv2.HoughLinesP(e, 1, np.pi / 180, thr, minLineLength=min_len, maxLineGap=max_gap)
    return [] if L is None else np.asarray(L).reshape(-1, 4).tolist()


def fit_ellipses(binary, min_area=50):
    cs, _ = cv2.findContours((np.asarray(binary) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [cv2.fitEllipse(c) for c in cs if len(c) >= 5 and cv2.contourArea(c) >= min_area]


def bounding_shapes(binary, kind="caixa mínima"):
    pts = cv2.findNonZero((np.asarray(binary) > 0).astype(np.uint8))
    if pts is None:
        raise ValueError("Imagem sem pixels de objeto.")
    if kind == "caixa mínima":
        r = cv2.minAreaRect(pts); return "rect", cv2.boxPoints(r), r
    if kind == "círculo mínimo":
        (x, y), rad = cv2.minEnclosingCircle(pts); return "circle", (x, y, rad), None
    if kind == "caixa alinhada":
        x, y, w, h = cv2.boundingRect(pts); return "rect", np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float), (x, y, w, h)
    hull = cv2.convexHull(pts)[:, 0, :]; return "poly", hull, None


def draw_shapes(img, polys=(), circles=(), lines=(), ellipses=(), pts=()):
    base = to_u8(np.asarray(img))
    out = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) if base.ndim == 2 else base.copy()
    for p in polys:
        cv2.polylines(out, [np.asarray(p, np.int32)], True, (255, 70, 70), 1)
    for x, y, r in circles:
        cv2.circle(out, (int(x), int(y)), int(r), (70, 255, 70), 1)
    for x1, y1, x2, y2 in lines:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)
    for e in ellipses:
        cv2.ellipse(out, e, (0, 200, 255), 1)
    for x, y in pts:
        cv2.drawMarker(out, (int(x), int(y)), (255, 0, 255), cv2.MARKER_CROSS, 7)
    return out


def template_match(img, tpl):
    a, t = np.float32(_f(img)), np.float32(_f(tpl))
    R = cv2.matchTemplate(a, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(R)
    return R, loc, mx


def histogram(img, bins=256):
    a = np.asarray(img)
    if a.ndim == 3:
        return {c: np.histogram(a[..., i], bins, (0, 255))[0] for i, c in enumerate("RGB")}
    f = np.isfinite(a.astype(float))
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    h, edges = np.histogram(a[f], bins, (lo, hi if hi > lo else lo + 1))
    return {"I": h, "_edges": edges}


def histogram_plot(img, bins=256, w=640, h=320):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = histogram(img, bins)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    cols = {"R": "red", "G": "green", "B": "blue", "I": "black"}
    for k, v in H.items():
        if k.startswith("_"):
            continue
        x = H["_edges"][:-1] if "_edges" in H else np.arange(len(v))
        ax.plot(x, v, color=cols[k], lw=1)
    ax.set_xlabel("valor"); ax.set_ylabel("pixels"); fig.tight_layout()
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out


def profile(img, x1, y1, x2, y2):
    n = int(np.hypot(x2 - x1, y2 - y1)) + 1
    xs, ys = np.linspace(x1, x2, n), np.linspace(y1, y2, n)
    return ndimage.map_coordinates(_f(img), [ys, xs], order=1)


# --------------------------------------------------------------------------
# Cor e geometria
# --------------------------------------------------------------------------
def split_rgb(img):
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError("A imagem não tem cor.")
    return [a[..., i] for i in range(3)]


def merge_rgb(r, g, b):
    return np.dstack([to_u8(r), to_u8(g), to_u8(b)])


def flip(img, mode):
    a = np.asarray(img)
    return a[:, ::-1] if mode == "horizontal" else a[::-1] if mode == "vertical" else a[::-1, ::-1]


def rot90(img, clockwise=True):
    return np.rot90(np.asarray(img), -1 if clockwise else 1)


def polar(img, inverse=False):
    a = np.asarray(img)
    h, w = a.shape[:2]
    c = (w / 2, h / 2); R = min(c)
    flags = cv2.WARP_POLAR_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    src = a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32)
    return cv2.warpPolar(src, (w, h), c, R, flags)


def projective(img, M, size=None):
    a = np.asarray(img)
    h, w = a.shape[:2]
    return cv2.warpPerspective(a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32),
                               np.asarray(M, float), size or (w, h))


def align_images(ref, mov, mode="projetivo"):
    """Alinha 'mov' a 'ref' com pontos ORB + RANSAC."""
    a, b = to_u8(to_gray(np.asarray(ref))), to_u8(to_gray(np.asarray(mov)))
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(a, None); k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        raise ValueError("Sem pontos característicos suficientes.")
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d2, d1), key=lambda x: x.distance)[:500]
    if len(m) < 8:
        raise ValueError("Correspondências insuficientes para alinhar.")
    p2 = np.float32([k2[x.queryIdx].pt for x in m]); p1 = np.float32([k1[x.trainIdx].pt for x in m])
    if mode == "projetivo":
        M, _ = cv2.findHomography(p2, p1, cv2.RANSAC, 3.0)
        out = cv2.warpPerspective(np.asarray(mov), M, (a.shape[1], a.shape[0]))
    else:
        M, _ = cv2.estimateAffinePartial2D(p2, p1, method=cv2.RANSAC)
        out = cv2.warpAffine(np.asarray(mov), M, (a.shape[1], a.shape[0]))
    return out, M


def halftone(img, kind="pontos", cell=8):
    g = _f(img) / 255.0 if np.asarray(img).dtype == np.uint8 or np.nanmax(img) > 1 else _f(img)
    g = np.clip(g, 0, 1)
    h, w = g.shape
    out = np.full((h, w), 255, np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            v = 1 - g[y:y + cell, x:x + cell].mean()
            if kind == "pontos":
                r = int(round(v * cell * 0.71))
                if r > 0:
                    cv2.circle(out, (x + cell // 2, y + cell // 2), r, 0, -1)
            else:
                t = int(round(v * cell))
                if t > 0:
                    out[y + (cell - t) // 2:y + (cell - t) // 2 + t, x:x + cell] = 0
    return out


def morph_photos(a, b, t=0.5):
    A, B = np.asarray(a, float), np.asarray(b, float)
    if A.shape != B.shape:
        B = cv2.resize(B, (A.shape[1], A.shape[0]))
    return (A * (1 - t) + B * t).astype(np.asarray(a).dtype)


def rasterize_shapes(shapes, px_mm=0.1, margin=2.0):
    """Vetor → bitmap: desenha formas 2D fechadas como imagem binária."""
    from .shapes2d import iter_paths
    b = np.array([s.geom.bounds for s in shapes])
    x0, y0 = b[:, 0].min() - margin, b[:, 1].min() - margin
    x1, y1 = b[:, 2].max() + margin, b[:, 3].max() + margin
    W, H = int((x1 - x0) / px_mm) + 1, int((y1 - y0) / px_mm) + 1
    if W * H > 60e6:
        raise ValueError("Imagem demasiado grande: aumente o tamanho do pixel.")
    out = np.zeros((H, W), np.uint8)

    def px(coords):
        c = np.asarray(coords)
        return np.column_stack([(c[:, 0] - x0) / px_mm, (y1 - c[:, 1]) / px_mm]).astype(np.int32)

    for s in shapes:
        g = s.geom
        if s.closed:
            for poly in (g.geoms if hasattr(g, "geoms") else [g]):
                cv2.fillPoly(out, [px(poly.exterior.coords)], 255)
                for hole in poly.interiors:
                    cv2.fillPoly(out, [px(hole.coords)], 0)
        else:
            for p in iter_paths(g):
                cv2.polylines(out, [px(p)], False, 255, 1)
    return out


def mandelbrot(w=800, h=600, cx=-0.6, cy=0.0, zoom=1.0, iters=200):
    x = np.linspace(cx - 1.75 / zoom, cx + 1.75 / zoom, w)
    y = np.linspace(cy - 1.3 / zoom, cy + 1.3 / zoom, h)
    C = x[None, :] + 1j * y[:, None]
    Z = np.zeros_like(C); N = np.zeros(C.shape, int)
    for k in range(iters):
        m = np.abs(Z) <= 2
        Z[m] = Z[m] ** 2 + C[m]
        N[m] += 1
    return N.astype(float)


def synthetic(kind, w=512, h=512, period=32.0):
    Y, X = np.mgrid[:h, :w]
    if kind == "gaussiana":
        return np.exp(-(((X - w / 2) / (w / 6)) ** 2 + ((Y - h / 2) / (h / 6)) ** 2))
    if kind == "rampa":
        return X / (w - 1)
    if kind == "xadrez":
        return (((X // period) + (Y // period)) % 2).astype(float)
    if kind == "dente de serra":
        return (X % period) / period
    if kind == "anéis":
        return 0.5 + 0.5 * np.cos(2 * np.pi * np.hypot(X - w / 2, Y - h / 2) / period)
    return np.random.default_rng().random((h, w))


# --------------------------------------------------------------------------
# Formatos
# --------------------------------------------------------------------------
RAW_TYPES = {"int8": "s8", "uint8": "u8", "int16": "s16", "uint16": "u16", "int32": "s32", "uint32": "u32",
             "float32": "f32", "float64": "dbl"}


def load_raw(path, w, h, dtype, offset=0, little=True, depth=1):
    dt = np.dtype(dtype).newbyteorder("<" if little else ">")
    a = np.fromfile(path, dtype=dt, offset=offset, count=w * h * depth)
    if a.size < w * h * depth:
        raise ValueError(f"O ficheiro tem {a.size} valores; esperados {w * h * depth}.")
    return a.reshape((depth, h, w) if depth > 1 else (h, w)).astype(dtype)


def load_ascii(path):
    rows = []
    for line in open(path, errors="ignore"):
        line = line.replace(",", " ").replace(";", " ").replace("\t", " ").strip()
        if not line or line[0] in "#%":
            continue
        try:
            rows.append([float(v) for v in line.split()])
        except ValueError:
            continue
    if not rows:
        raise ValueError("Sem dados numéricos.")
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows])


def save_ascii(path, a, fmt="%.6g"):
    np.savetxt(path, np.asarray(a, float), fmt=fmt)


def load_dicom(path):
    import pydicom
    d = pydicom.dcmread(path)
    a = d.pixel_array.astype(float)
    a = a * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
    return a


def load_video(path, step=1, max_frames=500):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo.")
    out = []; k = 0
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        k += 1
    cap.release()
    return out


def save_video(path, imgs, fps=10):
    arrs = [to_u8(np.asarray(i)) for i in imgs]
    h, w = arrs[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*("MJPG" if path.lower().endswith(".avi") else "mp4v")), fps, (w, h))
    for a in arrs:
        a = cv2.resize(a, (w, h))
        vw.write(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    vw.release()


def load_folder(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    from .imageops import load
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    return [(f, load(os.path.join(folder, f))) for f in files]
