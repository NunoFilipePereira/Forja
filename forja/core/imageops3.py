"""Operações de imagem adicionais: máscaras, cor, campos vetoriais, janelas, interpolação, alinhamento elástico,
histograma 2D, inserção, projeções, caminho mínimo, limiar por dois picos, integração, variância."""
import numpy as np
import cv2
from scipy import ndimage, signal
from .imageops import to_gray, to_u8

INTERP = {"vizinho mais próximo": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cúbica": cv2.INTER_CUBIC,
          "Lanczos": cv2.INTER_LANCZOS4, "área": cv2.INTER_AREA}


def _f(a):
    a = np.asarray(a)
    return a.astype(float) if a.ndim == 2 else a.astype(float).mean(2)


def mask_from_range(img, lo, hi):
    a = _f(img)
    return (((a >= lo) & (a <= hi)) * 255).astype(np.uint8)


def fill_mask(img, mask, value=0.0, inside=True):
    a = np.asarray(img).astype(float).copy()
    m = np.asarray(mask) > 0
    if m.shape != a.shape[:2]:
        raise ValueError("A máscara tem de ter o mesmo tamanho da imagem.")
    a[m if inside else ~m] = value
    return a


def color_layer(img, channel="R"):
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError("A imagem não tem cor.")
    k = "RGB".index(channel)
    out = np.zeros_like(a[..., :3]); out[..., k] = a[..., k]
    return out


def color_to_intensity(img, mode="luminância"):
    a = np.asarray(img, float)
    if a.ndim != 3:
        return a
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return {"luminância": 0.299 * r + 0.587 * g + 0.114 * b, "média": (r + g + b) / 3, "máximo": a[..., :3].max(2),
            "mínimo": a[..., :3].min(2), "saturação": a[..., :3].max(2) - a[..., :3].min(2)}[mode]


def copy_true_color(intensity, color_src):
    """Aplica a cor (matiz e saturação) de uma imagem à intensidade de outra."""
    I = to_u8(_f(intensity))
    c = np.asarray(color_src)
    if c.ndim != 3:
        raise ValueError("A imagem de origem da cor não tem cor.")
    c = cv2.resize(c[..., :3].astype(np.uint8), (I.shape[1], I.shape[0]))
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = I
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def vector_map(img, step=16, scale=1.0, base=None):
    a = ndimage.gaussian_filter(_f(img), 1.5)
    gy, gx = np.gradient(a)
    shown = to_u8(base if base is not None else a)
    out = cv2.cvtColor(shown, cv2.COLOR_GRAY2RGB) if shown.ndim == 2 else shown.copy()
    mag = np.hypot(gx, gy)
    k = step * 0.8 / max(np.percentile(mag, 99), 1e-9) * scale
    for y in range(step // 2, a.shape[0], step):
        for x in range(step // 2, a.shape[1], step):
            dx, dy = gx[y, x] * k, gy[y, x] * k
            if abs(dx) + abs(dy) > 0.5:
                cv2.arrowedLine(out, (x, y), (int(x + dx), int(y + dy)), (255, 60, 60), 1, tipLength=0.35)
    return out


def window(img, kind="Hann"):
    a = _f(img)
    h, w = a.shape
    fn = {"Hann": np.hanning, "Hamming": np.hamming, "Blackman": np.blackman,
          "cosseno": lambda n: np.sin(np.pi * (np.arange(n) + 0.5) / n)}[kind]
    return a * np.outer(fn(h), fn(w))


def resize_interp(img, factor, method="linear"):
    a = np.asarray(img)
    return cv2.resize(a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32), None, fx=factor, fy=factor,
                      interpolation=INTERP[method])


def rotate_expand(img, angle, method="linear", expand=True):
    a = np.asarray(img)
    a = a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32)
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    if expand:
        c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * s + w * c), int(h * c + w * s)
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    else:
        nw, nh = w, h
    return cv2.warpAffine(a, M, (nw, nh), flags=INTERP[method])


def elastic_align(ref, mov):
    """Alinhamento elástico (não rígido) de 'mov' a 'ref' por fluxo ótico denso."""
    A, B = to_u8(_f(ref)), to_u8(_f(mov))
    if A.shape != B.shape:
        B = cv2.resize(B, (A.shape[1], A.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(A, B, None, 0.5, 5, 31, 5, 7, 1.5, 0)
    h, w = A.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    mx, my = (X + flow[..., 0]).astype(np.float32), (Y + flow[..., 1]).astype(np.float32)
    src = np.asarray(mov)
    if src.shape[:2] != A.shape:
        src = cv2.resize(src, (w, h))
    src = src if src.dtype in (np.uint8, np.float32) else src.astype(np.float32)
    return cv2.remap(src, mx, my, cv2.INTER_LINEAR), flow


def hist2d(a, b, bins=128):
    A, B = _f(a).ravel(), _f(b).ravel()
    m = np.isfinite(A) & np.isfinite(B)
    H, xe, ye = np.histogram2d(B[m], A[m], bins=bins)
    return np.log1p(H)[::-1], (xe, ye)


def insert(mother, child, x, y, alpha=1.0):
    M = np.asarray(mother).astype(float).copy()
    C = np.asarray(child).astype(float)
    if M.ndim == 3 and C.ndim == 2:
        C = np.dstack([C] * M.shape[2])
    if M.ndim == 2 and C.ndim == 3:
        C = C.mean(2)
    h, w = C.shape[:2]
    y0, x0 = max(0, y), max(0, x)
    y1, x1 = min(M.shape[0], y + h), min(M.shape[1], x + w)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("A imagem a inserir fica fora da imagem de destino.")
    M[y0:y1, x0:x1] = (1 - alpha) * M[y0:y1, x0:x1] + alpha * C[y0 - y:y1 - y, x0 - x:x1 - x]
    return M.astype(np.asarray(mother).dtype) if np.asarray(mother).dtype == np.uint8 else M


def projections(img):
    a = _f(img)
    return np.nansum(a, axis=0), np.nansum(a, axis=1)


def min_path(img, p0, p1, dark=True):
    """Caminho de custo mínimo entre dois pontos (segue zonas escuras ou claras)."""
    from skimage.graph import route_through_array
    a = _f(img)
    a = (a - np.nanmin(a)) / max(np.nanmax(a) - np.nanmin(a), 1e-9)
    cost = (a if dark else 1 - a) + 1e-3
    path, c = route_through_array(cost, (p0[1], p0[0]), (p1[1], p1[0]), fully_connected=True, geometric=True)
    path = np.array(path)[:, ::-1]
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    return path, float(length), float(c)


def two_peak_threshold(img, bins=256):
    """Limiar no mínimo do histograma entre os dois picos principais (histograma bimodal)."""
    a = _f(img)
    h, e = np.histogram(a[np.isfinite(a)], bins)
    hs = ndimage.gaussian_filter1d(h.astype(float), 2)
    peaks, _ = signal.find_peaks(hs, distance=bins // 10)
    if len(peaks) < 2:
        raise ValueError("O histograma não tem dois picos distintos.")
    p1, p2 = sorted(peaks[np.argsort(hs[peaks])[-2:]])
    seg = hs[p1:p2 + 1]
    lows = np.nonzero(seg <= seg.min() + 1e-9 * max(seg.max(), 1))[0]
    valley = p1 + int(round(lows.mean()))  # centro da zona mais baixa entre os picos
    t = (e[valley] + e[valley + 1]) / 2
    return ((a > t) * 255).astype(np.uint8), float(t)


def integrate(img, axis="X", step=1.0):
    a = np.nan_to_num(_f(img))
    return np.cumsum(a, axis=1 if axis == "X" else 0) * step


def local_variance(img, size=7):
    a = _f(img)
    m = ndimage.uniform_filter(a, size)
    return ndimage.uniform_filter(a * a, size) - m * m


def weighted_mean(imgs, weights):
    S = np.stack([_f(i) for i in imgs]); W = np.stack([np.asarray(w, float) for w in weights])
    if W.shape != S.shape:
        W = np.broadcast_to(W.mean(axis=(1, 2))[:, None, None], S.shape)
    return (S * W).sum(0) / np.maximum(W.sum(0), 1e-12)


def preview_grid(imgs, names, thumb=160, cols=0):
    import math
    n = len(imgs)
    cols = cols or int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    out = np.full((rows * (thumb + 22), cols * (thumb + 6), 3), 40, np.uint8)
    for k, (a, nm) in enumerate(zip(imgs, names)):
        a = to_u8(np.asarray(a, float) if np.asarray(a).dtype != np.uint8 else np.asarray(a))
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        h, w = a.shape[:2]; s = thumb / max(h, w)
        t = cv2.resize(a[..., :3], (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        r, c = divmod(k, cols)
        y, x = r * (thumb + 22), c * (thumb + 6)
        out[y:y + t.shape[0], x:x + t.shape[1]] = t
        cv2.putText(out, nm[:22], (x + 2, y + thumb + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
    return out
