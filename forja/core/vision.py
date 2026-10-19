"""Visão por computador: rostos, marcadores, códigos QR, alvos de referência, foco total, movimento, efeitos."""
import os
import numpy as np
import cv2
from scipy import ndimage
from .imageops import to_gray, to_u8

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _rgb(img):
    a = to_u8(np.asarray(img))
    return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) if a.ndim == 2 else a[..., :3].copy()


# --------------------------------------------------------------------------
# Rostos (YuNet)
# --------------------------------------------------------------------------
def detect_faces(img, score=0.7):
    """Devolve lista de rostos: caixa (x, y, w, h), 5 pontos (olhos, nariz, cantos da boca) e confiança."""
    rgb = _rgb(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(os.path.join(DATA, "face_detection_yunet.onnx"), "", (w, h), score, 0.3, 5000)
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    out = []
    if faces is not None:
        for f in faces:
            out.append({"caixa": tuple(float(v) for v in f[:4]), "pontos": f[4:14].reshape(5, 2).tolist(), "confiança": float(f[14])})
    return out


def draw_faces(img, faces):
    out = _rgb(img)
    names = ["olho dir.", "olho esq.", "nariz", "boca dir.", "boca esq."]
    for k, f in enumerate(faces, 1):
        x, y, w, h = (int(v) for v in f["caixa"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (60, 255, 60), 2)
        cv2.putText(out, f"{k}: {f['confiança']:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 255, 60), 1)
        for (px, py), n in zip(f["pontos"], names):
            cv2.circle(out, (int(px), int(py)), 3, (255, 80, 80), -1)
    return out


def anonymize_faces(img, faces, mode="desfocar"):
    out = _rgb(img)
    for f in faces:
        x, y, w, h = (int(v) for v in f["caixa"])
        x, y = max(x, 0), max(y, 0)
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        if mode == "desfocar":
            out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (0, 0), max(w, h) / 8)
        else:
            small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)))
            out[y:y + h, x:x + w] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


def face_descriptor(img, face, size=96):
    """Descritor simples (HOG do rosto alinhado pelos olhos) para comparação aproximada."""
    rgb = _rgb(img)
    (rx, ry), (lx, ly) = face["pontos"][0], face["pontos"][1]
    ang = np.degrees(np.arctan2(ly - ry, lx - rx))
    cx, cy = (rx + lx) / 2, (ry + ly) / 2
    d = max(np.hypot(lx - rx, ly - ry), 1)
    M = cv2.getRotationMatrix2D((cx, cy), ang, size * 0.4 / d)
    M[0, 2] += size / 2 - cx; M[1, 2] += size * 0.38 - cy
    face_img = cv2.warpAffine(to_gray(rgb), M, (size, size))
    from skimage.feature import hog
    v = hog(cv2.equalizeHist(face_img), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return v / (np.linalg.norm(v) + 1e-9)


def face_similarity(img_a, img_b):
    fa, fb = detect_faces(img_a), detect_faces(img_b)
    if not fa or not fb:
        raise ValueError("Não foi encontrado um rosto em cada imagem.")
    return float(face_descriptor(img_a, fa[0]) @ face_descriptor(img_b, fb[0]))


# --------------------------------------------------------------------------
# Marcadores e códigos
# --------------------------------------------------------------------------
ARUCO_DICTS = {"4x4 (50)": "DICT_4X4_50", "5x5 (100)": "DICT_5X5_100", "6x6 (250)": "DICT_6X6_250", "AprilTag 36h11": "DICT_APRILTAG_36h11"}


def aruco_generate(marker_id=0, size_px=300, dict_name="4x4 (50)", border=40):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTS[dict_name]))
    m = cv2.aruco.generateImageMarker(d, int(marker_id), int(size_px))
    return cv2.copyMakeBorder(m, border, border, border, border, cv2.BORDER_CONSTANT, value=255)


def aruco_detect(img, dict_name="4x4 (50)"):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTS[dict_name]))
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(to_u8(to_gray(np.asarray(img))))
    out = _rgb(img)
    res = []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for c, i in zip(corners, ids.ravel()):
            pts = c.reshape(4, 2)
            res.append({"id": int(i), "centro": pts.mean(0).tolist(), "cantos": pts.tolist(),
                        "lado_px": float(np.mean(np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)))})
    return res, out


def qr_generate(text, scale=8):
    enc = cv2.QRCodeEncoder.create()
    q = enc.encode(text)
    return cv2.resize(q, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def qr_detect(img):
    det = cv2.QRCodeDetector()
    a = _rgb(img)
    ok, texts, pts, _ = det.detectAndDecodeMulti(a)
    out = a.copy(); res = []
    if ok and pts is not None:
        for t, p in zip(texts, pts):
            cv2.polylines(out, [p.astype(np.int32)], True, (255, 0, 255), 2)
            res.append({"texto": t, "cantos": p.tolist()})
    return res, out


# --------------------------------------------------------------------------
# Alvos de referência (quadrados/círculos pretos, grelhas)
# --------------------------------------------------------------------------
def black_squares(img, min_area=100, max_area=None, dark=True):
    g = to_u8(to_gray(np.asarray(img)))
    _, b = cv2.threshold(g, 0, 255, (cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY) | cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < min_area or (max_area and a > max_area):
            continue
        ap = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            (cx, cy), (w, h), ang = cv2.minAreaRect(ap)
            if min(w, h) / max(w, h) > 0.8:
                res.append({"centro": (cx, cy), "lado": (w + h) / 2, "ângulo": ang, "cantos": ap.reshape(4, 2).tolist()})
    return res


def black_circles(img, min_area=30, dark=True):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True; p.minArea = min_area; p.maxArea = 1e7
    p.filterByCircularity = True; p.minCircularity = 0.75
    p.filterByColor = True; p.blobColor = 0 if dark else 255
    det = cv2.SimpleBlobDetector_create(p)
    kps = det.detect(to_u8(to_gray(np.asarray(img))))
    return [{"centro": k.pt, "diâmetro": k.size} for k in kps]


def circle_grid(img, cols=7, rows=6, symmetric=True):
    g = to_u8(to_gray(np.asarray(img)))
    flag = cv2.CALIB_CB_SYMMETRIC_GRID if symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID
    ok, centers = cv2.findCirclesGrid(g, (cols, rows), flags=flag)
    out = _rgb(img)
    if ok:
        cv2.drawChessboardCorners(out, (cols, rows), centers, ok)
    return ok, (centers.reshape(-1, 2) if ok else None), out


def circle_grid_pattern(cols=7, rows=6, spacing=40, radius=12, margin=40):
    w = margin * 2 + (cols - 1) * spacing; h = margin * 2 + (rows - 1) * spacing
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            cv2.circle(img, (margin + c * spacing, margin + r * spacing), radius, 0, -1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# Foco total, movimento, fundo
# --------------------------------------------------------------------------
def all_in_focus(stack, window=9, smooth=2.0):
    """Combina uma pilha de fotografias com focos diferentes numa imagem toda focada."""
    imgs = [_rgb(i) for i in stack]
    F = np.stack([ndimage.uniform_filter(np.abs(cv2.Laplacian(cv2.GaussianBlur(to_gray(i), (3, 3), 0), cv2.CV_32F)), window)
                  for i in imgs])
    if smooth:
        F = np.stack([ndimage.gaussian_filter(f, smooth) for f in F])
    k = np.argmax(F, axis=0)
    S = np.stack(imgs)
    out = np.take_along_axis(S, k[None, ..., None].repeat(3, -1), axis=0)[0]
    return out, k.astype(float)


def motion(a, b, threshold=25, min_area=50):
    d = cv2.absdiff(to_u8(to_gray(np.asarray(a))), to_u8(to_gray(np.asarray(b))))
    _, m = cv2.threshold(cv2.GaussianBlur(d, (5, 5), 0), threshold, 255, cv2.THRESH_BINARY)
    m = cv2.dilate(m, None, iterations=2)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cs if cv2.contourArea(c) >= min_area]
    out = _rgb(b)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 60, 60), 2)
    return m, boxes, out


def background_subtraction(frames, history=50, var=16):
    bs = cv2.createBackgroundSubtractorMOG2(history, var, False)
    masks = [bs.apply(_rgb(f)) for f in frames]
    return masks, bs.getBackgroundImage()


def time_average(frames):
    return np.mean([np.asarray(f, float) for f in frames], axis=0)


# --------------------------------------------------------------------------
# Cor e efeitos
# --------------------------------------------------------------------------
def color_segment(img, h_min=0, h_max=20, s_min=80, v_min=60):
    hsv = cv2.cvtColor(_rgb(img), cv2.COLOR_RGB2HSV)
    if h_min <= h_max:
        m = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))
    else:  # intervalo que dá a volta (vermelhos)
        m = cv2.inRange(hsv, (h_min, s_min, v_min), (179, 255, 255)) | cv2.inRange(hsv, (0, s_min, v_min), (h_max, 255, 255))
    return m


def effect(img, kind):
    a = _rgb(img)
    if kind == "pintura":
        return cv2.stylization(a, sigma_s=60, sigma_r=0.45)
    if kind == "desenho a lápis":
        g, _ = cv2.pencilSketch(a, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return g
    if kind == "banda desenhada":
        col = a
        for _ in range(3):
            col = cv2.bilateralFilter(col, 9, 75, 75)
        e = cv2.adaptiveThreshold(cv2.medianBlur(to_gray(a), 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        return cv2.bitwise_and(col, col, mask=e)
    if kind == "realçar detalhe":
        return cv2.detailEnhance(a, sigma_s=10, sigma_r=0.15)
    if kind == "sépia":
        k = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        return np.clip(a.astype(float) @ k.T, 0, 255).astype(np.uint8)
    raise ValueError(kind)
