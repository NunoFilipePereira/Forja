"""Metrologia (3): calibração com dois planos de referência, fusão de profundidade, DIC em sequência,
calibração multivista, CT de feixe cónico (FDK), nuvem regularizada, distorção radial."""
import numpy as np
import cv2
from scipy import ndimage


def two_plane_height(phi_obj, phi_ref1, phi_ref2, h1=0.0, h2=10.0):
    """Altura absoluta por interpolação linear da fase, pixel a pixel, entre dois planos de referência
    colocados a alturas conhecidas h1 e h2 (compensa a geometria divergente da projeção)."""
    P, A, B = (np.asarray(x, float) for x in (phi_obj, phi_ref1, phi_ref2))
    d = B - A
    d = np.where(np.abs(d) < 1e-9, np.nan, d)
    return h1 + (P - A) / d * (h2 - h1)


def fuse_depth(z1, z2, q1=None, q2=None, mode="média ponderada"):
    Z1, Z2 = np.asarray(z1, float), np.asarray(z2, float)
    Q1 = np.ones_like(Z1) if q1 is None else np.clip(np.asarray(q1, float), 0, None)
    Q2 = np.ones_like(Z2) if q2 is None else np.clip(np.asarray(q2, float), 0, None)
    Q1 = np.where(np.isfinite(Z1), Q1, 0); Q2 = np.where(np.isfinite(Z2), Q2, 0)
    a, b = np.nan_to_num(Z1), np.nan_to_num(Z2)
    if mode == "melhor qualidade":
        out = np.where(Q1 >= Q2, a, b)
    else:
        out = (a * Q1 + b * Q2) / np.maximum(Q1 + Q2, 1e-12)
    out[(Q1 + Q2) == 0] = np.nan
    return out


def dic_sequence(frames, mode="com a primeira", subset=31, step=16, search=20):
    from .metro2 import dic
    res = []
    for k in range(1, len(frames)):
        ref = frames[0] if mode == "com a primeira" else frames[k - 1]
        DX, DY, C, xs, ys = dic(ref, frames[k], subset, step, search)
        res.append({"frame": k, "dx": float(np.nanmean(DX)), "dy": float(np.nanmean(DY)), "corr": float(np.nanmean(C)),
                    "DX": DX, "DY": DY})
    return res


def multiview_calibration(images, cols=9, rows=6, square=10.0, pattern="tabuleiro"):
    """Calibra a câmara e devolve a pose (rotação/translação) do alvo em cada vista."""
    if pattern == "tabuleiro":
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    else:
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    O, P, used = [], [], []
    size = None
    from .imageops import to_gray, to_u8
    for k, im in enumerate(images):
        g = to_u8(to_gray(np.asarray(im)))
        size = g.shape[::-1]
        if pattern == "tabuleiro":
            ok, c = cv2.findChessboardCorners(g, (cols, rows))
            if ok:
                c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        else:
            ok, c = cv2.findCirclesGrid(g, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        if ok:
            O.append(objp); P.append(c); used.append(k)
    if len(O) < 3:
        raise ValueError(f"Alvo encontrado em {len(O)} vistas; são precisas pelo menos 3.")
    err, K, D, rv, tv = cv2.calibrateCamera(O, P, size, None, None)
    poses = []
    for k, r, t in zip(used, rv, tv):
        R, _ = cv2.Rodrigues(r)
        cam_pos = (-R.T @ t).ravel()
        poses.append({"vista": k + 1, "rotação (°)": np.degrees(r.ravel()).tolist(), "translação (mm)": t.ravel().tolist(),
                      "posição da câmara (mm)": cam_pos.tolist(), "distância (mm)": float(np.linalg.norm(t))})
    return {"erro_rms_px": err, "K": K, "distorção": D.ravel(), "poses": poses}


def radial_undistort(img, k1=0.0, k2=0.0, cx=None, cy=None, f=None):
    a = np.asarray(img)
    h, w = a.shape[:2]
    f = f or max(w, h)
    K = np.array([[f, 0, cx if cx is not None else w / 2], [0, f, cy if cy is not None else h / 2], [0, 0, 1]], float)
    return cv2.undistort(a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32), K, np.array([k1, k2, 0, 0], float))


def regularize_points(P, spacing=1.0, method="linear"):
    """Reamostra uma nuvem 2,5D numa grelha XY regular (mapa de altura)."""
    from scipy.interpolate import griddata
    P = np.asarray(P, float)
    x0, y0 = P[:, :2].min(0); x1, y1 = P[:, :2].max(0)
    xs = np.arange(x0, x1 + spacing / 2, spacing); ys = np.arange(y1, y0 - spacing / 2, -spacing)
    X, Y = np.meshgrid(xs, ys)
    Z = griddata(P[:, :2], P[:, 2], (X, Y), method=method)
    return Z, (x0, y1), spacing


# --------------------------------------------------------------------------
# CT de feixe cónico (geometria circular, detetor plano) — FDK
# --------------------------------------------------------------------------
def cone_forward(vol, n_angles=90, sdd=400.0, sod=250.0, det_px=1.0, det_shape=None):
    """Projeções cónicas sintéticas de um volume (z, y, x) com voxel de 1 mm, a rodar em Z."""
    V = np.asarray(vol, float)
    nz, ny, nx = V.shape
    dv, du = det_shape or (int(nz * sdd / sod * 1.2), int(nx * sdd / sod * 1.2))
    us = (np.arange(du) - du / 2 + 0.5) * det_px
    vs = (np.arange(dv) - dv / 2 + 0.5) * det_px
    U, Vv = np.meshgrid(us, vs)
    ts = np.linspace(-1, 1, max(nx, ny) * 2) * max(nx, ny) * 0.75
    out = []
    for a in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        src = np.array([sod * np.cos(a), sod * np.sin(a), 0.0])
        dcen = np.array([-(sdd - sod) * np.cos(a), -(sdd - sod) * np.sin(a), 0.0])
        eu = np.array([-np.sin(a), np.cos(a), 0.0]); ev = np.array([0, 0, 1.0])
        P = dcen[None, None] + U[..., None] * eu + Vv[..., None] * ev
        d = P - src; d /= np.linalg.norm(d, axis=-1, keepdims=True)
        acc = np.zeros(U.shape)
        for t in ts + sod:
            q = src + d * t
            ix = q[..., 0] + nx / 2 - 0.5; iy = q[..., 1] + ny / 2 - 0.5; iz = q[..., 2] + nz / 2 - 0.5
            acc += ndimage.map_coordinates(V, [iz, iy, ix], order=1, mode="constant", cval=0.0)
        out.append(acc * (ts[1] - ts[0]))
    return out


def fdk(projections, sdd=400.0, sod=250.0, det_px=1.0, n=None, nz=None):
    """Reconstrução FDK (Feldkamp–Davis–Kress) para órbita circular completa (360°)."""
    P = np.stack([np.asarray(p, float) for p in projections])
    na, dv, du = P.shape
    n = n or int(du * sod / sdd)
    nz = nz or int(dv * sod / sdd)
    us = (np.arange(du) - du / 2 + 0.5) * det_px
    vs = (np.arange(dv) - dv / 2 + 0.5) * det_px
    U, Vv = np.meshgrid(us, vs)
    W = sdd / np.sqrt(sdd ** 2 + U ** 2 + Vv ** 2)                 # ponderação cosseno
    # filtro rampa (Ram-Lak) linha a linha
    nf = int(2 ** np.ceil(np.log2(2 * du)))
    freqs = np.fft.fftfreq(nf, d=det_px)
    ramp = np.abs(freqs)
    Pf = np.real(np.fft.ifft(np.fft.fft(P * W[None], nf, axis=2) * ramp[None, None], axis=2))[..., :du]
    z, y, x = np.meshgrid(np.arange(nz) - nz / 2 + 0.5, np.arange(n) - n / 2 + 0.5, np.arange(n) - n / 2 + 0.5, indexing="ij")
    vol = np.zeros(z.shape)
    angles = np.linspace(0, 2 * np.pi, na, endpoint=False)
    for k, a in enumerate(angles):
        s = x * np.cos(a) + y * np.sin(a)         # projeção na direção da fonte
        t = -x * np.sin(a) + y * np.cos(a)        # direção do detetor (u)
        L = sod - s
        mag = sdd / L
        u = t * mag; v = z * mag
        iu = u / det_px + du / 2 - 0.5; iv = v / det_px + dv / 2 - 0.5
        vol += ndimage.map_coordinates(Pf[k], [iv, iu], order=1, mode="constant", cval=0.0) * (sod / L) ** 2
    return vol * (np.pi / na) * (sdd / sod) / det_px
