"""Metrologia avançada: DIC, shearografia, holografia digital, desembrulhamento por qualidade (PGV),
filtragem de fase, estéreo, linha laser, nuvens de pontos (ICP, Delaunay)."""
import heapq
import numpy as np
import cv2
from scipy import ndimage
from .imageops import to_gray, to_u8


def _g(a):
    a = np.asarray(a)
    return (to_gray(a) if a.ndim == 3 else a).astype(np.float32)


# --------------------------------------------------------------------------
# Correlação digital de imagem (DIC)
# --------------------------------------------------------------------------
def dic(ref, deformed, subset=31, step=10, search=20, min_corr=0.6):
    """Deslocamentos por correlação de subconjuntos (NCC) com refinamento sub-pixel parabólico.
    Devolve mapas (dx, dy, corr) na grelha de pontos e as coordenadas dessa grelha."""
    A, B = _g(ref), _g(deformed)
    h, w = A.shape
    half = subset // 2
    ys = np.arange(half + search, h - half - search, step)
    xs = np.arange(half + search, w - half - search, step)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Imagem demasiado pequena para o subconjunto/área de procura escolhidos.")
    DX = np.full((len(ys), len(xs)), np.nan, np.float32); DY = DX.copy(); C = DX.copy()
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            tpl = A[y - half:y + half + 1, x - half:x + half + 1]
            if tpl.std() < 1e-3:
                continue
            win = B[y - half - search:y + half + search + 1, x - half - search:x + half + search + 1]
            R = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, (px, py) = cv2.minMaxLoc(R)
            if mx < min_corr:
                continue
            sx = sy = 0.0
            if 0 < px < R.shape[1] - 1:
                l, c, r = R[py, px - 1], R[py, px], R[py, px + 1]
                d = l - 2 * c + r
                sx = 0.5 * (l - r) / d if d != 0 else 0
            if 0 < py < R.shape[0] - 1:
                u, c, dn = R[py - 1, px], R[py, px], R[py + 1, px]
                d = u - 2 * c + dn
                sy = 0.5 * (u - dn) / d if d != 0 else 0
            DX[i, j] = px + sx - search
            DY[i, j] = py + sy - search
            C[i, j] = mx
    return DX, DY, C, xs, ys


def dic_dense(ref, deformed):
    """Fluxo ótico denso (Farnebäck) — deslocamento por pixel, rápido."""
    A, B = to_u8(_g(ref)), to_u8(_g(deformed))
    f = cv2.calcOpticalFlowFarneback(A, B, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    return f[..., 0], f[..., 1]


def strains(DX, DY, step=1.0, smooth=1.0):
    """Deformações (pequenas deformações) a partir dos campos de deslocamento: exx, eyy, exy (em mili-deformação)."""
    u = ndimage.gaussian_filter(np.nan_to_num(DX), smooth) if smooth else np.nan_to_num(DX)
    v = ndimage.gaussian_filter(np.nan_to_num(DY), smooth) if smooth else np.nan_to_num(DY)
    du_dy, du_dx = np.gradient(u, step)
    dv_dy, dv_dx = np.gradient(v, step)
    exx, eyy, exy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    nan = ~np.isfinite(DX)
    out = {"exx": exx * 1000, "eyy": eyy * 1000, "exy": exy * 1000,
           "von Mises": np.sqrt(exx ** 2 + eyy ** 2 - exx * eyy + 3 * exy ** 2) * 1000}
    for k in out:
        out[k][nan] = np.nan
    return out


def vector_plot(img, DX, DY, xs, ys, scale=3.0):
    base = to_u8(np.asarray(img))
    out = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) if base.ndim == 2 else base.copy()
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            dx, dy = DX[i, j], DY[i, j]
            if np.isfinite(dx):
                cv2.arrowedLine(out, (int(x), int(y)), (int(x + dx * scale), int(y + dy * scale)), (255, 60, 60), 1, tipLength=0.3)
    return out


def speckle_pattern(w=400, h=300, density=0.5, size=2.5, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 230, np.float32)
    n = int(w * h * density / (np.pi * size ** 2))
    for x, y, r in zip(rng.uniform(0, w, n), rng.uniform(0, h, n), rng.uniform(size * 0.6, size * 1.4, n)):
        cv2.circle(img, (int(x), int(y)), int(round(r)), 30, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 0.8).astype(np.uint8)


def warp_affine_field(img, exx=0.01, eyy=0.0, exy=0.0, tx=0.0, ty=0.0):
    h, w = img.shape[:2]
    M = np.array([[1 + exx, exy, tx - exx * w / 2 - exy * h / 2], [exy, 1 + eyy, ty - exy * w / 2 - eyy * h / 2]], np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------
# Fase: subtração, filtragem, média, qualidade e desembrulhamento guiado
# --------------------------------------------------------------------------
def wrap(p):
    return np.angle(np.exp(1j * p))


def subtract_phase(a, b):
    return wrap(np.asarray(a, float) - np.asarray(b, float))


def filter_wrapped(phi, size=5, iterations=1):
    """Filtragem seno/cosseno (preserva os saltos de 2π)."""
    s, c = np.sin(phi), np.cos(phi)
    for _ in range(iterations):
        s = ndimage.uniform_filter(s, size); c = ndimage.uniform_filter(c, size)
    return np.arctan2(s, c)


def average_phase(phases):
    Z = np.mean([np.exp(1j * np.asarray(p, float)) for p in phases], axis=0)
    return np.angle(Z)


def phase_gradient_variance(phi, size=3):
    """Mapa de variância do gradiente de fase (PGV): baixo = boa qualidade."""
    dx = wrap(np.diff(phi, axis=1, append=phi[:, -1:]))
    dy = wrap(np.diff(phi, axis=0, append=phi[-1:, :]))
    vx = ndimage.uniform_filter(dx ** 2, size) - ndimage.uniform_filter(dx, size) ** 2
    vy = ndimage.uniform_filter(dy ** 2, size) - ndimage.uniform_filter(dy, size) ** 2
    return np.sqrt(np.maximum(vx, 0)) + np.sqrt(np.maximum(vy, 0))


def second_difference_quality(phi):
    """Fiabilidade por segundas diferenças da fase embrulhada (alta = fiável)."""
    p = np.pad(phi, 1, mode="edge")
    c = p[1:-1, 1:-1]
    H = wrap(p[1:-1, :-2] - c) - wrap(c - p[1:-1, 2:])
    V = wrap(p[:-2, 1:-1] - c) - wrap(c - p[2:, 1:-1])
    D1 = wrap(p[:-2, :-2] - c) - wrap(c - p[2:, 2:])
    D2 = wrap(p[:-2, 2:] - c) - wrap(c - p[2:, :-2])
    return 1.0 / (np.sqrt(H ** 2 + V ** 2 + D1 ** 2 + D2 ** 2) + 1e-6)


def unwrap_guided(phi, quality, mask=None):
    """Desembrulhamento guiado por um mapa de qualidade (maior = melhor), por crescimento de região."""
    phi = np.asarray(phi, float)
    q = np.asarray(quality, float)
    h, w = phi.shape
    valid = np.isfinite(phi) if mask is None else (np.asarray(mask, bool) & np.isfinite(phi))
    out = np.full(phi.shape, np.nan)
    done = np.zeros(phi.shape, bool)
    qq = np.where(valid, q, -np.inf)
    start = np.unravel_index(np.argmax(qq), qq.shape)
    out[start] = phi[start]; done[start] = True
    heap = []
    def push(y, x):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and valid[ny, nx] and not done[ny, nx]:
                # fiabilidade da ligação: soma das qualidades dos dois pixels
                heapq.heappush(heap, (-(q[ny, nx] + q[y, x]), ny, nx, y, x))
    push(*start)
    while heap:
        _, y, x, py, px = heapq.heappop(heap)
        if done[y, x]:
            continue
        out[y, x] = out[py, px] + wrap(phi[y, x] - phi[py, px])
        done[y, x] = True
        push(y, x)
    return out


# --------------------------------------------------------------------------
# Shearografia / interferometria: fase → deformação
# --------------------------------------------------------------------------
def phase_to_out_of_plane(dphi, wavelength_nm=532.0, angle_deg=0.0):
    """w = λ·Δφ / (4π·cos²(θ/2)) — deslocamento fora do plano (nm)."""
    return wavelength_nm * dphi / (4 * np.pi * np.cos(np.radians(angle_deg) / 2) ** 2)


def phase_to_in_plane(dphi, wavelength_nm=532.0, angle_deg=30.0):
    """u = λ·Δφ / (4π·sin θ) — deslocamento no plano para iluminação dupla simétrica (nm)."""
    return wavelength_nm * dphi / (4 * np.pi * np.sin(np.radians(angle_deg)))


def phase_to_gradient(dphi, wavelength_nm=532.0, shear_mm=5.0):
    """Shearografia: ∂w/∂x ≈ λ·Δφ / (4π·δx) (adimensional, apresentado em µm/m)."""
    return wavelength_nm * 1e-9 * dphi / (4 * np.pi * shear_mm * 1e-3) * 1e6


# --------------------------------------------------------------------------
# Holografia digital
# --------------------------------------------------------------------------
def holo_reconstruct(holo, wavelength_nm=632.8, pixel_um=3.45, distance_mm=100.0, method="espectro angular"):
    H = np.asarray(_g(holo), float)
    H = H - H.mean()
    lam = wavelength_nm * 1e-9; dx = pixel_um * 1e-6; z = distance_mm * 1e-3
    ny, nx = H.shape
    if method == "Fresnel (campo próximo)":
        x = (np.arange(nx) - nx / 2) * dx; y = (np.arange(ny) - ny / 2) * dx
        X, Y = np.meshgrid(x, y)
        chirp = np.exp(1j * np.pi / (lam * z) * (X ** 2 + Y ** 2))
        U = np.fft.fftshift(np.fft.fft2(H * chirp))
    elif method == "Fraunhofer (campo distante)":
        U = np.fft.fftshift(np.fft.fft2(H))
    else:
        fx = np.fft.fftfreq(nx, dx); fy = np.fft.fftfreq(ny, dx)
        FX, FY = np.meshgrid(fx, fy)
        arg = 1 - (lam * FX) ** 2 - (lam * FY) ** 2
        Hf = np.where(arg > 0, np.exp(1j * 2 * np.pi / lam * z * np.sqrt(np.maximum(arg, 0))), 0)
        U = np.fft.ifft2(np.fft.fft2(H) * Hf)
    return np.abs(U), np.angle(U)


def simulate_hologram(obj, wavelength_nm=632.8, pixel_um=3.45, distance_mm=100.0, ref_angle_deg=1.0):
    """Holograma off-axis sintético de um objeto de amplitude (para testes)."""
    lam = wavelength_nm * 1e-9; dx = pixel_um * 1e-6; z = distance_mm * 1e-3
    O = np.asarray(obj, float)
    ny, nx = O.shape
    fx = np.fft.fftfreq(nx, dx); fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1 - (lam * FX) ** 2 - (lam * FY) ** 2
    Hf = np.where(arg > 0, np.exp(1j * 2 * np.pi / lam * z * np.sqrt(np.maximum(arg, 0))), 0)
    U = np.fft.ifft2(np.fft.fft2(O) * Hf)
    x = np.arange(nx) * dx
    R = np.exp(1j * 2 * np.pi / lam * np.sin(np.radians(ref_angle_deg)) * x)[None, :] * np.abs(U).max()
    I = np.abs(U + R) ** 2
    return (I / I.max() * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Linha laser (triangulação)
# --------------------------------------------------------------------------
def laser_line_profile(img, threshold=0.3, axis=0):
    """Centro sub-pixel da linha laser por coluna (centróide de intensidade acima do limiar)."""
    a = _g(img).astype(float)
    if a.ndim == 3:
        a = a.mean(2)
    if axis == 1:
        a = a.T
    a = a - np.median(a, axis=0, keepdims=True)
    mx = a.max(axis=0)
    m = a >= threshold * mx[None, :]
    rows = np.arange(a.shape[0])[:, None]
    w = np.where(m, a, 0)
    s = w.sum(0)
    c = np.where(s > 0, (w * rows).sum(0) / np.maximum(s, 1e-12), np.nan)
    c[mx < 1e-6] = np.nan
    return c


def line_scan(images, ref_row=None, scale_mm_per_px=0.1, threshold=0.3):
    """Cada imagem (um perfil) é uma linha do mapa de altura."""
    P = np.array([laser_line_profile(i, threshold) for i in images])
    ref = np.nanmedian(P) if ref_row is None else ref_row
    return (ref - P) * scale_mm_per_px


def simulate_laser_images(Z, k_px_per_mm=2.0, base_row=60, h=160, width=3.0):
    out = []
    for z in Z:
        img = np.zeros((h, len(z)), np.float32)
        rows = np.arange(h)[:, None]
        cen = base_row - z * k_px_per_mm
        img += 220 * np.exp(-((rows - cen[None, :]) / width) ** 2)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


# --------------------------------------------------------------------------
# Estéreo (duas câmaras)
# --------------------------------------------------------------------------
def stereo_calibrate(left_imgs, right_imgs, cols=9, rows=6, square=10.0):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    O, L, R = [], [], []
    size = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for a, b in zip(left_imgs, right_imgs):
        ga, gb = to_u8(_g(a)), to_u8(_g(b))
        size = ga.shape[::-1]
        oka, ca = cv2.findChessboardCorners(ga, (cols, rows))
        okb, cb = cv2.findChessboardCorners(gb, (cols, rows))
        if oka and okb:
            O.append(objp)
            L.append(cv2.cornerSubPix(ga, ca, (11, 11), (-1, -1), crit))
            R.append(cv2.cornerSubPix(gb, cb, (11, 11), (-1, -1), crit))
    if len(O) < 3:
        raise ValueError(f"Tabuleiro encontrado em {len(O)} pares; são precisos pelo menos 3.")
    e1, K1, D1, _, _ = cv2.calibrateCamera(O, L, size, None, None)
    e2, K2, D2, _, _ = cv2.calibrateCamera(O, R, size, None, None)
    err, K1, D1, K2, D2, Rm, T, E, F = cv2.stereoCalibrate(O, L, R, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC)
    return {"erro_rms_px": err, "K1": K1, "D1": D1, "K2": K2, "D2": D2, "R": Rm, "T": T, "size": size,
            "pares": len(O), "base_mm": float(np.linalg.norm(T))}


def stereo_depth(left, right, calib, num_disp=128, block=7):
    K1, D1, K2, D2, Rm, T, size = (calib[k] for k in ("K1", "D1", "K2", "D2", "R", "T", "size"))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, Rm, T)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    a = cv2.remap(to_u8(_g(left)), *m1, cv2.INTER_LINEAR); b = cv2.remap(to_u8(_g(right)), *m2, cv2.INTER_LINEAR)
    sgbm = cv2.StereoSGBM_create(0, num_disp, block, P1=8 * block * block, P2=32 * block * block, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2)
    disp = sgbm.compute(a, b).astype(np.float32) / 16
    pts = cv2.reprojectImageTo3D(disp, Q)
    Z = pts[..., 2].copy()
    Z[disp <= 0] = np.nan
    return disp, Z, pts


def disparity_uncalibrated(left, right, num_disp=64, block=7):
    a, b = to_u8(_g(left)), to_u8(_g(right))
    sgbm = cv2.StereoSGBM_create(0, num_disp, block, P1=8 * block * block, P2=32 * block * block)
    d = sgbm.compute(a, b).astype(np.float32) / 16
    d[d <= 0] = np.nan
    return d


def render_chessboard_view(rvec, tvec, K, cols=9, rows=6, square=10.0, size=(640, 480)):
    """Imagem sintética de um tabuleiro visto por uma câmara (para testes de calibração)."""
    sq = 40
    board = np.full(((rows + 3) * sq, (cols + 3) * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    # cantos internos do tabuleiro (em mm) ↔ pixels da imagem do tabuleiro
    k = square / sq
    corners_px = np.float32([[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]])
    obj = np.float32([[(x - 2 * sq) * k, (y - 2 * sq) * k, 0] for x, y in corners_px])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    M = cv2.getPerspectiveTransform(corners_px, proj.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(board, M, size, borderValue=128)


# --------------------------------------------------------------------------
# Nuvens de pontos
# --------------------------------------------------------------------------
def icp_align(source_pts, target_pts, iterations=50):
    import trimesh
    M, transformed, cost = trimesh.registration.icp(source_pts, target_pts, max_iterations=iterations)
    return M, cost


def delaunay_mesh(points, max_edge=None):
    """Triangulação 2.5D (em XY) de uma nuvem de pontos → malha."""
    import trimesh
    from scipy.spatial import Delaunay
    P = np.asarray(points, float)
    tri = Delaunay(P[:, :2])
    F = tri.simplices
    if max_edge:
        e = np.max([np.linalg.norm(P[F[:, i]] - P[F[:, (i + 1) % 3]], axis=1) for i in range(3)], axis=0)
        F = F[e <= max_edge]
    m = trimesh.Trimesh(P, F, process=False)
    if m.face_normals[:, 2].mean() < 0:
        m.invert()
    return m


def merge_turntable(clouds, angles_deg, axis_point=(0, 0, 0)):
    """Junta nuvens de pontos tiradas com o objeto a rodar (eixo vertical)."""
    out = []
    c = np.asarray(axis_point, float)
    for P, a in zip(clouds, np.radians(angles_deg)):
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        out.append((np.asarray(P) - c) @ R.T + c)
    return np.vstack(out)


def load_points(path):
    from .imageops2 import load_ascii
    if path.lower().endswith(".ply"):
        import trimesh
        return np.asarray(trimesh.load(path).vertices)
    a = load_ascii(path)
    if a.shape[1] < 3:
        raise ValueError("O ficheiro precisa de pelo menos 3 colunas (x y z).")
    return a[:, :3]


def save_points_txt(path, P, normals=None):
    P = np.asarray(P)
    data = P if normals is None else np.hstack([P, normals])
    np.savetxt(path, data, fmt="%.6f")
