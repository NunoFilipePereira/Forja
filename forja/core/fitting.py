"""Ajuste de formas (planos, polinómios, círculos, elipses, retas, esferas) e curvatura."""
import numpy as np
from scipy import ndimage, optimize


# --------------------------------------------------------------------------
# Superfícies em mapas de altura (imagens)
# --------------------------------------------------------------------------
def _grid(Z):
    r, c = Z.shape
    Y, X = np.mgrid[:r, :c]
    return X.astype(float), Y.astype(float)


def _terms(X, Y, order):
    return [X ** i * Y ** (k - i) for k in range(order + 1) for i in range(k + 1)]


def fit_surface(Z, order=1, weights=None, mask=None):
    """Ajusta z = Σ a_ij x^i y^j (ordem 0–5). order=1 é o plano. Devolve (superfície, coeficientes, rms)."""
    Z = np.asarray(Z, float)
    X, Y = _grid(Z)
    # normaliza coordenadas para estabilidade numérica
    sx, sy = max(Z.shape[1] - 1, 1), max(Z.shape[0] - 1, 1)
    Xn, Yn = X / sx, Y / sy
    m = np.isfinite(Z) if mask is None else (np.asarray(mask, bool) & np.isfinite(Z))
    T = _terms(Xn, Yn, order)
    A = np.column_stack([t[m] for t in T])
    b = Z[m]
    if weights is not None:
        w = np.sqrt(np.clip(np.asarray(weights, float)[m], 0, None))
        A, b = A * w[:, None], b * w
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    S = sum(c * t for c, t in zip(coef, T))
    res = Z - S
    rms = float(np.sqrt(np.nanmean(res[m] ** 2)))
    return S, coef, rms


def plane_equation(Z, weights=None, mask=None):
    """Plano z = a·x + b·y + c em coordenadas de pixel → Ax + By + Cz = D normalizado."""
    S, coef, rms = fit_surface(Z, 1, weights, mask)
    sx, sy = max(Z.shape[1] - 1, 1), max(Z.shape[0] - 1, 1)
    c0, cy, cx = coef  # termos: 1, y, x  (ordem de _terms para k=1: x^0 y^1, x^1 y^0)
    a, b, c = cx / sx, cy / sy, c0
    n = np.array([a, b, -1.0]); k = np.linalg.norm(n)
    return (a / k, b / k, -1 / k, -c / k), S, rms


def fit_plane_points(P, weights=None):
    """Plano por mínimos quadrados totais a pontos 3D. Devolve (normal, ponto, rms)."""
    P = np.asarray(P, float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    c = (P * w[:, None]).sum(0) / w.sum()
    U, S, Vt = np.linalg.svd((P - c) * np.sqrt(w)[:, None], full_matrices=False)
    n = Vt[-1]
    d = (P - c) @ n
    return n, c, float(np.sqrt(np.mean(d ** 2)))


def fit_sphere(P):
    P = np.asarray(P, float)
    A = np.column_stack([2 * P, np.ones(len(P))])
    b = (P ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]; r = np.sqrt(x[3] + c @ c)
    rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2)))
    return c, float(r), rms


def fit_cylinder_z(P):
    """Cilindro com eixo aproximadamente vertical: círculo no plano XY."""
    c, r, rms = fit_circle(np.asarray(P)[:, :2])
    return c, r, rms


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------
def fit_circle(P):
    """Ajuste geométrico (Kåsa + refinamento de Levenberg–Marquardt)."""
    P = np.asarray(P, float)
    A = np.column_stack([2 * P, np.ones(len(P))])
    b = (P ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c0 = x[:2]; r0 = np.sqrt(x[2] + c0 @ c0)

    def resid(p):
        return np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) - p[2]
    sol = optimize.least_squares(resid, [c0[0], c0[1], r0], method="lm")
    cx, cy, r = sol.x
    return (float(cx), float(cy)), float(abs(r)), float(np.sqrt(np.mean(resid(sol.x) ** 2)))


def fit_line(P):
    P = np.asarray(P, float)
    c = P.mean(0)
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    dist = (P - c) @ Vt[1]
    t = (P - c) @ d
    return c, d, (c + d * t.min(), c + d * t.max()), float(np.sqrt(np.mean(dist ** 2)))


def fit_ellipse(P):
    import cv2
    P = np.asarray(P, np.float32)
    if len(P) < 5:
        raise ValueError("São precisos pelo menos 5 pontos.")
    (cx, cy), (a, b), ang = cv2.fitEllipse(P)
    return (cx, cy), (a / 2, b / 2), ang


def min_bounding(P, kind):
    import cv2
    P = np.asarray(P, np.float32)
    if kind == "caixa mínima":
        (cx, cy), (w, h), a = cv2.minAreaRect(P)
        return {"cx": cx, "cy": cy, "w": w, "h": h, "angle": a}
    if kind == "círculo mínimo":
        (x, y), r = cv2.minEnclosingCircle(P)
        return {"cx": x, "cy": y, "r": r}
    if kind == "invólucro convexo":
        return {"pts": cv2.convexHull(P).reshape(-1, 2)}
    if kind == "triângulo mínimo":
        _, tri = cv2.minEnclosingTriangle(P.reshape(-1, 1, 2))
        return {"pts": tri.reshape(-1, 2)}
    raise ValueError(kind)


def geom_points(g, step=0.5):
    """Amostra pontos ao longo dos contornos de uma geometria shapely."""
    from .shapes2d import iter_paths
    pts = []
    for p in iter_paths(g):
        for a, b in zip(p[:-1], p[1:]):
            n = max(1, int(np.hypot(*(b[:2] - a[:2])) / step))
            for t in np.linspace(0, 1, n, endpoint=False):
                pts.append(a[:2] + (b[:2] - a[:2]) * t)
        if len(p):
            pts.append(p[-1][:2])
    if g.geom_type == "Point":
        pts.append((g.x, g.y))
    if hasattr(g, "geoms"):
        for sub in g.geoms:
            if sub.geom_type == "Point":
                pts.append((sub.x, sub.y))
    return np.array(pts, float)


# --------------------------------------------------------------------------
# Curvatura
# --------------------------------------------------------------------------
def curvature_image(Z, px=1.0, smooth=1.0):
    """Curvaturas de uma superfície z(x,y): Gauss, média, mínima, máxima, em X e em Y."""
    Z = np.asarray(Z, float)
    Zs = ndimage.gaussian_filter(np.nan_to_num(Z, nan=np.nanmean(Z)), smooth) if smooth > 0 else Z
    zy, zx = np.gradient(Zs, px)
    zxy, zxx = np.gradient(zx, px)
    zyy, _ = np.gradient(zy, px)
    g = 1 + zx ** 2 + zy ** 2
    K = (zxx * zyy - zxy ** 2) / g ** 2
    H = ((1 + zx ** 2) * zyy - 2 * zx * zy * zxy + (1 + zy ** 2) * zxx) / (2 * g ** 1.5)
    d = np.sqrt(np.maximum(H ** 2 - K, 0))
    out = {"Gauss": K, "média": H, "mínima": H - d, "máxima": H + d,
           "em X": zxx / (1 + zx ** 2) ** 1.5, "em Y": zyy / (1 + zy ** 2) ** 1.5}
    nan = ~np.isfinite(Z)
    for k in out:
        out[k][nan] = np.nan
    return out


def mesh_curvature(mesh, kind="média", radius=None):
    import trimesh
    r = radius or mesh.scale / 50
    pts = mesh.vertices
    if kind == "Gauss":
        v = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, pts, r)
    else:
        v = trimesh.curvature.discrete_mean_curvature_measure(mesh, pts, r)
    return v / (np.pi * r ** 2)


def values_to_colors(v, cmap="coolwarm", clip_pct=2):
    import matplotlib
    lo, hi = np.nanpercentile(v, [clip_pct, 100 - clip_pct])
    if lo < 0 < hi:
        m = max(-lo, hi); lo, hi = -m, m
    n = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
    return matplotlib.colormaps[cmap](n).astype(np.float32), (lo, hi)


# --------------------------------------------------------------------------
# Isolinhas
# --------------------------------------------------------------------------
def iso_contours(Z, levels, px=1.0):
    from skimage import measure
    Z = np.asarray(Z, float)
    rows = Z.shape[0]
    fill = np.nanmin(Z) - 1 if np.isfinite(Z).any() else 0
    Zf = np.nan_to_num(Z, nan=fill)
    out = []
    for lv in levels:
        for c in measure.find_contours(Zf, lv):
            if len(c) >= 2:
                out.append((lv, np.column_stack([c[:, 1] * px, (rows - 1 - c[:, 0]) * px])))
    return out
