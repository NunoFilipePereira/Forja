"""Litofanias com as 18 formas:
Flat, Arc, CMYK, Dual, Rectangle, Polygon, Frustum, Cylinder, Cone, Vase, Sphere, Spiral, Stencil, Balloon, Box,
Bronze, Tattoo, Tile. Cada forma usa os mesmos campos do painel (ver ui/create_tools.py)."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from scipy import ndimage
from shapely.geometry import Point, Polygon, box as sbox
from shapely import affinity
from . import cimg as C

SHAPES = ["Flat", "Arc", "CMYK", "Dual", "Rectangle", "Polygon", "Frustum", "Cylinder", "Cone", "Vase", "Sphere", "Spiral",
          "Stencil", "Balloon", "Box", "Bronze", "Tattoo", "Tile"]


# --------------------------------------------------------------------------
# auxiliares
# --------------------------------------------------------------------------
def _res(v, W, fast):
    r = max(float(v.get("res", 0.1)), 0.05)
    if fast:
        r = max(r, W / 110)
    return r


def _photo(v, rows, cols, upright=True):
    g = C.load_gray(v.get("photo", ""), 700)
    if not upright:
        g = g[::-1, ::-1]
    return C.resample(g, rows, cols)


def _T(v, g):
    return C.litho_T(g, v["tmin"], v["tmax"], v.get("polarity", "Positive") == "Positive")


def _surface(S, T, outward=True, center=None):
    """Sólido a partir da superfície base S (R × C × 3) deslocada pela espessura T ao longo da normal."""
    du = np.gradient(S, axis=1); dv = np.gradient(S, axis=0)
    N = np.cross(du, dv)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
    if center is not None:
        sgn = np.sign(np.sum(N * (S - center), -1).mean())
        N *= sgn if sgn != 0 else 1
    if not outward:
        N = -N
    return S + N * T[..., None], S


def _border_T(T, v, rows_b, cols_b, sides="all"):
    """Acrescenta uma margem (Bright = fina, Dark = espessa) à volta da matriz de espessuras."""
    bt = v["tmin"] if v.get("border", "Dark") == "Bright" else max(v.get("border_t", 4.0), v["tmin"])
    if rows_b <= 0 and cols_b <= 0:
        return T
    r, c = T.shape
    pr = (rows_b, rows_b) if sides in ("all", "tb") else (0, 0)
    pc = (cols_b, cols_b) if sides in ("all", "lr") else (0, 0)
    return np.pad(T, (pr, pc), constant_values=bt)


def _base_ring(R_out, R_in, h, z0=0.0, n=128):
    if h <= 0:
        return mf.Manifold()
    M = mf.Manifold.cylinder(h, R_out, R_out, n)
    if R_in > 0:
        M = M - mf.Manifold.cylinder(h + 2, R_in, R_in, n).translate((0, 0, -1))
    return M.translate((0, 0, z0))


def _poly_base(poly, h, hollow, wall, z0=0.0):
    if h <= 0:
        return mf.Manifold()
    M = C.ext(poly, h, z0)
    if hollow:
        inner = poly.buffer(-wall)
        if not inner.is_empty:
            M = M - C.ext(inner, h + 2, z0 - 1)
    return M


def _hanger(top_z, cx=0.0, cy=0.0, d=6.0, w=2.5, t=3.0, axis="y"):
    """Argola para pendurar, no topo."""
    ring = Point(0, 0).buffer(d / 2 + w, quad_segs=24).difference(Point(0, 0).buffer(d / 2, quad_segs=24))
    ring = ring.union(sbox(-(d / 2 + w), -(d / 2 + w) - 1, d / 2 + w, 0))
    M = C.ext(ring, t, -t / 2)
    M = M.rotate((90, 0, 0)) if axis == "y" else M.rotate((90, 0, 90))
    return M.translate((cx, cy, top_z + d / 2 + w))


def _stand_up(m, slant=0.0):
    """Placa construída em XY → de pé (altura em Z), inclinada para trás slant graus."""
    C.rot(m, 90 + slant, [1, 0, 0])
    return m


# --------------------------------------------------------------------------
# superfícies de revolução / prismas
# --------------------------------------------------------------------------
def _path_poly(n, R, k=256, phase=None):
    """Polígono regular (ou círculo se n == 0) amostrado em k pontos, sentido anti-horário."""
    t = np.linspace(0, 1, k, endpoint=False)
    if n <= 2:
        a = 2 * np.pi * t
        return np.c_[np.cos(a), np.sin(a)] * R
    a0 = -np.pi / 2 - np.pi / n if phase is None else phase
    P = np.array([[math.cos(a0 + 2 * np.pi * i / n), math.sin(a0 + 2 * np.pi * i / n)] for i in range(n + 1)]) * R
    s = t * n
    i = np.floor(s).astype(int); f = (s - i)[:, None]
    return P[i] * (1 - f) + P[i + 1] * f


def _wrap(v, fast, n_sides, R_bot, R_top, H, profile=None, fixed=False, perspective=True):
    """Lâmpada com a fotografia à volta: n_sides = 0 → cilindro/cone; ≥ 3 → prisma/tronco."""
    res = _res(v, 2 * math.pi * max(R_bot, R_top), fast)
    circ = (2 * math.pi if n_sides <= 2 else 2 * n_sides * math.sin(math.pi / n_sides)) * max(R_bot, R_top)
    cols = int(np.clip(circ / res, 24, 900)); rows = int(np.clip(H / res, 8, 600))
    if n_sides >= 3:
        cols = (cols // n_sides) * n_sides
    upright = v.get("orient", "Upright") == "Upright"
    if fixed:
        g0 = C.load_gray(v.get("photo", ""), 700)
        if not upright:
            g0 = g0[::-1, ::-1]
        wcols = int(min(cols, rows * g0.shape[1] / g0.shape[0]))
        g = np.ones((rows, cols)) * (1.0 if v.get("polarity", "Positive") == "Positive" else 0.0)
        c0 = (cols - wcols) // 2
        g[:, c0:c0 + wcols] = C.resample(g0, rows, max(wcols, 1))
    elif n_sides >= 3 and not v.get("wrap_all", False):
        # uma fotografia por face
        per = cols // n_sides
        g1 = _photo(v, rows, per, upright)
        g = np.tile(g1, (1, n_sides))
    else:
        g = _photo(v, rows, cols, upright)
    T = _T(v, g)
    zs = np.linspace(H, 0, rows)
    f = (zs / H)[:, None]
    Rz = R_bot + (R_top - R_bot) * f
    if profile is not None:
        Rz = Rz * profile(zs / H)[:, None]
    P = _path_poly(n_sides, 1.0, cols)
    S = np.zeros((rows, cols, 3))
    S[..., 0] = P[None, :, 0] * Rz; S[..., 1] = P[None, :, 1] * Rz; S[..., 2] = zs[:, None]
    ext_ = v.get("side", "Exterior") == "Exterior"
    Po, Pi = _surface(S, T, outward=True, center=np.array([0, 0, H / 2]))
    if not ext_:
        Po, Pi = S, S - (Po - S)
    m = C.grid_solid(Po, Pi, wrap=True)
    return m, Rz


def _rim(n_sides, R, w, h, z0, hollow=True, inner_R=None):
    if n_sides >= 3:
        poly = Polygon(_path_poly(n_sides, R + w, n_sides * 4))
        M = C.ext(poly, h, z0)
        if hollow:
            M = M - C.ext(Polygon(_path_poly(n_sides, inner_R if inner_R is not None else R - 0.2, n_sides * 4)), h + 2, z0 - 1)
        return M
    return _base_ring(R + w, (inner_R if inner_R is not None else R - 0.2) if hollow else 0, h, z0)


def _lamp(v, fast, n_sides, R_bot, R_top, H, profile=None, fixed=False):
    """Lâmpada completa: corpo com a fotografia, margens em cima/em baixo, base e argola."""
    tmax = v["tmax"]
    m, Rz = _wrap(v, fast, n_sides, R_bot, R_top, H, profile, fixed)
    bs = v.get("border_size", 2.0)
    parts = [C.to_mf(m)]
    bt = v["tmin"] if v.get("border", "Dark") == "Bright" else max(v.get("border_t", 4.0), v["tmin"])
    Rb, Rt = float(Rz[-1, 0]), float(Rz[0, 0])
    if bs > 0:
        parts.append(_rim(n_sides, Rb, bt, bs + 0.3, -bs, True, Rb - 0.1))
        parts.append(_rim(n_sides, Rt, bt, bs + 0.3, H - 0.3, True, Rt - 0.1))
    bh = v.get("base_t", 4.0)
    if bh > 0:
        z0 = -bs - bh + 0.01
        wall = bt + tmax
        hollow = v.get("base", "Solid") == "Hollow"
        if n_sides >= 3:
            poly = Polygon(_path_poly(n_sides, Rb + bt, n_sides * 4))
            parts.append(_poly_base(poly, bh, hollow, wall + 2, z0))
        else:
            parts.append(_base_ring(Rb + bt, (Rb - wall) if hollow else 0, bh, z0))
    out = C.union(parts)
    if v.get("hanger", "No") == "Yes":
        out = C.union([out, _hanger(out.bounds[1][2] - 0.5, 0, Rt + bt / 2 - 1)])
    return C.on_floor(out)


# --------------------------------------------------------------------------
# formas
# --------------------------------------------------------------------------
def flat(v, fast):
    W = v["sx"]; Hh = v["sz"]
    res = _res(v, W, fast)
    g0 = C.load_gray(v.get("photo", ""), 700)
    if Hh <= 0:
        Hh = W * g0.shape[0] / g0.shape[1]
    rows, cols = C.grid_for(g0, W, Hh, res)
    T = _T(v, C.resample(g0, rows, cols))
    bs = v.get("border_size", 2.0)
    nb = int(round(bs / res))
    T = _border_T(T, v, nb, nb)
    Wt, Ht = W + 2 * bs, Hh + 2 * bs
    plate = C.plate_relief(T - v["tmin"], Wt, Ht, base=v["tmin"])
    parts = [C.to_mf(plate)]
    if v.get("hanger", "No") == "Yes":
        parts.append(_hanger(0, 0, Ht / 2, axis="x").rotate((-90, 0, 0)).translate((0, 0, 0)))
    m = C.union(parts)
    m.apply_translation((0, 0, 0))
    slant = v.get("slant", 0.0)
    _stand_up(m, slant)
    m = C.on_floor(m)
    parts = [C.to_mf(m)]
    # base com casquilho (O = anel fechado; U = aberto atrás)
    bl, bth = v.get("base_len", 10.0), v.get("base_t", 4.0)
    if bl > 0 and bth > 0:
        y0, y1 = m.bounds[0][1], m.bounds[1][1]
        depth = max(bl, (y1 - y0) + 4)
        cy = (y0 + y1) / 2
        parts = [C.to_mf(m).translate((0, 0, bth - 0.6))]
        parts.append(C.ext(sbox(-Wt / 2, cy - depth / 2, Wt / 2, cy + depth / 2), bth))
        sd, sh = v.get("socket_d", 20.0), v.get("socket_h", 10.0)
        if sd > 0 and sh > 0:
            wall = 2.0
            ring = Point(0, 0).buffer(sd / 2 + wall, quad_segs=48).difference(Point(0, 0).buffer(sd / 2, quad_segs=48))
            if v.get("socket", "O") == "U":
                ring = ring.difference(sbox(-sd / 2, 0, sd / 2, sd))
            yc = cy + depth / 2 + sd / 2 + wall - 0.5
            parts.append(C.ext(affinity.translate(ring, 0, yc), sh + bth))
            parts.append(C.ext(sbox(-sd / 2 - wall, cy, sd / 2 + wall, yc), bth))
    if v.get("support", "No") == "Yes":
        # escoras triangulares atrás da placa
        H_ = m.bounds[1][2]
        tri = Polygon([(0, 0), (H_ * 0.35, 0), (0, H_ * 0.5)])
        for x in (-Wt / 2 + 3, Wt / 2 - 5):
            st = C.ext(tri, 2.0).rotate((90, 0, 90)).translate((x, m.bounds[1][1] - 0.5, 0))
            parts.append(st)
    return [C.on_floor(C.union(parts))]


def arc(v, fast):
    from .litho import bend
    ang = max(v.get("arc_angle", 30.0), 1.0)
    Hh = v["sz"]
    g0 = C.load_gray(v.get("photo", ""), 700)
    W = Hh * g0.shape[1] / g0.shape[0]
    res = _res(v, W, fast)
    rows, cols = C.grid_for(g0, W, Hh, res)
    T = _T(v, C.resample(g0, rows, cols))
    bs = v.get("border_size", 2.0); nb = int(round(bs / res))
    T = _border_T(T, v, nb, nb)
    plate = C.plate_relief(T - v["tmin"], W + 2 * bs, Hh + 2 * bs, base=v["tmin"])
    if v.get("side", "Exterior") == "Interior":
        plate.apply_transform(trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1]))
        plate.invert(); plate.apply_translation((0, 0, -plate.bounds[0][2]))
    m = bend(plate, ang)
    _stand_up(m)
    return flat_base(C.on_floor(m), v)


def flat_base(m, v):
    bl, bth = v.get("base_len", 10.0), v.get("base_t", 4.0)
    if bl <= 0 or bth <= 0:
        return [m]
    lo, hi = m.bounds
    cy = (lo[1] + hi[1]) / 2
    depth = max(bl, hi[1] - lo[1] + 4)
    base = C.ext(sbox(lo[0] - 1, cy - depth / 2, hi[0] + 1, cy + depth / 2), bth)
    return [C.on_floor(C.union([C.to_mf(m).translate((0, 0, bth - 0.6)), base]))]


def cmyk(v, fast):
    W, Hh = v["sx"], v["sy"]
    res = _res(v, W, fast)
    rgb = C.load_rgb(v.get("photo", ""), 700)
    rows, cols = C.grid_for(rgb, W, Hh, res)
    rgb = C.resample(rgb, rows, cols)
    g = C.to_gray(rgb)
    T = C.litho_T(g, v["tmin"], v["tmax"])
    white = C.plate_relief(T - v["tmin"], W, Hh, base=v["tmin"])
    ct = v.get("color_t", 0.6)
    clar = max(v.get("clarity", 8.0), 0.1)
    k = np.clip(1 - rgb.max(-1), 0, 1)
    out, names, cols_ = [white], ["Branco (luminância)"], [(0.95, 0.95, 0.95)]
    z = 0.0
    for ch, nm, col in ((0, "Ciano", (0.0, 0.7, 0.9)), (1, "Magenta", (0.9, 0.1, 0.6)), (2, "Amarelo", (0.98, 0.9, 0.1))):
        dens = np.clip((1 - rgb[..., ch] - k) / np.maximum(1 - k, 1e-3), 0, 1) ** (8.0 / clar)
        z -= ct
        lay = C.plate_relief(np.maximum(dens * ct, 0.02), W, Hh, base=0.0)
        lay.apply_translation((0, 0, z))
        out.append(lay); names.append(nm); cols_.append(col)
    out = C.group_floor(out)
    return out, names, cols_


def dual(v, fast):
    W, Hh = v["sx"], v["sz"]
    n = max(int(v.get("sections", 3)), 1)
    res = _res(v, W, fast)
    g0 = C.load_gray(v.get("photo", ""), 700)
    rows, cols = C.grid_for(g0, W, Hh, res)
    cols = max((cols // n) * n, n * 4)
    T = _T(v, C.resample(g0, rows, cols))
    bs = v.get("border_size", 2.0); nb = int(round(bs / res))
    T = _border_T(T, v, nb, 0, "tb")
    rows = T.shape[0]
    Ht = Hh + 2 * bs
    # acordeão: cada secção é uma face, alternando ±35°
    ang = math.radians(35)
    seg = W / n
    u = np.linspace(0, W, cols)
    k = np.minimum((u // seg).astype(int), n - 1)
    f = u - k * seg
    x = k * seg * math.cos(ang) + f * math.cos(ang)
    y = np.where(k % 2 == 0, f, seg - f) * math.sin(ang)
    S = np.zeros((rows, cols, 3))
    S[..., 0] = x[None]; S[..., 2] = np.linspace(Ht, 0, rows)[:, None]; S[..., 1] = y[None]
    Po, Pi = _surface(S, T, True)
    if np.mean(Po[..., 1] - Pi[..., 1]) < 0:
        Po, Pi = Pi, Po
    m = C.grid_solid(Po, Pi)
    m.apply_translation((-m.bounds[:, 0].mean(), -m.bounds[:, 1].mean(), 0))
    return [C.on_floor(m)]


def rectangle(v, fast):
    W, D, Hh = v["sx"], v["sy"], v["sz"]
    # retângulo: lados com o tamanho pedido (mapeia o caminho num retângulo)
    return [_box_lamp(v, fast, W, D, Hh)]


def _box_lamp(v, fast, W, D, Hh):
    res = _res(v, 2 * (W + D), fast)
    per = 2 * (W + D)
    cols = int(np.clip(per / res, 32, 1000)); rows = int(np.clip(Hh / res, 8, 600))
    upright = v.get("orient", "Upright") == "Upright"
    # uma fotografia em cada face
    nW = max(int(cols * W / per), 4); nD = max(int(cols * D / per), 4)
    gW = _photo(v, rows, nW, upright); gD = _photo(v, rows, nD, upright)
    g = np.concatenate([gW, gD, gW, gD], 1)
    cols = g.shape[1]
    T = _T(v, g)
    s = np.r_[np.linspace(0, W, nW, endpoint=False), W + np.linspace(0, D, nD, endpoint=False),
              W + D + np.linspace(0, W, nW, endpoint=False), 2 * W + D + np.linspace(0, D, nD, endpoint=False)]
    corners = np.array([[-W / 2, -D / 2], [W / 2, -D / 2], [W / 2, D / 2], [-W / 2, D / 2], [-W / 2, -D / 2]])
    acc = np.r_[0, W, W + D, 2 * W + D, 2 * W + 2 * D]
    P = np.zeros((cols, 2))
    for i in range(4):
        sel = (s >= acc[i]) & (s < acc[i + 1])
        f = ((s[sel] - acc[i]) / (acc[i + 1] - acc[i]))[:, None]
        P[sel] = corners[i] * (1 - f) + corners[i + 1] * f
    zs = np.linspace(Hh, 0, rows)
    S = np.zeros((rows, cols, 3)); S[..., 0] = P[None, :, 0]; S[..., 1] = P[None, :, 1]; S[..., 2] = zs[:, None]
    Po, Pi = _surface(S, T, True, center=np.array([0, 0, Hh / 2]))
    m = C.grid_solid(Po, Pi, wrap=True)
    parts = [C.to_mf(m)]
    bs = v.get("border_size", 2.0)
    bt = v["tmin"] if v.get("border", "Dark") == "Bright" else max(v.get("border_t", 4.0), v["tmin"])
    rect = sbox(-W / 2, -D / 2, W / 2, D / 2)
    ring = rect.buffer(bt, join_style="mitre").difference(rect.buffer(-0.2, join_style="mitre"))
    if bs > 0:
        parts += [C.ext(ring, bs + 0.3, -bs), C.ext(ring, bs + 0.3, Hh - 0.3)]
    bh = v.get("base_t", 4.0)
    if bh > 0:
        base = rect.buffer(bt, join_style="mitre")
        parts.append(_poly_base(base, bh, v.get("base", "Solid") == "Hollow", bt + v["tmax"] + 2, -bs - bh + 0.01))
    out = C.union(parts)
    if v.get("hanger", "No") == "Yes":
        out = C.union([out, _hanger(out.bounds[1][2] - 0.5, 0, 0)])
    return C.on_floor(out)


def polygon(v, fast):
    n = max(int(v.get("sides", 3)), 3)
    W = v["sx"]
    R = W / 2
    if v.get("body", "One body") == "Assembly":
        return _assembly_panels(v, fast, n, R, R, v["sz"])
    return [_lamp(v, fast, n, R, R, v["sz"])]


def _assembly_panels(v, fast, n, Rb, Rt, H):
    """Faces separadas (para imprimir deitadas) + base com ranhuras."""
    side = 2 * Rb * math.sin(math.pi / n)
    res = _res(v, side, fast)
    g0 = C.load_gray(v.get("photo", ""), 700)
    rows, cols = C.grid_for(g0, side, H, res)
    T = _T(v, C.resample(g0, rows, cols))
    out, names = [], []
    for i in range(n):
        p = C.plate_relief(T - v["tmin"], side, H, base=v["tmin"])
        p.apply_translation((i * (side + 5), 0, 0))
        out.append(p); names.append(f"Face {i + 1}")
    poly = Polygon(_path_poly(n, Rb + 3, n * 4))
    base = C.to_tm(C.ext(poly, v.get("base_t", 4.0) + 3) - C.ext(Polygon(_path_poly(n, Rb - v["tmax"] - 0.4, n * 4)).difference(
        Polygon(_path_poly(n, Rb - v["tmax"] - 3, n * 4))), 4, v.get("base_t", 4.0)))
    base.apply_translation((0, -H, 0))
    out.append(base); names.append("Base")
    return out, names, None


def frustum(v, fast):
    n = max(int(v.get("sides", 3)), 3)
    Rt, Rb = v["top_x"] / 2, v["bot_x"] / 2
    return [_lamp(v, fast, n, Rb, Rt, v["sz"])]


def cylinder(v, fast):
    R = v["diam"] / 2
    return [_lamp(v, fast, 0, R, R, v["sz"], fixed=v.get("fit", "Fixed") == "Fixed")]


def cone(v, fast):
    return [_lamp(v, fast, 0, v["bot_d"] / 2, v["top_d"] / 2, v["sz"], fixed=v.get("fit", "Fixed") == "Fixed")]


def vase(v, fast):
    from scipy.interpolate import CubicSpline
    n = max(int(v.get("ctrl", 3)), 2)
    ks = [v.get(f"cp{i}", 1.0) for i in range(1, 4)][:n]
    while len(ks) < n:
        ks.append(1.0)
    t = np.linspace(0, 1, n)
    cs = CubicSpline(t, ks, bc_type="natural") if n > 2 else (lambda x: np.interp(x, t, ks))
    prof = lambda z: np.clip(cs(z), 0.2, 3)
    R = v["diam"] / 2
    return [_lamp(v, fast, 0, R, R, v["sz"], profile=prof, fixed=v.get("fit", "Fixed") == "Fixed")]


def sphere(v, fast):
    D = v["diam"]; R = D / 2 - v["tmax"]
    ang = np.clip(v.get("sphere_angle", 180.0), 10, 360)
    op = v.get("opening", 0.0) or D * 0.436 * 2 / 2
    res = _res(v, math.pi * D, fast)
    cols = int(np.clip(math.pi * D / res, 48, 900)); rows = int(np.clip(math.pi * D / 2 / res, 24, 500))
    lat_min = -math.asin(min(op / 2 / R, 0.98))
    lats = np.linspace(math.pi / 2 * 0.985, lat_min, rows)
    lons = np.linspace(0, 2 * math.pi, cols, endpoint=False)
    # fotografia na faixa de longitude pedida (o resto fica claro = fino)
    g = np.ones((rows, cols)) * (1.0 if v.get("polarity", "Positive") == "Positive" else 0.0)
    wcols = max(int(cols * ang / 360), 2)
    pos = v.get("align", "Center to surface center")
    g0 = C.load_gray(v.get("photo", ""), 700)
    rr = rows
    if pos.startswith("Top"):
        sub = C.resample(g0, rows // 2, wcols); g[:rows // 2, :wcols] = sub
    elif pos.startswith("Bottom"):
        sub = C.resample(g0, rows // 2, wcols); g[rows - rows // 2:, :wcols] = sub
    else:
        g[:, :wcols] = C.resample(g0, rr, wcols)
    g = np.roll(g, (cols - wcols) // 2 + int(v.get("bottom_off", 0.0) / 360 * cols), 1)
    T = _T(v, g)
    S = np.zeros((rows, cols, 3))
    S[..., 0] = R * np.cos(lats)[:, None] * np.cos(lons)[None]
    S[..., 1] = R * np.cos(lats)[:, None] * np.sin(lons)[None]
    S[..., 2] = R * np.sin(lats)[:, None] + np.zeros(cols)[None]
    Po, Pi = _surface(S, T, True, center=np.zeros(3))
    if v.get("side", "Exterior") == "Interior":
        Po, Pi = S, S - (Po - S)
    m = C.grid_solid(Po, Pi, wrap=True)
    # tampa do polo (pequeno disco) e casquilho em baixo
    parts = [C.to_mf(m)]
    ztop = float(Pi[0, :, 2].min())
    rtop = float(np.linalg.norm(Po[0, :, :2], axis=1).max())
    parts.append(mf.Manifold.cylinder(R + v["tmax"] - ztop + 0.5, rtop + 0.3, rtop + 0.3, 64).translate((0, 0, ztop - 0.3)) ^
                 mf.Manifold.sphere(R + v["tmin"], 96))
    zb = float(S[-1, 0, 2])
    rb = float(np.linalg.norm(S[-1, 0, :2]))
    sd, sh = v.get("socket_d", 20.0), v.get("socket_h", 10.0)
    bt = v.get("base_t", 4.0)
    if sd > 0 and sh > 0:
        Ro = max(rb + v["tmax"], sd / 2 + 2)
        parts.append(_base_ring(Ro, sd / 2, sh, zb - sh + 1.0))
    if v.get("border_size", 2.0) > 0:
        parts.append(_base_ring(rb + v.get("border_t", 4.0), rb - 0.5, v.get("border_size", 2.0) + 0.5, zb - 0.5))
    if bt > 0:
        zc = zb - (sh if sd > 0 and sh > 0 else 0) - bt + 1.0
        parts.append(_base_ring(max(rb + v["tmax"], sd / 2 + 2) + 4, sd / 2 if sd > 0 else 0, bt, zc))
    return [C.on_floor(C.union(parts))]


def spiral(v, fast):
    D, Hh = v["diam"], v["sz"]
    R = D / 2
    turns = 1.25
    t = np.linspace(0, 1, 2000)
    r = R * (0.35 + 0.65 * t)
    a = 2 * math.pi * turns * t
    xy = np.c_[r * np.cos(a), r * np.sin(a)]
    L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    res = _res(v, L, fast)
    g0 = C.load_gray(v.get("photo", ""), 700)
    rows, cols = int(np.clip(Hh / res, 8, 500)), int(np.clip(L / res, 32, 1200))
    T = _T(v, C.resample(g0, rows, cols))
    bs = v.get("border_size", 2.0); nb = int(round(bs / res))
    T = _border_T(T, v, nb, 0, "tb"); rows = T.shape[0]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    su = np.linspace(0, L, cols)
    P = np.c_[np.interp(su, s, xy[:, 0]), np.interp(su, s, xy[:, 1])]
    S = np.zeros((rows, cols, 3)); S[..., :2] = P[None]; S[..., 2] = np.linspace(Hh + 2 * bs, 0, rows)[:, None]
    Po, Pi = _surface(S, T, True, center=np.array([0, 0, Hh / 2]))
    m = C.grid_solid(Po, Pi)
    return [C.on_floor(m)], {"Comprimento do arco": f"{L:.0f} mm"}


def _template(v, W, H):
    tp = v.get("template", "")
    if tp:
        g = C.load_gray(tp, 400)
        geom = C.mask_to_geom(C.photo_mask(g, True), 0.6)
        if not geom.is_empty:
            return C.fit(geom, W, H)
    # coração por omissão
    t = np.linspace(0, 2 * np.pi, 200)
    p = Polygon(np.c_[16 * np.sin(t) ** 3, 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)])
    return C.fit(p, W, H)


def _placed_photo(v, rows, cols):
    """Fotografia posicionada pelo centro (−1..1) e escala, sobre fundo claro."""
    g0 = C.load_gray(v.get("photo", ""), 700)
    sc = max(v.get("pscale", 1.0), 0.05)
    cx, cy = v.get("pcx", 0.0), v.get("pcy", 0.0)
    out = np.ones((rows, cols))
    h = max(int(rows * sc), 2); w = max(int(cols * sc), 2)
    sub = C.resample(g0, h, w)
    r0 = int(rows / 2 - h / 2 - cy * rows / 2); c0 = int(cols / 2 - w / 2 + cx * cols / 2)
    rs, cs_ = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + h, rows), min(c0 + w, cols)
    if re > rs and ce > cs_:
        out[rs:re, cs_:ce] = sub[rs - r0:re - r0, cs_ - c0:ce - c0]
    return out


def stencil(v, fast):
    W, Hh = v["sx"], v["sz"]
    geom = _template(v, W, Hh)
    b = geom.bounds; W2, H2 = b[2] - b[0], b[3] - b[1]
    res = _res(v, W2, fast)
    rows, cols = int(np.clip(H2 / res, 8, 700)), int(np.clip(W2 / res, 8, 700))
    g = _placed_photo(v, rows, cols)
    T = _T(v, g)
    bs = v.get("border_size", 2.0)
    dbl = v.get("sided", "Single-sided") == "Double-sided"
    plate = C.plate_relief(T - v["tmin"], W2, H2, base=v["tmin"] / (2 if dbl else 1))
    M = C.to_mf(plate) ^ C.ext(geom, v["tmax"] + 2, -1)
    if dbl:
        Mb = M.mirror((0, 0, 1))
        M = M + Mb.translate((0, 0, 0.01))
    if bs > 0:
        bt = v["tmin"] if v.get("border", "Dark") == "Bright" else v["tmax"]
        ring = geom.buffer(bs).difference(geom.buffer(-0.2))
        M = M + C.ext(ring, bt, -bt if dbl else 0) + (C.ext(ring, bt, 0) if dbl else mf.Manifold())
    out = [C.on_floor(C.to_tm(M))]
    if v.get("cmyk", "No") == "Yes":
        rgb = C.resample(C.load_rgb(v.get("photo", ""), 600), rows, cols)
        names = ["Litofania"]; cols_ = [(0.95, 0.95, 0.95)]
        ct = v.get("color_t", 0.6)
        for ch, nm, col in ((0, "Ciano", (0.0, 0.7, 0.9)), (1, "Magenta", (0.9, 0.1, 0.6)), (2, "Amarelo", (0.98, 0.9, 0.1))):
            dens = np.clip(1 - rgb[..., ch], 0, 1) ** (8.0 / max(v.get("clarity", 8.0), 0.1))
            lay = C.to_mf(C.plate_relief(np.maximum(dens * ct, 0.02), W2, H2)) ^ C.ext(geom, ct + 2, -1)
            lm = C.to_tm(lay); lm.apply_translation((0, 0, -ct * (len(out))))
            out.append(lm); names.append(nm); cols_.append(col)
        return C.group_floor(out), names, cols_
    return out


def balloon(v, fast):
    W, Z = v["sx"], v["sz"]
    geom = _template(v, W, W)
    b = geom.bounds; W2, H2 = b[2] - b[0], b[3] - b[1]
    res = _res(v, W2, fast)
    rows, cols = int(np.clip(H2 / res, 8, 500)), int(np.clip(W2 / res, 8, 500))
    # máscara rasterizada da silhueta e altura de "insuflado"
    from shapely import contains_xy
    xs = np.linspace(b[0], b[2], cols); ys = np.linspace(b[3], b[1], rows)
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(geom, X, Y)
    d = ndimage.distance_transform_edt(inside) * res
    dn = d / max(d.max(), 1e-6)
    h = (Z / 2) * np.sqrt(np.clip(dn, 0, 1)) ** (1.0 / max(v.get("expansion", 1.0), 0.1)) + v.get("extra", 0.0)
    g = _placed_photo(v, rows, cols)
    T = _T(v, g)
    S = np.dstack([X, Y, h])
    Po = S.copy(); Po[..., 2] = h + T
    top = C.grid_solid(Po, S)
    bot = top.copy(); bot.apply_transform(trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1])); bot.invert()
    if v.get("side", "Exterior") == "Interior":
        pass
    M = (C.to_mf(top) + C.to_mf(bot)) ^ C.ext(geom.buffer(0.01), Z * 2 + 20, -Z - 10)
    return [C.on_floor(C.to_tm(M))]


def box(v, fast):
    W, D, Hh = v["sx"], v["sy"], v["sz"]
    bt = v.get("box_t", 5.0)
    body = _box_lamp(dict(v, base_t=0, border_size=bt, base="Solid", hanger="No", border="Dark", border_t=bt), fast, W, D, Hh)
    # fundo
    rect = sbox(-W / 2 - bt, -D / 2 - bt, W / 2 + bt, D / 2 + bt)
    bottom = C.ext(rect, bt)
    bm = C.union([C.to_mf(body).translate((0, 0, bt - 0.01)), bottom])
    # tampa com encaixe (folga = tolerance)
    tol = v.get("tol", -0.2)
    lip = sbox(-W / 2 + v["tmax"] + 0.2 - tol, -D / 2 + v["tmax"] + 0.2 - tol, W / 2 - v["tmax"] - 0.2 + tol, D / 2 - v["tmax"] - 0.2 + tol)
    lid = C.ext(rect, bt) + C.ext(lip, 3, bt - 0.01)
    if v.get("template", ""):
        tpl = _template(v, W * 0.7, D * 0.7)
        lid = lid - C.ext(tpl, bt + 6, -1)
    lid = C.to_tm(lid)
    C.rot(lid, 180, [1, 0, 0]); C.on_floor(lid)
    lid.apply_translation((W + 2 * bt + 10, 0, 0))
    return [C.on_floor(bm), lid], ["Caixa", "Tampa"], None


def _mesh_src(v):
    p = v.get("model", "")
    if p:
        try:
            m = trimesh.load(p, force="mesh")
            if len(m.faces):
                return m
        except Exception:
            pass
    m = C.CURRENT.get("mesh")
    if m is not None:
        return m.copy()
    return trimesh.creation.capsule(40, 15, count=[32, 32])


def _depth_map(m, rows, cols, axis=2):
    """Mapa de alturas visto de cima (+Z) por amostragem densa da superfície."""
    pts = trimesh.sample.sample_surface_even(m, min(400000, rows * cols * 8))[0] if len(m.faces) else np.zeros((0, 3))
    lo, hi = m.bounds
    ax = [i for i in range(3) if i != axis]
    u = ((pts[:, ax[0]] - lo[ax[0]]) / max(hi[ax[0]] - lo[ax[0]], 1e-9) * (cols - 1)).astype(int)
    w = ((hi[ax[1]] - pts[:, ax[1]]) / max(hi[ax[1]] - lo[ax[1]], 1e-9) * (rows - 1)).astype(int)
    Z = np.full((rows, cols), np.nan)
    val = (pts[:, axis] - lo[axis]) / max(hi[axis] - lo[axis], 1e-9)
    flat = w * cols + u
    out = np.full(rows * cols, -1.0)
    np.maximum.at(out, flat, val)
    Z = out.reshape(rows, cols)
    mask = Z >= 0
    Z[~mask] = 0
    Z = ndimage.grey_closing(Z, size=3)
    return Z


def bronze(v, fast):
    W, D = v["sx"], v["sy"]
    m = _mesh_src(v)
    res = _res(v, W, fast) if "res" in v else max(W / (110 if fast else 300), 0.1)
    rows, cols = int(np.clip(D / res, 16, 500)), int(np.clip(W / res, 16, 500))
    Z = _depth_map(m, rows, cols)
    c = v.get("contrast", 0.0)
    g = 1 - Z  # o relevo do modelo aparece com a luz atrás
    g = np.clip((g - 0.5) * (1 + c / 50) + 0.5, 0, 1)
    T = _T(v, g)
    plate = C.plate_relief(T - v["tmin"], W, D, base=v["tmin"])
    if v.get("side", "Exterior") == "Interior":
        plate.apply_transform(trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1])); plate.invert()
    return [C.on_floor(plate)]


DIRS = {"+X": (1, 0, 0), "+Y": (0, 1, 0), "+Z": (0, 0, 1), "-X": (-1, 0, 0), "-Y": (0, -1, 0), "-Z": (0, 0, -1)}


def _fit_model(m, v):
    e = np.maximum(m.extents, 1e-9)
    s = np.array([v["sx"], v["sy"], v["sz"]]) / e
    m = m.copy(); m.apply_scale(s)
    m.apply_translation(-m.bounds.mean(0))
    return m


def _refined(m, fast, target=None):
    M = C.to_mf(m)
    L = target or max(m.extents) / (60 if fast else 180)
    return C.to_tm(M.refine_to_length(L))


def tattoo(v, fast):
    m = _refined(_fit_model(_mesh_src(v), v), fast)
    d = np.array(DIRS.get(v.get("direction", "+Z"), (0, 0, 1)), float)
    V = m.vertices; N = m.vertex_normals
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    u = V @ e1; w = V @ e2
    sx = max(np.ptp(u), 1e-9) * max(v.get("pscale_x", 1.0), 0.05); sy = max(np.ptp(w), 1e-9) * max(v.get("pscale_y", 1.0), 0.05)
    uu = (u - u.mean()) / sx + 0.5 - v.get("pcx", 0.0) / 2
    ww = (w - w.mean()) / sy + 0.5 - v.get("pcy", 0.0) / 2
    g0 = C.load_gray(v.get("photo", ""), 600)
    rr, cc = g0.shape
    inside = (uu >= 0) & (uu <= 1) & (ww >= 0) & (ww <= 1) & ((N @ d) > 0.15)
    gi = np.ones(len(V))
    gi[inside] = ndimage.map_coordinates(g0, [(1 - ww[inside]) * (rr - 1), uu[inside] * (cc - 1)], order=1)
    T = _T(v, gi)
    sgn = 1 if v.get("side", "Exterior") == "Exterior" else -1
    m2 = trimesh.Trimesh(V + N * (T - v["tmin"])[:, None] * sgn * inside[:, None], m.faces, process=False)
    return [C.on_floor(m2)]


def tile(v, fast):
    m = _refined(_fit_model(_mesh_src(v), v), fast)
    ax = {"X": 0, "Y": 1, "Z": 2}[v.get("tdir", "X")]
    V = m.vertices; N = m.vertex_normals
    o = [i for i in range(3) if i != ax]
    c = V.mean(0)
    th = np.arctan2(V[:, o[1]] - c[o[1]], V[:, o[0]] - c[o[0]]) + math.radians(v.get("angle", 0.0))
    uu = (th / (2 * math.pi)) % 1.0
    ww = (V[:, ax] - V[:, ax].min()) / max(np.ptp(V[:, ax]), 1e-9)
    g0 = C.load_gray(v.get("photo", ""), 600)
    rr, cc = g0.shape
    gi = ndimage.map_coordinates(g0, [(1 - ww) * (rr - 1), uu * (cc - 1)], order=1, mode="wrap")
    T = _T(v, gi)
    sgn = 1 if v.get("side", "Exterior") == "Exterior" else -1
    m2 = trimesh.Trimesh(V + N * (T - v["tmin"])[:, None] * sgn, m.faces, process=False)
    return [C.on_floor(m2)]


FN = {"Flat": flat, "Arc": arc, "CMYK": cmyk, "Dual": dual, "Rectangle": rectangle, "Polygon": polygon, "Frustum": frustum,
      "Cylinder": cylinder, "Cone": cone, "Vase": vase, "Sphere": sphere, "Spiral": spiral, "Stencil": stencil,
      "Balloon": balloon, "Box": box, "Bronze": bronze, "Tattoo": tattoo, "Tile": tile}


def run(mesh, v, fast):
    if v["tmax"] <= v["tmin"]:
        raise ValueError("A espessura máxima tem de ser maior do que a mínima.")
    r = FN[v["method"]](v, fast)
    extra = None
    if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], dict):
        r, extra = r
    if isinstance(r, tuple):
        ms, names, cols = r
    else:
        ms, names, cols = r, None, None
    if v.get("body", "One body") == "Assembly" and len(ms) == 1:
        pass
    return C.result(ms, names, cols, extra)
