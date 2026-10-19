"""Objetos do menu Criar: Projection (Fence, Pandora, Squama, Torch), Bottle, Bamboo-copter, Dice, Eyeglasses
e Glider — com os mesmos parâmetros do painel."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from scipy import ndimage
from shapely.geometry import Point, Polygon, LineString, box as sbox
from shapely import affinity
from shapely.ops import unary_union
from . import cimg as C


def _M(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add) if len(ms) > 1 else (ms[0] if ms else mf.Manifold())


def _rounded_box(x, y, z, r):
    if r <= 0:
        return mf.Manifold.cube((x, y, z), True)
    r = min(r, x / 2 - 0.01, y / 2 - 0.01, z / 2 - 0.01)
    s = [mf.Manifold.sphere(r, 24).translate((sx * (x / 2 - r), sy * (y / 2 - r), sz * (z / 2 - r)))
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return mf.Manifold.batch_hull(s)


# ==========================================================================
# Projection
# ==========================================================================
def fence(v, fast):
    g = C.load_gray(v.get("shadow", ""), 300)
    mask = C.photo_mask(g, True)
    rows, cols = mask.shape
    Ls = v["shadow_len"]; Ws = Ls * cols / rows
    y0 = v["shadow_off"]
    Ld, Lh = v["light_dist"], v["light_h"]
    n = max(int(v["fences"]), 1)
    fw = v["fence_w"]; t = v["thick"]
    L = np.array([0.0, -Ld, Lh])
    # sombra no chão: linha 0 da imagem = mais longe
    geom = C.mask_to_geom(mask, 0.6)
    geom = affinity.scale(geom, Ws / cols, Ls / rows, origin=(0, 0))
    geom = affinity.translate(geom, -Ws / 2, y0)
    parts = []
    for k in range(n):
        # cada vedação projeta uma faixa da sombra
        ya, yb = y0 + Ls * k / n, y0 + Ls * (k + 1) / n
        band = geom.intersection(sbox(-Ws, ya, Ws, yb))
        if band.is_empty:
            continue
        yf = -k * fw * 3.0  # posição da vedação (à frente da sombra)

        def proj(x, y):
            tt = (yf - L[1]) / (y - L[1])
            return x * tt, L[2] * (1 - tt)
        polys = []
        for p in getattr(band, "geoms", [band]):
            if p.geom_type != "Polygon":
                continue
            ext = [proj(x, y) for x, y in p.exterior.coords]
            holes = [[proj(x, y) for x, y in h.coords] for h in p.interiors]
            polys.append(Polygon(ext, holes).buffer(0))
        wall = unary_union(polys)
        if wall.is_empty:
            continue
        b = wall.bounds
        # pés e travessa de apoio no chão
        posts = unary_union([sbox(b[0] - fw, 0, b[0], b[3]), sbox(b[2], 0, b[2] + fw, b[3]), sbox(b[0] - fw, 0, b[2] + fw, fw)])
        frame = wall.union(posts)
        parts.append(C.ext(frame, t, -t / 2).rotate((90, 0, 0)).translate((0, yf, 0)))
    return [C.on_floor(C.to_tm(_M(parts)))], {"Luz": f"x 0, y {-Ld:.0f}, z {Lh:.0f} mm", "Sombra": f"{Ws:.0f} × {Ls:.0f} mm"}


def pandora(v, fast):
    sx, sy = v["sock_x"], v["sock_y"]
    t = v["thick"]
    tol = v.get("tol", 0.0)
    base = C.ext(sbox(-sx / 2, -sy / 2, sx / 2, sy / 2).buffer(3, join_style="round"), t)
    sd, sh = v["sock_d"], v["sock_h"]
    ring = C.ext(Point(0, 0).buffer(sd / 2 + 2).difference(Point(0, 0).buffer(sd / 2 + tol)), sh, t - 0.01)
    stp = mf.Manifold.cylinder(v["stop_h"], v["stop_d"] / 2, v["stop_d"] / 2, 32).translate((0, 0, t + sh - 0.01))
    n = max(int(v["pillars"]), 1)
    ps, ph = v["pillar_s"], v["pillar_h"]
    R = min(sx, sy) / 2 - ps
    pillars = [mf.Manifold.cylinder(ph, ps / 2, ps / 2, 24).translate((R * math.cos(2 * math.pi * k / n + math.pi / 2),
                                                                         R * math.sin(2 * math.pi * k / n + math.pi / 2), t - 0.01))
               for k in range(n)]
    body = _M([base, ring, stp] + pillars)
    # tampa: estêncil da sombra (projetada pela luz até ao tamanho pedido) ou litofania
    top_d = min(sx, sy)
    lp = v.get("litho", "")
    if lp:
        g = C.load_gray(lp, 300)
        res = max(v["litho_res"], top_d / (100 if fast else 400))
        r_, c_ = int(top_d / res), int(top_d / res)
        T = C.litho_T(C.resample(g, r_, c_), v["litho_min"], v["litho_max"])
        lid = C.to_mf(C.plate_relief(T - v["litho_min"], top_d, top_d, base=v["litho_min"])) ^ C.ext(Point(0, 0).buffer(top_d / 2), 50, -1)
    else:
        g = C.load_gray(v.get("shadow", ""), 300)
        geom = C.fit(C.mask_to_geom(C.photo_mask(g, True), 0.6), top_d * 0.8, top_d * 0.8)
        disk = Point(0, 0).buffer(top_d / 2, quad_segs=48)
        lid = C.ext(disk.difference(geom), t)
    holes = [mf.Manifold.cylinder(t * 4, ps / 2 + tol + 0.15, ps / 2 + tol + 0.15, 24).translate(
        (R * math.cos(2 * math.pi * k / n + math.pi / 2), R * math.sin(2 * math.pi * k / n + math.pi / 2), -t)) for k in range(n)]
    lid = lid - _M(holes) + (C.ext(Point(0, 0).buffer(top_d / 2 + 3).difference(Point(0, 0).buffer(top_d / 2 - 0.2)), t)
                              if lp else mf.Manifold())
    lm = C.to_tm(lid); lm.apply_translation((sx + 15, 0, 0))
    Lh = v["light_h"]
    k = v["shadow_size"] / max(top_d * 0.8, 1)
    return [C.on_floor(C.to_tm(body)), C.on_floor(lm)], ["Base", "Tampa"], None, \
        {"Distância à parede para a sombra pedida": f"{Lh * k:.0f} mm"}


def squama(v, fast):
    W, Hh = v["sx"], v["sz"]
    cs = max(v["cell"], 1.0)
    if fast:
        cs = max(cs, W / 60)
    nx, ny = int(W / cs), int(Hh / cs)
    g = C.resample(C.load_gray(v.get("shadow", ""), 300), ny, nx)
    sub = 4
    sa = math.radians(v["surface_angle"]); la = math.radians(v["light_angle"])
    tilt = (g * sa + la * 0.5)  # células claras mais inclinadas (refletem/iluminam mais)
    u = (np.arange(sub) + 0.5) / sub * cs
    src = v.get("light", "Left")
    H = np.zeros((ny * sub, nx * sub))
    for i in range(ny):
        for j in range(nx):
            s = math.tan(tilt[i, j])
            if src in ("Left", "Right"):
                prof = (u if src == "Left" else cs - u) * s
                H[i * sub:(i + 1) * sub, j * sub:(j + 1) * sub] = prof[None, :]
            else:
                prof = (u if src == "Top" else cs - u) * s
                H[i * sub:(i + 1) * sub, j * sub:(j + 1) * sub] = prof[:, None]
    m = C.plate_relief(H, W, Hh, base=v["thick"])
    C.rot(m, 90, [1, 0, 0])
    return [C.on_floor(m)]


def torch(v, fast):
    fs, fd = v["film_size"], v["film_dist"]
    ss, sd = v["sock_size"], v["sock_depth"]
    t = v["thick"]; tol = v.get("tol", 0.0)
    rs = ss / 2 + tol + 0.1
    R2 = max(fs / 2 + 1, rs)
    # casquilho que encaixa na lanterna + cone até ao porta-filme
    sock = mf.Manifold.cylinder(sd, rs + t, rs + t, 96) - mf.Manifold.cylinder(sd + 1, rs, rs, 96).translate((0, 0, -0.5))
    cone = mf.Manifold.cylinder(fd, rs + t, R2 + t, 96) - mf.Manifold.cylinder(fd + 0.02, rs, R2, 96).translate((0, 0, -0.01))
    holder = mf.Manifold.cylinder(3, R2 + t, R2 + t, 96) - mf.Manifold.cylinder(4, fs / 2 - 1, fs / 2 - 1, 96).translate((0, 0, -0.5))
    body = _M([sock, cone.translate((0, 0, sd - 0.01)), holder.translate((0, 0, sd + fd - 0.02))])
    g = C.load_gray(v.get("shadow", ""), 300)
    geom = C.mask_to_geom(C.photo_mask(g, v.get("polarity", "Positive") == "Positive"), 0.5)
    shape = Point(0, 0).buffer(fs / 2, quad_segs=48) if v.get("shape", "Circle") == "Circle" else sbox(-fs / 2, -fs / 2, fs / 2, fs / 2)
    geom = affinity.scale(C.fit(geom, fs * 0.85, fs * 0.85), -1, 1, origin=(0, 0))  # o filme é espelhado
    film = C.ext(shape.difference(geom), 1.2)
    fm = C.to_tm(film); fm.apply_translation((R2 * 2 + 15, 0, 0))
    k = v["proj_dist"] / max(fd, 1)
    return [C.on_floor(C.to_tm(body)), fm], ["Suporte", "Filme"], [(0.3, 0.3, 0.3), (0.95, 0.95, 0.95)], \
        {"Imagem projetada": f"≈ {fs * 0.85 * k:.0f} mm a {v['proj_dist']:.0f} mm"}


PROJ = {"Fence": fence, "Pandora": pandora, "Squama": squama, "Torch": torch}


def run_projection(mesh, v, fast):
    r = PROJ[v["method"]](v, fast)
    extra = None
    if isinstance(r, tuple) and isinstance(r[-1], dict):
        extra = r[-1]; r = r[:-1] if len(r) > 2 else r[0]
    if isinstance(r, tuple):
        ms, names, cols = r
    else:
        ms, names, cols = r, None, None
    return C.result(ms, names, cols, extra)


# ==========================================================================
# Bottle
# ==========================================================================
def _section(v, W, D):
    p = v.get("h_section", "")
    if p:
        geom = C.mask_to_geom(C.photo_mask(C.load_gray(p, 300), True), 0.5)
        if not geom.is_empty:
            geom = max(getattr(geom, "geoms", [geom]), key=lambda q: q.area)
            return C.fit(Polygon(geom.exterior), W, D, keep=False)
    return affinity.scale(Point(0, 0).buffer(0.5, quad_segs=48), W, D)


def _profile(v, n):
    p = v.get("v_section", "")
    z = np.linspace(0, 1, n)
    if p:
        m = C.photo_mask(C.load_gray(p, 300), True)
        rows = np.where(m.any(1))[0]
        if len(rows) > 4:
            m = m[rows[0]:rows[-1] + 1]
            w = np.array([(np.where(r)[0].ptp() + 1) if r.any() else 0 for r in m], float)[::-1]
            w = ndimage.gaussian_filter1d(w, 1.5)
            return np.clip(np.interp(z, np.linspace(0, 1, len(w)), w / max(w.max(), 1)), 0.08, 1)
    # garrafa clássica: corpo, ombro e gargalo
    r = np.ones(n)
    sh = (z > 0.62) & (z <= 0.82)
    r[sh] = 0.32 + 0.68 * (0.5 + 0.5 * np.cos(np.pi * (z[sh] - 0.62) / 0.2))
    r[z > 0.82] = 0.32
    r[z < 0.04] = 0.9 + 0.1 * z[z < 0.04] / 0.04
    return r


def bottle(v, fast):
    W, D, Hh = v["sx"], v["sy"], v["sz"]
    t = v["thick"]
    n = 30 if fast else 80
    prof = _profile(v, n + 1)
    sec = _section(v, W, D)
    zs = np.linspace(0, Hh, n + 1)
    slabs, holes = [], []
    hollow = v.get("solid", "Hollow") == "Hollow"
    for k in range(n):
        r0, r1 = prof[k], prof[k + 1]
        dz = zs[k + 1] - zs[k]
        s0 = affinity.scale(sec, r0, r0, origin=(0, 0))
        slabs.append(mf.Manifold.extrude(C.cs(s0), dz + 0.001, 0, 0, (r1 / r0, r1 / r0)).translate((0, 0, zs[k])))
        if hollow and zs[k + 1] > t:
            si = s0.buffer(-t)
            if not si.is_empty and si.area > 1:
                si = max(getattr(si, "geoms", [si]), key=lambda q: q.area)
                ri = (r1 * 1.0) / r0
                z0 = max(zs[k], t)
                holes.append(mf.Manifold.extrude(C.cs(si), zs[k + 1] - z0 + (0.5 if k == n - 1 else 0.002), 0, 0, (ri, ri)).translate((0, 0, z0)))
    body = _M(slabs)
    neck = affinity.scale(sec, prof[-1], prof[-1], origin=(0, 0))
    capped = v.get("cap", "Yes") == "Yes"
    tol = v.get("tol", 0.0) + 0.2
    ct, chh = v["cap_t"], v["cap_h"]
    # rosca helicoidal no gargalo (lóbulo excêntrico torcido: uma volta por passo)
    nb = neck.bounds
    rn = min(nb[2] - nb[0], nb[3] - nb[1]) / 2
    pitch = max(2.0, min(4.0, rn * 0.5))
    dth = min(1.2, rn * 0.15)
    neck_len = Hh * 0.18 - 2
    th_h = min(chh - ct - 1.0, neck_len)
    thread = capped and th_h > pitch and rn > 3

    def lobe(extra, h):
        turns = h / pitch
        c = Point(dth, 0).buffer(rn - 0.02 + extra, quad_segs=24)
        return mf.Manifold.extrude(C.cs(c), h, int(turns * 36) + 2, 360 * turns)
    if thread:
        body = body + lobe(0, th_h).translate((0, 0, Hh - th_h - 1.0))
    if holes:
        body = body - _M(holes)
    out, names = [C.to_tm(body)], ["Garrafa"]
    if capped:
        R = rn + dth + tol
        outer = Point(0, 0).buffer(R + ct, quad_segs=48).union(neck.buffer(tol + ct))
        cap = C.ext(outer, chh) - C.ext(neck.buffer(tol), chh, ct)
        if thread:
            cap = cap - lobe(tol, chh - ct + 0.5).translate((0, 0, ct))
        # estrias exteriores para agarrar
        ribs = [C.ext(affinity.rotate(sbox(R + ct - 0.6, -0.6, R + ct + 0.6, 0.6), a, origin=(0, 0)), chh) for a in range(0, 360, 15)]
        cap = cap + _M(ribs)
        cm = C.to_tm(cap); cm.apply_translation((max(W, D) + 15, 0, 0))
        out.append(cm); names.append("Tampa")
    return C.result(out, names, None, {"Volume interior": f"{abs(_M(holes).volume()) / 1000:.0f} ml" if holes else "—"})


# ==========================================================================
# Bamboo-copter
# ==========================================================================
def bamboo(v, fast):
    n = max(int(v["blades"]), 1)
    ang = v["blade_angle"]; L = v["blade_len"]; rw = v["root_w"]; t = v["blade_t"]
    sd, sl = v["shaft_d"], v["shaft_len"]
    tol = v.get("tol", -0.1)
    hub_r = sd / 2 + 3
    tip_w = rw * 3.2
    # pá: raiz estreita junto ao cubo, ponta larga e arredondada
    pts = [(hub_r - 1, -rw / 2), (L * 0.25, -rw * 0.7), (L - tip_w / 2, -tip_w / 2), (L - tip_w / 2, tip_w / 2), (L * 0.25, rw * 0.7), (hub_r - 1, rw / 2)]
    blade2d = Polygon(pts).union(Point(L - tip_w / 2, 0).buffer(tip_w / 2, quad_segs=16))
    blade = mf.Manifold.extrude(C.cs(blade2d), t).translate((0, 0, -t / 2)).rotate((ang, 0, 0))
    hub_h = max(tip_w * math.sin(math.radians(ang)), rw) * 0.6 + t
    hub = mf.Manifold.cylinder(hub_h, hub_r, hub_r, 48).translate((0, 0, -hub_h / 2))
    rotor = _M([hub] + [blade.rotate((0, 0, 360 * k / n)) for k in range(n)])
    shaft = mf.Manifold.cylinder(sl, sd / 2, sd / 2, 32)
    if v.get("body", "One body") == "One body":
        M = rotor + shaft.translate((0, 0, -sl))
        return C.result([C.on_floor(C.to_tm(M))])
    rotor = rotor - mf.Manifold.cylinder(hub_h + 2, sd / 2 - tol, sd / 2 - tol, 32).translate((0, 0, -hub_h / 2 - 1))
    rm = C.on_floor(C.to_tm(rotor))
    sm = C.to_tm(shaft.rotate((0, 90, 0))); sm.apply_translation((-sl / 2, L + 15, sd / 2))
    return C.result([rm, C.on_floor(sm)], ["Hélice", "Haste"])


# ==========================================================================
# Dice
# ==========================================================================
PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)], 6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]}


def dice(v, fast):
    s = v["size"]
    cube = _rounded_box(s, s, s, v["corner_r"])
    ch = v.get("chamfer", 0.0)
    if ch > 0:
        # chanfro das arestas: interseção com um octaedro "arredondado"
        k = s / 2 * 3 - ch * 1.5
        oct_ = mf.Manifold.cube((s * 3, s * 3, s * 3), True)
        for sx_ in (-1, 1):
            for sy_ in (-1, 1):
                for sz_ in (-1, 1):
                    nrm = np.array([sx_, sy_, sz_]) / math.sqrt(3)
                    oct_ = oct_.trim_by_plane(tuple(-nrm), -(s / 2 * math.sqrt(3) - ch * 1.2))
        cube = cube ^ oct_
    e = v["extrusion"]
    d = abs(e)
    pr = s * 0.085
    tpl = v.get("template", "")
    tgeom = None
    if tpl:
        tg = C.mask_to_geom(C.photo_mask(C.load_gray(tpl, 300), True), 0.5)
        if not tg.is_empty:
            tgeom = C.fit(tg, s * 0.6, s * 0.6)
    # faces: +Z=1, -Z=6, +X=2, -X=5, +Y=3, -Y=4
    faces = {1: (0, 0, 0), 6: (180, 0, 0), 2: (0, 90, 0), 5: (0, -90, 0), 3: (-90, 0, 0), 4: (90, 0, 0)}
    marks = []
    for num, r in faces.items():
        if num == 1 and tgeom is not None:
            g = tgeom
        else:
            g = unary_union([Point(x * s * 0.25, y * s * 0.25).buffer(pr, quad_segs=16) for x, y in PIPS[num]])
        z0 = s / 2 - d if e < 0 else s / 2 - 0.01
        marks.append(C.ext(g, d + (0.02 if e < 0 else 0.01), z0).rotate(r))
    mk = _M(marks)
    M = cube - mk if e < 0 else cube + mk
    return C.result([C.on_floor(C.to_tm(M))])


# ==========================================================================
# Eyeglasses
# ==========================================================================
def _lens(v):
    es = v["eye"]; ar = max(v["aspect"], 0.2)
    h = es / ar if ar >= 1 else es * ar
    h = es / ar
    p = v.get("template", "")
    if p:
        g = C.mask_to_geom(C.photo_mask(C.load_gray(p, 300), True), 0.5)
        if not g.is_empty:
            g = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
            return C.fit(Polygon(g.exterior), es, h, keep=False)
    # forma "wayfarer" suave: superelipse ligeiramente mais larga em cima
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    n = 3.2
    x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / n)
    y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / n)
    x = x * (1 + 0.08 * y)
    return C.fit(Polygon(np.c_[x, y]), es, h, keep=False)


def eyeglasses(v, fast):
    es = v["eye"]; ft = v["frame_t"]
    bx, bz = v["bridge_x"], v["bridge_z"]
    lens = _lens(v)
    lh = lens.bounds[3] - lens.bounds[1]
    rim_w = max(ft, 2.5)
    off = bx / 2 + es / 2
    lensL = affinity.translate(lens, -off, 0); lensR = affinity.translate(affinity.scale(lens, -1, 1, origin=(0, 0)), off, 0)
    rims = unary_union([lensL.buffer(rim_w), lensR.buffer(rim_w)])
    # ponte (curva) e ponte dupla opcional
    bh = v["bridge_h"]; cur = v["bridge_curv"]
    xs = np.linspace(-off + es * 0.35, off - es * 0.35, 30)
    ys = lh * 0.15 + cur * (1 - (xs / max(xs[-1], 1e-6)) ** 2) + bh * 0.2
    bridge = LineString(np.c_[xs, ys]).buffer(bz / 2 + 0.6, cap_style="round")
    parts2d = [rims, bridge]
    if v.get("double_bridge", "No") == "Yes":
        ys2 = lh / 2 + rim_w * 0.4 + np.zeros_like(xs)
        parts2d.append(LineString(np.c_[np.linspace(-off, off, 30), ys2]).buffer(1.2))
    cs_ = v["corner"]
    for sgn in (-1, 1):
        cx = sgn * (off + es / 2 + rim_w / 2)
        parts2d.append(sbox(cx - cs_ / 2 * (sgn < 0) - (cs_ / 2) * (sgn > 0) * 0, lh * 0.05, cx + cs_ * sgn, lh * 0.05 + v["temple_h"]).buffer(0.5))
    front2d = unary_union(parts2d).difference(unary_union([lensL, lensR]))
    front = C.ext(front2d, ft)
    # plaquetas do nariz
    ns, nh = v["nose_size"], v["nose_h"]
    pads = []
    for sgn in (-1, 1):
        pad = mf.Manifold.cube((ns * 0.5, ns, nh), True).rotate((0, sgn * (20 + v.get("nose_tilt", 0.0)), 0))
        pads.append(pad.translate((sgn * (bx / 2 + 1.5), -lh * 0.12, ft + nh / 2 - 0.5)))
    front = _M([front] + pads)
    # furos dos parafusos nas dobradiças
    sdm = v["screw_d"]
    hinge_x = off + es / 2 + rim_w / 2 + cs_ * 0.7
    if v.get("frame_screw", "Yes") == "Yes":
        for sgn in (-1, 1):
            front = front - mf.Manifold.cylinder(v["temple_h"] + 10, sdm / 2 - v.get("tol", -0.2) * 0.5, sdm / 2, 16).translate(
                (sgn * hinge_x, lh * 0.05 - 2, ft / 2)).rotate((0, 0, 0))
    # inclinação pantoscópica e curvatura (wrap)
    front = front.rotate((-v.get("panto", 5.0), 0, 0))
    # hastes
    TL, tt, tsz = v["temple_len"], v["temple_t"], v["temple_size"]
    el, ea = v["ear_len"], v["ear_angle"]
    a = math.radians(ea)
    path = [(0, 0), (TL - el, 0), (TL - el + el * math.cos(a), -el * math.sin(a))]
    arm2d = LineString(path).buffer(tsz / 2, cap_style="round", join_style="round")
    arm = C.ext(arm2d, tt)
    if v.get("body", "One body") == "One body":
        # a frente fica de pé (plano XZ) e as hastes saem para trás (+Y) a partir das dobradiças
        zh = lh * 0.05 + v["temple_h"] / 2
        ang = v.get("temple_angle", 90.0)
        temples = []
        for sgn in (-1, 1):
            t3 = arm.rotate((90, 0, 0)).rotate((0, 0, 90))  # comprimento em +Y, altura em Z, espessura em X
            t3 = t3.translate((-tt / 2 * sgn if sgn > 0 else tt / 2, 0, 0)).rotate((0, 0, sgn * (ang - 90)))
            temples.append(t3.translate((sgn * hinge_x, 0, zh)))
        wrap = v.get("wrap", 0.0)
        fr = front.rotate((90, 0, 0))
        M = _M([fr] + temples)
        return C.result([C.on_floor(C.to_tm(M))], extra={"Largura total": f"{2 * hinge_x + 2 * tt:.0f} mm"})
    # montagem: frente com charneiras (2 nós) e hastes deitadas com o nó do meio; parafuso de diâmetro screw_d
    kr = max(sdm / 2 + 1.0, ft / 2)  # raio dos nós
    kh = v["temple_h"] / 3
    ytop = lh * 0.05 + v["temple_h"]
    knuckles, holes_ = [], []
    for sgn in (-1, 1):
        x = sgn * (hinge_x + kr)
        for z0 in (ytop - kh, ytop - 3 * kh):
            knuckles.append(mf.Manifold.cylinder(kh - 0.2, kr, kr, 24).rotate((-90, 0, 0)).translate((x, z0 + kh - 0.2, -kr + ft)))
        holes_.append(mf.Manifold.cylinder(v["temple_h"] + 4, sdm / 2, sdm / 2, 16).rotate((-90, 0, 0)).translate((x, ytop - 3 * kh - 2 + v["temple_h"] + 4, -kr + ft)))
    front = (front + _M(knuckles)) - _M(holes_)
    out = [C.to_tm(front)]
    for k, sgn in enumerate((-1, 1)):
        # nó do meio na ponta da haste (deitada: eixo do nó em Z)
        knot = mf.Manifold.cylinder(tt, kr, kr, 24).translate((-kr * 0.5, 0, 0)) - \
            mf.Manifold.cylinder(tt + 2, sdm / 2 - v.get("tol", -0.2) / 2, sdm / 2 - v.get("tol", -0.2) / 2, 16).translate((-kr * 0.5, 0, -1))
        am = C.to_tm(arm + knot)
        am.apply_translation((-TL / 2, -lh - 15 - k * (tsz + 25), 0))
        out.append(am)
    return C.result(C.group_floor(out), ["Frente", "Haste esquerda", "Haste direita"],
                    [(0.15, 0.15, 0.15), (0.5, 0.3, 0.15), (0.5, 0.3, 0.15)])


# ==========================================================================
# Glider
# ==========================================================================
def _logo(path, W, H):
    if not path:
        return None
    g = C.mask_to_geom(C.photo_mask(C.load_gray(path, 300), True), 0.5)
    return None if g.is_empty else C.fit(g, W, H)


def glider(v, fast):
    FL, fw, fh = v["fus_len"], v["fus_w"], v["fus_h"]
    ch = v["cockpit_h"]
    # fuselagem: perfil em pé (plano XZ), com nariz mais alto (cabine) e deriva na cauda
    prof = Polygon([(0, 0), (FL, 0), (FL, fh + 22), (FL - 18, fh + 2), (FL * 0.3, fh), (FL * 0.12, ch), (FL * 0.03, ch * 0.7), (0, ch * 0.3)])
    fus = C.ext(prof, fw, -fw / 2).rotate((90, 0, 0))
    # asa trapezoidal (ângulo do bordo de ataque)
    WL, WW, wt = v["wing_len"], v["wing_w"], v["wing_t"]
    sweep = WL / 2 / math.tan(math.radians(max(v["wing_angle"], 10)))
    xw = FL * v["wing_pos"] * 0.6
    wing2d = Polygon([(xw, -WL / 2 * 0 - fw / 2), (xw + sweep, -WL / 2), (xw + sweep + WW * 0.6, -WL / 2), (xw + WW, -fw / 2),
                      (xw + WW, fw / 2), (xw + sweep + WW * 0.6, WL / 2), (xw + sweep, WL / 2), (xw, fw / 2)]).buffer(0)
    wing = C.ext(wing2d, wt)
    lt = v["logo_t"]
    lg = _logo(v.get("wing_logo", ""), WW * 0.5, WL * 0.3)
    parts = [fus, wing.translate((0, 0, fh - wt))]
    if lg is not None:
        for sgn in (-1, 1):
            parts.append(C.ext(affinity.translate(affinity.rotate(lg, 90), xw + sweep * 0.6 + WW * 0.3, sgn * WL * 0.28), lt, fh - 0.01))
    SL, SW = v["stab_len"], v["stab_w"]
    stab2d = Polygon([(FL - SW, -SL / 2), (FL, -SL / 2 + 3), (FL, SL / 2 - 3), (FL - SW, SL / 2)])
    parts.append(C.ext(stab2d, wt, 0))
    tl = _logo(v.get("tail_logo", ""), SW * 0.6, SL * 0.3)
    if tl is not None:
        parts.append(C.ext(affinity.translate(tl, FL - SW / 2, 0), lt, wt - 0.01))
    M = _M(parts)
    return C.result([C.on_floor(C.to_tm(M))], extra={"Envergadura": f"{WL:.0f} mm"})
