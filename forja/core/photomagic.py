"""Photo magic: 25 maneiras de transformar uma fotografia (ou texto) num objeto 3D.
Balloon, Block, Box, Cell, Channel, Clicker, Cutter, Extrusion, Fabric, Fidget, Flexi, Halftone, Layering, Lenticule,
Moire, Morphing, Painting, Pendant, Pillar, Puzzle, Ridge, Teacup, Whistle, Wire, Wormhole."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from scipy import ndimage
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box as sbox
from shapely import affinity
from shapely.ops import unary_union
from . import cimg as C

METHODS = ["Balloon", "Block", "Box", "Cell", "Channel", "Clicker", "Cutter", "Extrusion", "Fabric", "Fidget", "Flexi",
           "Halftone", "Layering", "Lenticule", "Moire", "Morphing", "Painting", "Pendant", "Pillar", "Puzzle", "Ridge",
           "Teacup", "Whistle", "Wire", "Wormhole"]


# --------------------------------------------------------------------------
def _gray(v, rows=None, cols=None, key="photo"):
    g = C.load_gray(v.get(key, ""), 500)
    if rows:
        g = C.resample(g, rows, cols)
    return g


def _grid(W, H, step, fast, lim=420):
    if fast:
        step = max(step, max(W, H) / 110)
    return int(np.clip(H / step, 8, lim)), int(np.clip(W / step, 8, lim))


def _raster_geom(mask, W, H):
    """Máscara (linha 0 em cima) → geometria em mm centrada, com W × H."""
    g = C.mask_to_geom(mask, 0.5)
    if g.is_empty:
        return g
    r, c = mask.shape
    g = affinity.scale(g, W / c, H / r, origin=(0, 0))
    return affinity.translate(g, -W / 2, -H / 2)


def _sil(v, W, H):
    g = C.silhouette(v, W, H, positive=v.get("polarity", "Positive") in ("Positive", "Bright"))
    return g


def _colors(v, rows, cols):
    rgb = C.resample(C.load_rgb(v.get("photo", ""), 400), rows, cols)
    user = None
    if v.get("color", "Automatic") == "User input":
        user = [(1, 1, 1), (0.1, 0.1, 0.1), (0.85, 0.2, 0.2), (0.2, 0.4, 0.85), (0.95, 0.85, 0.2), (0.2, 0.7, 0.3)]
    lab, pal = C.quantize(rgb, int(v.get("color_layers", 4)), user is None, user)
    return lab, pal


def _outline(v, W, H, key="photo"):
    """Contorno exterior (cheio) da silhueta da fotografia, ajustado a W × H."""
    tp = v.get("template", "") if key == "template" else ""
    if tp:
        g = C.mask_to_geom(C.photo_mask(C.load_gray(tp, 300), True), 0.6)
    else:
        g = C.silhouette(v, W, H)
    polys = [Polygon(p.exterior) for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and not p.is_empty]
    if not polys:
        return sbox(-W / 2, -H / 2, W / 2, H / 2)
    o = unary_union(polys).buffer(1.0).buffer(-1.0)
    o = max(getattr(o, "geoms", [o]), key=lambda q: q.area)
    return C.fit(o.simplify(0.15), W, H)


def _min_enclosing(shape, geom):
    """Menor forma regular (Triangle … Circle) que contém a geometria."""
    hull = geom.convex_hull
    b = hull.bounds
    w, h = b[2] - b[0], b[3] - b[1]
    if shape == "Square":
        w = h = max(w, h)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    best = None
    for dy in np.linspace(-0.6, 0.6, 13) * h:
        lo, hi = 0.9, 3.0
        for _ in range(16):
            k = (lo + hi) / 2
            g = affinity.translate(C.regular(shape, w * k, h * k), cx, cy + dy)
            if g.buffer(1e-6).contains(hull):
                hi = k
            else:
                lo = k
        if best is None or hi < best[0]:
            best = (hi, dy)
    k, dy = best
    return affinity.translate(C.regular(shape, w * k, h * k), cx, cy + dy)


def _frame(v, inner_geom):
    ring, inner = C.frame_ring(v.get("frame", "No frame"), inner_geom, v.get("frame_w", 3.0), v.get("round", "Yes") == "Yes")
    return ring, inner


def _loop(top_y, d=5.0, w=2.5, t=3.0):
    ring = Point(0, 0).buffer(d / 2 + w, quad_segs=24).difference(Point(0, 0).buffer(d / 2, quad_segs=24))
    return affinity.translate(ring, 0, top_y + d / 2 + w - 1), t


# --------------------------------------------------------------------------
def balloon(v, fast):
    W, Hs = v["sx"], v["sz"]
    geom = _sil(v, W, Hs)
    b = geom.bounds
    Z = min(b[2] - b[0], b[3] - b[1]) * 0.65
    W2, H2 = b[2] - b[0], b[3] - b[1]
    rows, cols = _grid(W2, H2, max(v.get("res", 0.1), 0.25), fast, 300)
    from shapely import contains_xy
    xs = np.linspace(b[0], b[2], cols); ys = np.linspace(b[3], b[1], rows)
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(geom, X, Y)
    d = ndimage.distance_transform_edt(inside)
    dn = d / max(d.max(), 1e-6)
    h = (Z / 2) * np.clip(dn, 0, 1) ** (0.5 / max(v.get("expansion", 1.0), 0.1)) + 0.3
    Po = np.dstack([X, Y, h]); Pi = np.dstack([X, Y, -h])
    m = C.grid_solid(Po, Pi)
    M = C.to_mf(m) ^ C.ext(geom.buffer(0.01), Z + 10, -Z / 2 - 5)
    out = C.to_tm(M)
    C.rot(out, 90, [1, 0, 0])  # de pé: silhueta no plano XZ, insuflado em Y
    return [C.on_floor(out)]


def block(v, fast):
    W, H, Z = v["sx"], v["sy"], v["sz"]
    bs, gap = max(v["block"], 0.5), max(v["gap"], 0)
    nx, ny = max(int((W + gap) / (bs + gap)), 1), max(int((H + gap) / (bs + gap)), 1)
    g = _gray(v, ny, nx)
    parts = []
    hollow = v.get("solid_top", "No") == "No" and bs > 3
    for i in range(ny):
        for j in range(nx):
            h = max(Z * (1 - g[i, j]), 0.6)
            x = -W / 2 + j * (bs + gap); y = H / 2 - (i + 1) * (bs + gap) + gap
            sq = sbox(x, y, x + bs, y + bs)
            parts.append(C.ext(sq, h, 0))
    M = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    if v.get("solid_base", "No") == "Yes":
        M = M.translate((0, 0, 2)) + C.ext(sbox(-W / 2, H / 2 - ny * (bs + gap) + gap, -W / 2 + nx * (bs + gap) - gap, H / 2), 2.01)
    else:
        # grelha fina a ligar os blocos
        lines = []
        for i in range(ny):
            y = H / 2 - (i + 1) * (bs + gap) + gap + bs / 2
            lines.append(sbox(-W / 2, y - 0.6, -W / 2 + nx * (bs + gap) - gap, y + 0.6))
        for j in range(nx):
            x = -W / 2 + j * (bs + gap) + bs / 2
            lines.append(sbox(x - 0.6, H / 2 - ny * (bs + gap) + gap, x + 0.6, H / 2))
        M = M + C.ext(unary_union(lines), 0.6)
    if hollow:
        holes = [C.ext(sbox(-W / 2 + j * (bs + gap) + 1, H / 2 - (i + 1) * (bs + gap) + gap + 1,
                            -W / 2 + j * (bs + gap) + bs - 1, H / 2 - (i + 1) * (bs + gap) + gap + bs - 1), Z + 2, 1.2)
                 for i in range(ny) for j in range(nx)]
        M = M - mf.Manifold.batch_boolean(holes, mf.OpType.Add)
    return [C.on_floor(C.to_tm(M))]


def box(v, fast):
    t = v["thick"]
    W, D, Hh = v["bx"], v["by"], v["bz"]
    if v.get("size_mode", "Inner size") == "Inner size":
        W, D, Hh = W + 2 * t, D + 2 * t, Hh + t
    tol = v.get("tol", 0.0) + 0.2
    lz = v["lid_z"]
    # a caixa tem a forma da silhueta da fotografia
    outer = _outline(v, W, D)
    inner = outer.buffer(-t)
    if inner.is_empty or inner.area < outer.area * 0.15:
        r = min(3.0, t)
        outer = sbox(-W / 2 + r, -D / 2 + r, W / 2 - r, D / 2 - r).buffer(r, quad_segs=8)
        inner = outer.buffer(-t)
    inner = max(getattr(inner, "geoms", [inner]), key=lambda q: q.area)
    lid_type = v.get("lid_type", "Basic")
    body = C.ext(outer, Hh) - C.ext(inner, Hh + 1, t)
    # tampa
    lid = C.ext(outer, lz)
    lip_h = min(v.get("lid_indent", 10.0), Hh - t - 1)
    if lid_type == "Inlay":
        # tampa embutida: assenta num rebaixo dentro da caixa
        body = body - C.ext(outer.buffer(-t / 2), t, Hh - t)
        lid = C.ext(outer.buffer(-t / 2 - tol), min(lz, t))
    elif lid_type == "Flat":
        lid = C.ext(outer, min(lz, t)) + C.ext(inner.buffer(-tol), lip_h, min(lz, t) - 0.01).translate((0, 0, 0))
        lid = lid - C.ext(inner.buffer(-tol - t), lip_h + 2, min(lz, t) + 0.5)
    else:
        lid = lid - C.ext(inner, lz, t)
        lip = C.ext(inner.buffer(-tol), lip_h, lz - 0.01) - C.ext(inner.buffer(-tol - min(t, 2)), lip_h + 2, lz - 1)
        lid = lid + lip
        if lid_type == "Twist":
            # patilhas de baioneta
            ib = inner.bounds
            for sx_ in (-1, 1):
                lug = sbox(-2, -1.5, 2, 1.5)
                lid = lid + C.ext(affinity.translate(lug, sx_ * (ib[2] - tol - 1.5), 0), 2, lz + lip_h - 3)
    # padrão da fotografia na tampa
    pat = v.get("pattern", "Ignore")
    if pat != "Ignore":
        g = _gray(v)
        geom = C.fit(C.mask_to_geom(C.photo_mask(g, True), 0.6), (W - 2 * t) * 0.85, (D - 2 * t) * 0.85)
        if not geom.is_empty:
            top = 0.0  # a tampa é construída de cima para baixo (topo em z = 0)
            if pat == "Extrude":
                lid = lid + C.ext(geom, 1.2, -1.19)
            elif pat in ("Indent", "Intrude"):
                lid = lid - C.ext(geom, 1.0 if pat == "Indent" else t * 0.6, -0.01)
            elif pat == "Hollow lid":
                lid = lid - C.ext(geom, lz + 2, -1)
            elif pat == "Hollow box":
                body = body - C.ext(geom.intersection(inner.buffer(-1)), t + 2, -1)
    if v.get("chamfer", 0.0) > 0:
        pass  # chanfro das arestas: aproximado pelo arredondamento dos cantos
    lid_m = C.to_tm(lid)
    C.rot(lid_m, 180, [1, 0, 0]) if lid_type in ("Basic", "Twist") else None
    C.on_floor(lid_m)
    lid_m.apply_translation((W + 10, 0, 0))
    ms, names = [C.on_floor(C.to_tm(body)), lid_m], ["Caixa", "Tampa"]
    if v.get("nested", "No") == "Yes":
        k = (W - 2 * t - 2 * tol - 2) / W
        nb = C.ext(affinity.scale(outer, k, k), Hh * 0.8) - C.ext(affinity.scale(inner, k, k), Hh, t)
        nbm = C.to_tm(nb); nbm.apply_translation((0, D + 10, 0))
        ms.append(C.on_floor(nbm)); names.append("Caixa interior")
    return ms, names, None


def cell(v, fast):
    W, H = v["sx"], v["sy"]
    pitch = max(v.get("res", 0.1) * 10, 0.8)
    if fast:
        pitch = max(pitch, W / 45)
    nx, ny = int(W / pitch), int(H / pitch)
    g = _gray(v, ny, nx)
    bright = v.get("polarity", "Bright") == "Bright"
    holes = []
    for i in range(ny):
        for j in range(nx):
            val = g[i, j] if bright else 1 - g[i, j]
            s = pitch * 0.92 * math.sqrt(max(val, 0))
            if s < 0.35:
                continue
            c = (-W / 2 + (j + 0.5) * pitch, H / 2 - (i + 0.5) * pitch)
            holes.append(affinity.translate(C.regular(v.get("cell_shape", "Square"), s), *c))
    plate = sbox(-W / 2, -H / 2, W / 2, H / 2)
    geom = plate.difference(unary_union(holes)) if holes else plate
    bw = v.get("border", 1.0)
    geom = geom.union(plate.difference(plate.buffer(-bw, join_style="mitre")))
    return [C.ext_tm(geom, v["thick"])]


def channel(v, fast):
    W, H = v["sx"], v["sy"]
    geom = _sil(v, W, H)
    cw, ch, wall = v["ch_w"], v["ch_h"], v["wall"]
    # linha central do canal = contorno da silhueta
    path = unary_union([LineString(p.exterior.coords) for p in getattr(geom, "geoms", [geom])] +
                       [LineString(h.coords) for p in getattr(geom, "geoms", [geom]) for h in p.interiors])
    chan = path.buffer(cw / 2, quad_segs=6)
    walls = path.buffer(cw / 2 + wall, quad_segs=6).difference(chan)
    bt = v["base_t"]
    base_geom = geom.union(walls).buffer(0) if v.get("solid_base", "No") == "Yes" else walls.union(chan)
    body = C.ext(base_geom, bt) + C.ext(walls, ch, bt - 0.01)
    cap = C.to_tm(C.ext(chan.buffer(-max(v.get("tol", 0.0), 0) - 0.1).union(walls), v["cap_t"]))
    cap.apply_translation((0, 0, bt + ch + 2))
    return [C.on_floor(C.to_tm(body)), cap], ["Canal", "Tampa"], [(0.85, 0.85, 0.85), (0.95, 0.4, 0.6)]


def _color_layers(v, geom_clip, W, H, z0, lh, fast):
    """Camadas de cor (uma malha por cor) — mais escuro = mais camadas por cima (estilo HueForge)."""
    rows, cols = _grid(W, H, max(v.get("res", 0.1), 0.25), fast, 260)
    lab, pal = _colors(v, rows, cols)
    out, names, cols_ = [], [], []
    n = len(pal)
    for k in range(n):
        mask = lab >= k if v.get("layers_mode", "Multi-layer") == "Multi-layer" else lab == k
        gk = _raster_geom(mask, W, H)
        if geom_clip is not None:
            gk = gk.intersection(geom_clip)
        if gk.is_empty:
            continue
        zk = z0 + (k * lh if v.get("layers_mode", "Multi-layer") == "Multi-layer" else 0)
        out.append(C.ext_tm(gk, lh, zk)); names.append(f"Cor {k + 1}"); cols_.append(tuple(float(x) for x in pal[k]))
    return out, names, cols_


def clicker(v, fast):
    W, H = v["sx"], v["sy"]
    t = v["thick"]
    shape = _outline(v, W, H, "template" if v.get("template") else "photo")
    tol = v.get("tol", 0.0) + 0.25
    cx = v["cap_xy"]; cl = v["cap_len"]
    bx = v["base_xy"]; bl = v["base_len"]
    travel = max(v["pressed"] - cl - bl, 2.0)
    Hb = bl + cl + travel  # altura do corpo
    stem_shape = Point(0, 0).buffer(cx / 2, quad_segs=24) if v.get("shape", "Circle") == "Circle" else sbox(-cx / 2, -cx / 2, cx / 2, cx / 2)
    guide_o = stem_shape.buffer(tol + 1.6)
    # corpo: prisma oco com fundo, guia central para a haste
    inner = shape.buffer(-t)
    body = C.ext(shape, Hb) - (C.ext(inner, Hb, t) if not inner.is_empty else mf.Manifold())
    body = body + C.ext(guide_o, Hb - 1) - C.ext(stem_shape.buffer(tol), Hb + 2, bl)
    # botão: tampa com a forma da silhueta (um pouco menor) + haste
    cap2d = shape.buffer(-t - tol) if not shape.buffer(-t - tol).is_empty else shape.buffer(-0.5)
    top = C.ext(cap2d, t, Hb + travel)
    stem = C.ext(stem_shape, cl + travel + 1, Hb - cl)
    button = top + stem
    lay, names, cols = _color_layers(v, cap2d.buffer(-0.4), W, H, Hb + travel + t, 0.4, fast)
    ms = [C.to_tm(body), C.to_tm(button)] + lay
    return C.group_floor(ms), ["Corpo", "Botão"] + names, [(0.2, 0.75, 0.8), (0.85, 0.85, 0.85)] + cols


def cutter(v, fast):
    W, H = v["inner_x"], v["inner_y"]
    geom = _sil(v, W, H)
    if v.get("flip", "Yes") == "Yes":
        geom = affinity.scale(geom, -1, 1, origin=(0, 0))
    outline = unary_union([Polygon(p.exterior) for p in getattr(geom, "geoms", [geom])]).buffer(0.5).buffer(-0.5)
    outline = outline.simplify(0.2)

    def ring(w):
        return outline.buffer(w / 2, quad_segs=6).difference(outline.buffer(-w / 2, quad_segs=6))
    bw, bh = v["base_w"], v["base_h"]
    mw, mh = v["mid_w"], v["mid_h"]
    tw, th = v["blade_w"], v["blade_h"]
    parts = [C.ext(outline.buffer(bw / 2).difference(outline.buffer(-mw / 2 - (0 if v.get("base_hole", "Yes") == "Yes" else 1e3))), bh),
             C.ext(ring(mw), mh, bh - 0.01),
             C.ext(ring(tw), th, bh + mh - 0.02)]
    if v.get("stopper", "Yes") == "Yes":
        b = outline.bounds
        parts.append(C.ext(sbox(b[0], -0.8, b[2], 0.8).intersection(outline.buffer(-0.1)), 1.6))
    M = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    out, names = [C.to_tm(M)], ["Cortador"]
    pat = v.get("inner", "Ignore")
    if pat != "Ignore":
        inner_lines = [h for p in getattr(geom, "geoms", [geom]) for h in p.interiors]
        det = unary_union([LineString(h.coords).buffer(0.6) for h in inner_lines]) if inner_lines else geom.buffer(-1.5)
        det = det.intersection(outline.buffer(-v.get("inner_clear", 5.0) * 0.2))
        if pat == "Integrate":
            out[0] = C.to_tm(M + C.ext(det, v.get("inner_depth", 7.0), bh - 0.01))
        else:
            st = C.ext(outline.buffer(-mw / 2 - 0.5), 3) + C.ext(det, v.get("inner_depth", 7.0), 2.99)
            sm = C.to_tm(st); sm.apply_translation((W + 15, 0, 0))
            out.append(sm); names.append("Carimbo")
    return [C.on_floor(m) for m in out], names, None


def extrusion(v, fast):
    """A silhueta deitada, elevada sobre um suporte com a mesma forma e assente numa base."""
    W, H = v["sx"], v["sy"]
    geom = _sil(v, W, H)
    e = v.get("expansion", 1.0)
    rc = v.get("round", "No") == "Yes"
    if e:
        geom = geom.buffer(e * 0.5, join_style="round" if rc else "mitre")
    ex, elev, bt = v["extrusion"], v["elevation"], v["base_t"]
    parts = [C.ext(geom, ex, bt + elev - 0.01)]
    # suporte (elevação): a mesma silhueta, estreitada conforme o perfil
    if elev > 0:
        n = 1 if v.get("support", "Concave Arc") == "Vertical" else 12
        poly = [p for p in getattr(geom, "geoms", [geom])]
        inr = max(min((p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1])) for p in poly) / 2
        k = min(elev, inr) * 0.45
        for i in range(n):
            f = (i + 0.5) / n  # 0 = em baixo, 1 = em cima
            sup = v.get("support", "Concave Arc")
            if sup == "Vertical":
                d = 0.0
            elif sup == "Triangle":
                d = k * f
            elif sup == "Convex Arc":
                d = k * (1 - math.sqrt(max(1 - f * f, 0)))
            else:
                d = k * math.sqrt(max(1 - (1 - f) ** 2, 0))
            d = k - d  # estreito em baixo, a abrir até à forma em cima
            g = geom.buffer(-d - v.get("clearance", 1.0) * 0) if d > 0 else geom
            if not g.is_empty:
                parts.append(C.ext(g, elev / n + 0.02, bt + elev * i / n - 0.01))
    if v.get("elev_angle", 0.0):
        pass  # o ângulo de elevação inclina a peça sobre a base (aplicado abaixo)
    # base com a forma escolhida, a envolver a silhueta
    shape = v.get("base_shape", "Triangle")
    b = geom.bounds
    w, h = b[2] - b[0], b[3] - b[1]
    cxy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    margin = v.get("clearance", 1.0) + 3
    if shape == "Original":
        bgeom = unary_union([Polygon(p.exterior) for p in getattr(geom, "geoms", [geom])]).buffer(margin)
    elif shape == "Photo size":
        bgeom = sbox(-W / 2, -H / 2, W / 2, H / 2)
    else:
        bgeom = _min_enclosing(shape, geom.buffer(margin))
    M = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    a = v.get("elev_angle", 0.0)
    if a:
        M = M.rotate((a, 0, 0))
        lo = M.bounding_box()[2]
        M = M.translate((0, 0, bt - lo - 0.01))
    if bt > 0:
        M = M + C.ext(bgeom, bt)
    kh = v.get("keychain", 0.0)
    if kh > 0:
        bb = bgeom.bounds
        ring = Point(0, 0).buffer(kh / 2 + 2.5).difference(Point(0, 0).buffer(kh / 2))
        M = M + C.ext(affinity.translate(ring, (bb[0] + bb[2]) / 2, bb[3] + kh / 2 + 1), max(bt, 2))
    return [C.on_floor(C.to_tm(M))]


def fabric(v, fast):
    W, H = v["sx"], v["sy"]
    cs = max(v["cell"], 2.0)
    if fast:
        cs = max(cs, W / 30)
    nx, ny = int(W / cs), int(H / cs)
    lab, pal = _colors(v, ny, nx)
    gap = 0.8
    out, names, cols = [], [], []
    for k in range(len(pal)):
        tiles = [sbox(-W / 2 + j * cs + gap / 2, H / 2 - (i + 1) * cs + gap / 2, -W / 2 + (j + 1) * cs - gap / 2, H / 2 - i * cs - gap / 2)
                 for i in range(ny) for j in range(nx) if lab[i, j] == k]
        if tiles:
            out.append(C.ext_tm(unary_union(tiles), 1.6, 0.4)); names.append(f"Cor {k + 1}"); cols.append(tuple(float(x) for x in pal[k]))
    # ligações flexíveis por baixo (rede fina)
    links = [sbox(-W / 2, H / 2 - (i + 0.5) * cs - 0.4, -W / 2 + nx * cs, H / 2 - (i + 0.5) * cs + 0.4) for i in range(ny)] + \
            [sbox(-W / 2 + (j + 0.5) * cs - 0.4, H / 2 - ny * cs, -W / 2 + (j + 0.5) * cs + 0.4, H / 2) for j in range(nx)]
    out.insert(0, C.ext_tm(unary_union(links), 0.41)); names.insert(0, "Rede"); cols.insert(0, (0.3, 0.3, 0.3))
    return out, names, cols


def fidget(v, fast):
    W, H, Z = v["sx"], v["sy"], v["sz"]
    geom = _sil(v, W, H)
    outer = unary_union([Polygon(p.exterior) for p in getattr(geom, "geoms", [geom])])
    hole = v["hole"]
    tol = v.get("tol", 0.0) + 0.3
    c = outer.representative_point() if not outer.contains(outer.centroid) else outer.centroid
    body = C.ext(outer.union(Point(c.x, c.y).buffer(hole / 2 + 3)), Z) - mf.Manifold.cylinder(Z + 2, hole / 2, hole / 2, 64).translate((c.x, c.y, -1))
    # rolamento impresso no lugar: eixo com abas (ranhura = slot thickness)
    st = v["slot"]
    axle = mf.Manifold.cylinder(Z, hole / 2 - tol, hole / 2 - tol, 64)
    caps = mf.Manifold.cylinder(st, hole / 2 + 2, hole / 2 + 2, 64)
    rotor = axle.translate((0, 0, 0)) + caps.translate((0, 0, -st)) + caps.translate((0, 0, Z))
    body = body.translate((0, 0, st + tol))
    rotor = rotor.translate((c.x, c.y, st))
    return C.group_floor([C.to_tm(body), C.to_tm(rotor)]), ["Corpo", "Eixo"], None


def flexi(v, fast):
    W, H = v["sx"], v["sy"]
    geom = _sil(v, W, H)
    t = v["thick"]
    n = max(int(v.get("section", 5)), 1)
    tol = v.get("tol", 0.0) + 0.5
    b = geom.bounds
    off = v.get("plane_off", 0.0) / 100 * (b[2] - b[0]) / n
    cuts = [b[0] + (b[2] - b[0]) * k / n + off for k in range(1, n)]
    solid = geom
    for x in cuts:
        solid = solid.difference(sbox(x - tol / 2, b[1] - 1, x + tol / 2, b[3] + 1))
    M = C.ext(solid, t)
    # dobradiças finas (0,4 mm) por baixo a ligar as secções
    hinge = unary_union([sbox(x - tol, b[1] - 1, x + tol, b[3] + 1) for x in cuts]).intersection(geom) if cuts else Polygon()
    if not hinge.is_empty:
        M = M + C.ext(hinge, 0.4)
    return [C.to_tm(M)]


def halftone(v, fast):
    W, H = v["sx"], v["sy"]
    step = 0.25 if not fast else max(W / 220, 0.25)
    rows, cols = int(H / step), int(W / step)
    g = _gray(v, rows, cols)
    k = max(v.get("contrast", 5.0), 0.1) / 5.0
    dark = np.clip((1 - g), 0, 1) ** (1 / k)
    if v.get("polarity", "Bright") == "Bright":
        dark = 1 - dark
    yy, xx = np.mgrid[0:rows, 0:cols] * step
    xx = xx - W / 2; yy = H / 2 - yy
    sw = max(v.get("stripe_w", 5.0), 0.5)
    a = math.radians(v.get("stripe_angle", 0.0))
    pat = v.get("pattern", "Stripe")
    if pat == "Stripe":
        ph = ((xx * math.sin(a) + yy * math.cos(a)) / sw) % 1.0
        mask = np.abs(ph - 0.5) * 2 < dark * 0.95
    elif pat == "Radial":
        th = np.arctan2(yy, xx) + a
        N = max(int(math.pi * min(W, H) / sw / 2), 8)
        ph = (th / (2 * math.pi) * N) % 1.0
        mask = np.abs(ph - 0.5) * 2 < dark * 0.95
    elif pat == "Spiral":
        r = np.hypot(xx, yy); th = np.arctan2(yy, xx) + a
        ph = (r / sw - th / (2 * math.pi)) % 1.0
        mask = np.abs(ph - 0.5) * 2 < dark * 0.95
    else:  # Stencil
        mask = dark > 0.5
    geom = _raster_geom(mask, W, H)
    plate = sbox(-W / 2, -H / 2, W / 2, H / 2)
    ring, inner = _frame(v, plate)
    clip = inner if not ring.is_empty else plate
    geom = geom.intersection(clip)
    bt = v.get("base_t", 1.0)
    parts = [C.ext(geom, v["thick"], bt - 0.01 if bt > 0 else 0)]
    if bt > 0:
        parts.append(C.ext(clip, bt))
    if not ring.is_empty:
        parts.append(C.ext(ring, bt + v["thick"]))
    return [C.to_tm(mf.Manifold.batch_boolean(parts, mf.OpType.Add))]


def layering(v, fast):
    """Uma fotografia por camada (lista "Layer order", a 1.ª fica à frente). Cada camada é uma placa
    com a silhueta da sua fotografia recortada (ou só a silhueta, com "Recortar"), afastadas por 'layer_off'.
    Sem lista: uma só fotografia dividida em 'layers' níveis de tom."""
    photos = [p for p in (v.get("photos") or []) if p]
    if photos:
        return _layering_photos(v, photos, fast)
    W, H = v["sx"], v["sy"]
    n = max(int(v.get("layers", 4)), 1)
    rows, cols = _grid(W, H, 0.4, fast, 300)
    g = _gray(v, rows, cols)
    g = ndimage.gaussian_filter(g, 1)
    lh, off = v["layer_h"], v["layer_off"]
    plate = sbox(-W / 2, -H / 2, W / 2, H / 2)
    ring, inner = _frame(v, plate)
    clip = inner if not ring.is_empty else plate
    out, names, cols_ = [], [], []
    qs = np.quantile(g, np.linspace(0, 1, n + 1)[1:-1]) if n > 1 else []
    for k in range(n):
        # camada k: material onde a imagem é mais escura do que o limiar (camadas de trás são cheias)
        thr = qs[n - 2 - k] if k < n - 1 else -1
        if k == n - 1:
            geom = clip
        else:
            geom = _raster_geom(g > thr, W, H)
            geom = clip.difference(geom.buffer(v.get("expansion", 0.0) or 0.01)) if v.get("clipping", "No") == "No" else geom
            geom = geom.intersection(clip)
            if v.get("solid_base", "No") == "Yes":
                geom = geom.union(clip.difference(clip.buffer(-2)))
        z = (n - 1 - k) * (lh + off)
        if not geom.is_empty:
            out.append(C.ext_tm(geom, lh, z)); names.append(f"Camada {k + 1}")
            sh = 0.25 + 0.7 * k / max(n - 1, 1)
            cols_.append((sh, sh, sh * 1.05))
    if not ring.is_empty:
        out.append(C.ext_tm(ring, n * (lh + off))); names.append("Moldura"); cols_.append((0.4, 0.3, 0.2))
    return C.group_floor(out), names, cols_


def _layering_photos(v, photos, fast):
    W, H = v["sx"], v["sy"]
    lh, off = v["layer_h"], v["layer_off"]
    plate = sbox(-W / 2, -H / 2, W / 2, H / 2)
    ring, inner = _frame(v, plate)
    clip = inner if not ring.is_empty else plate
    n = len(photos)
    out, names, cols_ = [], [], []
    for k, ph in enumerate(photos):
        g = C.load_gray(ph, 400)
        sil = C.fit(C.mask_to_geom(C.photo_mask(g, True), 0.5), W, H)
        e = v.get("expansion", 0.0)
        if e:
            sil = sil.buffer(e)
        geom = sil.intersection(clip) if v.get("clipping", "No") == "Yes" else clip.difference(sil)
        if v.get("solid_base", "No") == "Yes" and k == n - 1:
            geom = clip
        z = (n - 1 - k) * (lh + off)
        if not geom.is_empty:
            import os
            out.append(C.ext_tm(geom, lh, z)); names.append(f"Camada {k + 1} ({os.path.basename(ph)})")
            sh = 0.25 + 0.7 * k / max(n - 1, 1)
            cols_.append((sh, sh, sh * 1.05))
    if not ring.is_empty:
        out.append(C.ext_tm(ring, n * (lh + off))); names.append("Moldura"); cols_.append((0.4, 0.3, 0.2))
    return C.group_floor(out), names, cols_


def lenticule(v, fast):
    """Moldura de pé com lâminas verticais; a profundidade de cada lâmina ao longo da altura segue a
    fotografia, de modo que a imagem aparece quando se olha de lado. "Reinforced bar" = número de barras horizontais."""
    W, Z = v["sx"], v["sz"]
    t = v["thick"]
    bs = v.get("border", 3.0)
    pitch = max(t * 0.9, 1.5) if not fast else max(W / 50, t * 0.9)
    n = max(int(W / pitch), 2)
    rows = int(np.clip(Z / max(v.get("res", 0.1) * 5, 0.5), 10, 300 if not fast else 60))
    g = _gray(v, rows, n)
    val = g if v.get("polarity", "Bright") == "Bright" else 1 - g
    sw = max(pitch * 0.3, 0.6)  # espessura da lâmina
    parts = []
    zs = np.linspace(Z, 0, rows)
    for j in range(n):
        x = -W / 2 + (j + 0.5) * pitch
        d = 0.6 + val[:, j] * (t * 3)  # profundidade da lâmina (Y) ao longo de Z
        prof = Polygon(np.r_[np.c_[d, zs], [[0, 0], [0, Z]]]).buffer(0)
        if prof.is_empty:
            continue
        # perfil (profundidade, altura) extrudido na espessura da lâmina → X = posição, Y = profundidade, Z = altura
        parts.append(C.ext(prof, sw, x - sw / 2).transform([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]))
    M = mf.Manifold.batch_boolean(parts, mf.OpType.Add) if parts else mf.Manifold()
    frame = sbox(-W / 2 - bs, -bs, W / 2 + bs, Z + bs).difference(sbox(-W / 2, 0, W / 2, Z))
    nb = max(int(v.get("bar", 5.0)), 0)
    bars = [sbox(-W / 2, Z * (k + 1) / (nb + 1) - 0.6, W / 2, Z * (k + 1) / (nb + 1) + 0.6) for k in range(nb)]
    fr = C.ext(unary_union([frame] + bars), t, -t).rotate((90, 0, 0)).translate((0, 0, 0))
    M = M + fr
    return [C.on_floor(C.to_tm(M))]


def moire(v, fast):
    W, H = v["sx"], v["sy"]
    s = max(v["slit"], 0.5)
    P = 2 * s
    step = min(s / 6, 0.4) if not fast else max(W / 250, s / 4)
    rows, cols = int(H / step), int(W / step)
    g = _gray(v, rows, cols)
    xx = np.arange(cols) * step
    shift = np.where(g < 0.5, P / 2, 0.0)
    mask = ((xx[None, :] + shift) % P) < s
    lines = _raster_geom(mask, W, H)
    t = v["thick"]
    base = C.ext(sbox(-W / 2, -H / 2, W / 2, H / 2), 1.2) + C.ext(lines, t, 1.19)
    tol = v.get("tol", 0.0)
    xs = np.arange(-W / 2, W / 2, P)
    screen = sbox(-W / 2 - 3, -H / 2 - 3, W / 2 + 3, H / 2 + 3).difference(
        unary_union([sbox(x + tol, -H / 2, x + s - tol, H / 2) for x in xs]))
    sm = C.ext_tm(screen, 1.0); sm.apply_translation((0, H + 10, 0))
    return [C.to_tm(base), sm], ["Base", "Grelha (deslizar por cima)"], [(0.15, 0.15, 0.15), (0.9, 0.9, 0.9)]


def morphing(v, fast):
    W, H, Z = v["sx"], v["sy"], v["sz"]
    if v.get("source", "Photo") == "Text":
        ts = [v.get("text1") or "A", v.get("text2") or "B", v.get("text3") or ""]
        geoms = [C.text_geom(t, v.get("font", "Arial"), 20) if t.strip() else None for t in ts]
    else:
        # 2 ou 3 fotografias (frente, lado, cima); sem a 2.ª usa a mesma
        def sil(key):
            return C.mask_to_geom(C.photo_mask(_gray(v, key=key), True), 0.6)
        g1 = sil("photo")
        g2 = sil("photo2") if v.get("photo2") else g1
        g3 = sil("photo3") if v.get("photo3") else None
        geoms = [g1, g2, g3]
    big = max(W, H, Z) * 2
    M = mf.Manifold.cube((W, H, Z), True)
    # vista de frente (XZ), de lado (YZ) e de cima (XY)
    if geoms[0] is not None:
        g1 = C.fit(geoms[0], W, Z, keep=False)
        M = M ^ C.ext(g1, big, -big / 2).rotate((90, 0, 0))
    if geoms[1] is not None:
        g2 = C.fit(geoms[1], H, Z, keep=False)
        M = M ^ C.ext(g2, big, -big / 2).rotate((90, 0, 90))
    if geoms[2] is not None:
        g3 = C.fit(geoms[2], W, H, keep=False)
        M = M ^ C.ext(g3, big, -big / 2)
    if v.get("polarity", "Positive") == "Negative":
        M = mf.Manifold.cube((W, H, Z), True) - M
    if v.get("grid", "No") == "Yes":
        bt = v.get("bar", 2.0)
        bars = []
        for i in range(int(v.get("x_sec", 5)) + 1):
            x = -W / 2 + W * i / max(int(v.get("x_sec", 5)), 1)
            bars.append(mf.Manifold.cube((bt, H + 2, Z + 2), True).translate((x, 0, 0)))
        for i in range(int(v.get("y_sec", 5)) + 1):
            y = -H / 2 + H * i / max(int(v.get("y_sec", 5)), 1)
            bars.append(mf.Manifold.cube((W + 2, bt, Z + 2), True).translate((0, y, 0)))
        for i in range(int(v.get("z_sec", 5)) + 1):
            z = -Z / 2 + Z * i / max(int(v.get("z_sec", 5)), 1)
            bars.append(mf.Manifold.cube((W + 2, H + 2, bt), True).translate((0, 0, z)))
        M = M ^ mf.Manifold.batch_boolean(bars, mf.OpType.Add)
    return [C.on_floor(C.to_tm(M))]


def painting(v, fast):
    W, H = v["sx"], v["sy"]
    lh = v["layer_h"]
    clip = sbox(-W / 2, -H / 2, W / 2, H / 2)
    tpl = v.get("template", "")
    if tpl:
        gt = C.mask_to_geom(C.photo_mask(C.load_gray(tpl, 300), True), 0.6)
        if not gt.is_empty:
            clip = C.fit(gt, W, H)
    base_n = int(v.get("base_color", 0))
    z0 = lh * (1 + base_n)
    out, names, cols = _color_layers(dict(v, layers_mode=v.get("layers_mode", "One layer") if v.get("layers_mode") else "One layer"),
                                     clip, W, H, z0, lh, fast)
    base = C.ext_tm(clip, z0)
    ms = [base] + out
    nms = ["Base"] + names
    cc = [(0.95, 0.95, 0.95)] + cols
    bs = v.get("border", 2.0)
    bt = v.get("box_t", 3.0)
    if bs > 0 or bt > 0:
        rim = clip.buffer(max(bs, 0.01) + bt).difference(clip.buffer(-0.01))
        hh = z0 + lh * len(out) + v.get("box_inner_h", 0.0)
        ms.append(C.ext_tm(rim, hh)); nms.append("Moldura"); cc.append((0.35, 0.25, 0.2))
    if v.get("sided", "Single-sided") == "Double-sided":
        back = []
        for m in ms[1:len(out) + 1]:
            b = m.copy()
            if v.get("mirror", "Yes") == "Yes":
                b.apply_transform(trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1])); b.invert()
            else:
                b.apply_translation((0, 0, -b.bounds[1][2] - b.bounds[0][2]))
            back.append(b)
        ms += back; nms += [n + " (verso)" for n in names]; cc += cols
    return C.group_floor(ms), nms, cc


def pendant(v, fast):
    W, H = v["sx"], v["sy"]
    geom = _sil(v, W, H)
    ring, inner = _frame(v, geom)
    g = geom.union(ring) if not ring.is_empty else geom
    if not ring.is_empty:
        # liga a silhueta à moldura
        b = inner.bounds
        g = g.union(sbox((b[0] + b[2]) / 2 - 1, b[1], (b[0] + b[2]) / 2 + 1, b[3]).intersection(inner).difference(Polygon()))
    t = v["thick"]
    lp, _ = _loop(g.bounds[3])
    M = C.ext(unary_union([g, lp]), t)
    return [C.to_tm(M)]


def pillar(v, fast):
    """4 pilares numerados entre uma base e um topo; o recorte das arestas de cada pilar segue uma
    faixa da fotografia (a imagem aparece ao juntar os pilares pela ordem dos números)."""
    wx = v["width_x"]; Hz = v["height_z"]; t = v["thick"]
    n = 4
    lod = int(np.clip(v.get("lod", 100), 10, 400 if not fast else 60))
    g = _gray(v, lod, n * 2)
    dark = 1 - g if v.get("polarity", "Bright") == "Bright" else g
    cap, base = v["cap"], v["base"]
    pw = wx * 2.2  # largura do núcleo do pilar
    gap = wx * 2.6
    zs = np.linspace(base + Hz, base, lod)
    pat = v.get("pattern", "Line")
    rng = np.random.default_rng(1)
    parts = []
    f = (zs - base) / Hz
    for j in range(n):
        x0 = j * (pw + gap)
        if pat == "Sine":
            mod = 0.5 + 0.5 * np.sin(f * 2 * math.pi * 3)
        elif pat == "Sawtooth":
            mod = (f * 6) % 1
        elif pat == "Ellipse":
            mod = np.sqrt(np.clip(1 - (2 * f - 1) ** 2, 0, 1))
        elif pat == "Semiellipse":
            mod = np.sqrt(np.clip(1 - f ** 2, 0, 1))
        elif pat == "Random":
            mod = np.clip(ndimage.gaussian_filter1d(rng.random(lod), 2), 0, 1)
        else:
            mod = np.ones(lod)
        L = x0 - dark[:, 2 * j] * wx * (0.4 + 0.6 * mod)
        R = x0 + pw + dark[:, 2 * j + 1] * wx * (0.4 + 0.6 * mod)
        # degraus (estilo "nível de detalhe")
        prof = Polygon(np.r_[np.c_[np.repeat(R, 2)[1:], np.repeat(zs, 2)[:-1]],
                             [[x0 + pw, base - 0.1], [x0, base - 0.1]],
                             np.c_[np.repeat(L, 2)[1:], np.repeat(zs, 2)[:-1]][::-1]]).buffer(0)
        P = C.ext(prof, t, -t / 2).rotate((90, 0, 0))
        try:
            num = C.text_geom(str(j + 1), "Arial", v.get("text_size", 5.0))
            nb = num.bounds
            num = affinity.translate(num, x0 + pw / 2 - (nb[0] + nb[2]) / 2, base + Hz * 0.45)
            P = P - C.ext(num, 1.2, -0.6).rotate((90, 0, 0)).translate((0, -t / 2, 0))
        except Exception:
            pass
        parts.append(P)
    total = (n - 1) * (pw + gap) + pw
    x_lo = -wx * 1.2
    if base > 0:
        parts.append(mf.Manifold.cube((total + 2.4 * wx, t + 4, base), False).translate((x_lo, -t / 2 - 2, 0)))
    if cap > 0:
        top = mf.Manifold.cube((total + 2.4 * wx, t + 4, cap), False).translate((x_lo, -t / 2 - 2, base + Hz - 0.01))
        try:
            tg = C.text_geom("Cap", "Arial", v.get("text_size", 5.0))
            tb = tg.bounds
            tg = affinity.translate(tg, total / 2 - (tb[0] + tb[2]) / 2, -(tb[1] + tb[3]) / 2)
            top = top - C.ext(tg, 1.0, base + Hz + cap - 0.99)
        except Exception:
            pass
        parts.append(top)
    M = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    return [C.on_floor(C.to_tm(M))]


def _jigsaw(nx, ny, W, H, tol):
    """Peças de puzzle (com patilhas) a cobrir W × H, centradas na origem."""
    pw, ph = W / nx, H / ny
    r = min(pw, ph) * 0.18
    rng = np.random.default_rng(7)
    hs = rng.choice([-1, 1], (ny + 1, nx)); vs = rng.choice([-1, 1], (ny, nx + 1))
    pieces = []
    for i in range(ny):
        for j in range(nx):
            x0, y0 = -W / 2 + j * pw, -H / 2 + i * ph
            p = sbox(x0, y0, x0 + pw, y0 + ph)
            # patilhas: em cada aresta interior acrescenta ou retira um círculo
            if j < nx - 1:
                c = Point(x0 + pw + vs[i, j + 1] * r * 0.8, y0 + ph / 2).buffer(r)
                p = p.union(c) if vs[i, j + 1] > 0 else p.difference(c)
            if j > 0:
                c = Point(x0 + vs[i, j] * r * 0.8, y0 + ph / 2).buffer(r)
                p = p.difference(c) if vs[i, j] > 0 else p.union(c)
            if i < ny - 1:
                c = Point(x0 + pw / 2, y0 + ph + hs[i + 1, j] * r * 0.8).buffer(r)
                p = p.union(c) if hs[i + 1, j] > 0 else p.difference(c)
            if i > 0:
                c = Point(x0 + pw / 2, y0 + hs[i, j] * r * 0.8).buffer(r)
                p = p.difference(c) if hs[i, j] > 0 else p.union(c)
            pieces.append(p.buffer(-tol / 2 - 0.1))
    return pieces


def puzzle(v, fast):
    W, H = v["sx"], v["sy"]
    nx, ny = max(int(v["x_sec"]), 1), max(int(v["y_sec"]), 1)
    pieces = _jigsaw(nx, ny, W, H, v.get("tol", 0.0))
    bt = v["base_t"]
    ms, names, cols = [], [], []
    if v.get("mode", "Color layer") == "Height map":
        rows, cc = _grid(W, H, max(v.get("res", 0.1), 0.25), fast, 300)
        T = C.litho_T(_gray(v, rows, cc), 0.0, 2.0)
        relief = C.to_mf(C.plate_relief(T, W, H, base=bt))
        for k, p in enumerate(pieces):
            ms.append(C.to_tm(relief ^ C.ext(p, bt + 5, -1))); names.append(f"Peça {k + 1}")
        return ms, names, None
    clipall = unary_union(pieces)
    lay, lnames, lcols = _color_layers(v, clipall, W, H, bt, v.get("layer_h", 0.4), fast)
    for k, p in enumerate(pieces):
        ms.append(C.ext_tm(p, bt)); names.append(f"Peça {k + 1}"); cols.append((0.95, 0.95, 0.95))
    return ms + lay, names + lnames, cols + lcols


def ridge(v, fast):
    """Paredes verticais na silhueta e uma superfície em "telhado invertido" cuja altura desce com a
    distância à borda (cristas onde as encostas se encontram)."""
    W, H = v["sx"], v["sy"]
    geom = _sil(v, W, H)
    b = geom.bounds
    W2, H2 = b[2] - b[0], b[3] - b[1]
    rows, cols = _grid(W2, H2, 0.4, fast, 300)
    from shapely import contains_xy
    xs = np.linspace(b[0], b[2], cols); ys = np.linspace(b[3], b[1], rows)
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(geom, X, Y)
    d = ndimage.distance_transform_edt(inside) * (W2 / max(cols - 1, 1))
    Hm = 0.22 * max(W2, H2)
    h = np.clip(Hm - d * v.get("ridge_scale", 1.0), 0.6, Hm)
    m = C.plate_relief(h, W2, H2, base=v["base_t"])
    m.apply_translation(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0))
    M = C.to_mf(m) ^ C.ext(geom.buffer(0.01), Hm + v["base_t"] + 2, -1)
    return [C.on_floor(C.to_tm(M))]


def teacup(v, fast):
    """Taça cónica (largura em cima = diâmetro do modelo) com a fotografia em litofania na parede
    e um anel de base; a curvatura arqueia o perfil."""
    from . import lithoshapes as LS
    D, Z = v["diam"], v["sz"]
    t = v["thick"]
    bh = v["base_h"]
    curv = v.get("curvature", 0.0)
    Rt, Rb = D / 2 - t * 1.5, D * 0.2
    prof = (lambda z: 1 + curv * 0.15 * np.sin(np.pi * z)) if curv else None
    vv = dict(v, tmin=t * 0.5, tmax=t * 1.5, polarity="Positive", orient="Upright", side="Exterior", res=0.25)
    m, Rz = LS._wrap(vv, fast, 0, Rb, Rt, Z, prof)
    R0 = float(Rz[-1, 0])
    base = mf.Manifold.cylinder(bh, R0 + t * 1.5, R0 + t * 1.5, 128).translate((0, 0, -bh + 0.5))
    M = C.to_mf(m) + base
    return [C.on_floor(C.to_tm(M))]


def whistle(v, fast):
    """O corpo do apito tem a forma da silhueta da fotografia (de perfil), com a conduta por cima,
    a janela (opening) e a câmara oca por dentro."""
    ch = v["chamber_h"]; gap = v["duct_gap"]; dl = v["duct_len"]; op = v["opening"]; w = v["width"]; t = v["thick"]
    L = ch * 2.6
    sil = _outline(v, L, ch)
    b = sil.bounds
    top = b[3]
    # conduta: bloco por cima da silhueta, do início até à janela
    x0 = b[0]
    duct_blk = sbox(x0 - dl * 0.3, (b[1] + b[3]) / 2, x0 + dl + op + t, top + gap + 2 * t)
    body2d = sil.union(duct_blk).buffer(0)
    body = C.ext(body2d, w)
    # câmara: silhueta encolhida (sem as faces laterais)
    cham = sil.buffer(-t)
    if not cham.is_empty:
        body = body - C.ext(cham, w - 2 * t, t)
    # canal de ar da conduta e janela
    chan = sbox(x0 - dl * 0.3 - 1, top + t, x0 + dl + op, top + t + gap)
    body = body - C.ext(chan.intersection(body2d.buffer(-0.01).union(chan)), w - 2 * t, t)
    win = sbox(x0 + dl, top - t - 1, x0 + dl + op, top + t + gap + 0.01)
    body = body - C.ext(win, w - 2 * t, t)
    # ligar o canal à câmara pela janela
    body = body - C.ext(sbox(x0 + dl - 0.01, top - t - 1, x0 + dl + op, top + 0.5), w - 2 * t, t)
    m = C.to_tm(body)
    C.rot(m, 90, [1, 0, 0])
    return [C.on_floor(m)]


def wire(v, fast):
    W, H = v["sx"], v["sy"]
    rows, cols = _grid(W, H, 0.3, fast, 400)
    g = _gray(v, rows, cols)
    from skimage import feature, morphology
    edges = feature.canny(g, sigma=max(v.get("lod", 2.0), 0.5))
    edges = morphology.dilation(edges, morphology.disk(1)) > 0
    geom = _raster_geom(edges, W, H)
    ww = v.get("wire_w", 3.0)
    geom = geom.buffer(ww / 2 - 0.3 if ww > 0.6 else 0.01)
    plate = sbox(-W / 2, -H / 2, W / 2, H / 2)
    ring, inner = _frame(v, plate)
    if not ring.is_empty:
        geom = (geom.intersection(inner) if v.get("clipping", "Yes") == "Yes" else geom).union(ring)
    t = v["thick"]
    parts = [C.ext(geom, t)]
    if v.get("solid_base", "No") == "Yes":
        parts.append(C.ext(inner if not ring.is_empty else plate, 0.6))
    return [C.to_tm(mf.Manifold.batch_boolean(parts, mf.OpType.Add))]


def wormhole(v, fast):
    W, H, Z = v["sx"], v["sy"], v["sz"]
    geom = _sil(v, W, H)
    outer = unary_union([Polygon(p.exterior) for p in getattr(geom, "geoms", [geom])]).buffer(0)
    gap, t = v["gap"], v["thick"]
    parts = []
    k = 0
    d = 0.0
    rings = []
    while True:
        a = outer.buffer(-d)
        b = outer.buffer(-d - t)
        if a.is_empty or a.area < 1:
            break
        rings.append(a.difference(b))
        d += t + gap
        k += 1
        if k > 60:
            break
    n = max(len(rings), 1)
    # a parede exterior é a mais alta e os anéis descem para o centro (túnel)
    for i, r in enumerate(rings):
        if not r.is_empty:
            parts.append(C.ext(r, Z * (n - i) / n))
    parts.append(C.ext(outer, max(t * 0.5, 1.0)))
    return [C.to_tm(mf.Manifold.batch_boolean(parts, mf.OpType.Add))]


FN = {"Balloon": balloon, "Block": block, "Box": box, "Cell": cell, "Channel": channel, "Clicker": clicker, "Cutter": cutter,
      "Extrusion": extrusion, "Fabric": fabric, "Fidget": fidget, "Flexi": flexi, "Halftone": halftone, "Layering": layering,
      "Lenticule": lenticule, "Moire": moire, "Morphing": morphing, "Painting": painting, "Pendant": pendant, "Pillar": pillar,
      "Puzzle": puzzle, "Ridge": ridge, "Teacup": teacup, "Whistle": whistle, "Wire": wire, "Wormhole": wormhole}


def run(mesh, v, fast):
    r = FN[v["method"]](v, fast)
    if isinstance(r, tuple):
        ms, names, cols = r
    else:
        ms, names, cols = r, None, None
    return C.result(ms, names, cols)
