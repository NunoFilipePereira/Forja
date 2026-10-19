"""Corte de modelos: planar (vários planos), cónico, flexível, grelha (com padrão desfasado),
radial e modular (pelo tamanho da impressora), com conectores (macho/plug, cavilha/dowel), sobreposição (lap),
chanfro, fecho do corte e numeração das peças."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from shapely.geometry import Point, Polygon, box as sbox
from shapely.ops import unary_union, polylabel
from shapely import affinity
from .meshops import to_manifold, from_manifold, drop_to_floor

METHODS = ["Planar cut", "Conic cut", "Flexi cut", "Grid cut", "Radial cut", "Modular cut"]
UNITS = {"meter": 1000.0, "dm": 100.0, "cm": 10.0, "mm": 1.0, "inch": 25.4}
SHAPES = ["Triangle", "Square", "Pentagon", "Hexagon", "Octagon", "Circle", "D-shape", "Lips", "Native"]
NUMBERS = ["Any", "1", "2", "3", "4", "5", "6", "7", "8"]


# --------------------------------------------------------------------------
# Geometria auxiliar
# --------------------------------------------------------------------------
def frame(mesh):
    lo, hi = mesh.bounds
    return (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-9)


def plane_abcd(mesh, A, B, C, D):
    """Plano A·x' + B·y' + C·z' = D em coordenadas normalizadas (x' = (x − cx)/hx ∈ [−1, 1])."""
    c, h = frame(mesh)
    n = np.array([A / h[0], B / h[1], C / h[2]], float)
    k = np.linalg.norm(n)
    if k < 1e-12:
        raise ValueError("O plano precisa de A, B ou C diferente de zero.")
    d = D + A * c[0] / h[0] + B * c[1] / h[1] + C * c[2] / h[2]
    origin = n * d / (k * k)
    return origin, n / k


def connector_outline(shape, size, section=None):
    r = size / 2
    sides = {"Triangle": 3, "Square": 4, "Pentagon": 5, "Hexagon": 6, "Octagon": 8}
    if shape in sides:
        n = sides[shape]
        return Polygon([(r * math.cos(2 * math.pi * k / n + math.pi / n), r * math.sin(2 * math.pi * k / n + math.pi / n)) for k in range(n)])
    if shape == "Circle":
        return Point(0, 0).buffer(r, quad_segs=16)
    if shape == "D-shape":
        return Point(0, 0).buffer(r, quad_segs=16).intersection(sbox(-r * 0.6, -r, r, r))
    if shape == "Lips":  # duas metades em lente: encaixa numa só orientação
        a = Point(0, -r * 0.45).buffer(r, quad_segs=16); b = Point(0, r * 0.45).buffer(r, quad_segs=16)
        return a.intersection(b)
    return Point(0, 0).buffer(r, quad_segs=16)


def _basis(normal):
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


def _prism(outline2d, origin, normal, z0, z1, taper=1.0):
    """Extrude um contorno 2D (no referencial do plano) entre z0 e z1 ao longo da normal."""
    from .generators import cs
    u, v, n = _basis(normal)
    h = z1 - z0
    M = mf.Manifold.extrude(cs(outline2d), h, 0, 0.0, (taper, taper)).translate((0, 0, z0))
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = u, v, n, origin
    return M.transform(T[:3, :].tolist())


def section_polys(mesh, origin, normal):
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return [], None
    u, v, n = _basis(normal)
    polys = []
    for ent in sec.discrete:
        P = np.asarray(ent) - origin
        if len(P) >= 3:
            polys.append(Polygon(np.c_[P @ u, P @ v]).buffer(0))
    if not polys:
        return [], None
    g = polys[0]
    for p in polys[1:]:
        g = g.symmetric_difference(p)
    g = g.buffer(0)
    return (list(g.geoms) if hasattr(g, "geoms") else [g]), (u, v, n)


def connector_positions(poly, size, number):
    """Pontos dentro do polígono, afastados da borda, para colocar conectores."""
    inner = poly.buffer(-size * 1.2)
    if inner.is_empty:
        return []
    if number == "Any":
        n = int(max(1, min(8, poly.area // (size * size * 40) + 1)))
    else:
        n = int(number)
    if n == 1:
        p = polylabel(inner if inner.geom_type == "Polygon" else max(inner.geoms, key=lambda g: g.area), 0.5)
        return [(p.x, p.y)]
    # distribui pela direção mais comprida do polígono
    mr = inner.minimum_rotated_rectangle
    c = np.array(mr.exterior.coords[:4])
    e1, e2 = c[1] - c[0], c[2] - c[1]
    ax = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    cen = np.array(inner.centroid.coords[0])
    pts = []
    for t in np.linspace(-0.42, 0.42, n):
        q = cen + ax * t
        if inner.contains(Point(*q)):
            pts.append(tuple(q))
    if not pts:
        p = inner.representative_point(); pts = [(p.x, p.y)]
    # nunca dois conectores encostados: distância mínima entre centros = 2,5 × tamanho
    keep = []
    for q in pts:
        if all(np.hypot(q[0] - k[0], q[1] - k[1]) >= size * 2.5 for k in keep):
            keep.append(q)
    return keep


# --------------------------------------------------------------------------
# Um corte planar entre duas partes, com todas as opções
# --------------------------------------------------------------------------
def cut_pair(mesh, origin, normal, opt, region=None, label=None):
    """Corta 'mesh' pelo plano e devolve (parte +, parte −, extras)."""
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    M = to_manifold(mesh)
    A, B = M.split_by_plane(n.tolist(), float(n @ origin))
    if region is not None:  # corte regional: só dentro da região; fora fica unido
        R = to_manifold(region)
        A, B = A ^ R, (B ^ R) + (M - R)   # só a parte de A dentro da região sai; o resto fica inteiro
    polys, basis = section_polys(mesh, origin, n)
    extras = []
    size = opt.get("size", 5.0)
    tol = opt.get("tolerance", -0.2)
    clear = max(-tol, 0.0)
    depth = max(opt.get("depth_ratio", 1.5) * size, 0.5)
    kind = opt.get("connector", "Plug")
    taper = 0.75 if opt.get("profile", "Prism") == "Frustum" else 1.0
    # sobreposição (meia-madeira): passa uma laje de metade da secção de B para A
    if opt.get("lap") and polys:
        lap = opt.get("lap_depth", 5.0)
        for poly in polys:
            mr = poly.minimum_rotated_rectangle
            cc = np.array(mr.exterior.coords[:4]); e1 = cc[1] - cc[0]; e2 = cc[2] - cc[1]
            ax = e2 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e1  # corta transversalmente ao lado maior
            cen = np.array(poly.centroid.coords[0])
            half = Polygon([cen + ax * 0.0 + np.array([-ax[1], ax[0]]) * 1e3, cen + np.array([-ax[1], ax[0]]) * 1e3 + ax * 1e3,
                            cen - np.array([-ax[1], ax[0]]) * 1e3 + ax * 1e3, cen - np.array([-ax[1], ax[0]]) * 1e3])
            slab = _prism(poly.buffer(0.01).intersection(half), origin, n, -lap, 0.0)
            moved = B ^ slab
            A, B = A + moved, B - slab
    if kind != "None" and polys:
        for poly in polys:
            outl_base = None
            if opt.get("shape") == "Native":
                outl_base = poly.buffer(-max(size, 1.0))
                if outl_base.is_empty:
                    continue
                pts = [(0.0, 0.0)]
            else:
                pts = connector_positions(poly, size, opt.get("number", "Any"))
            for (px, py) in pts:
                if outl_base is not None:
                    outl = outl_base
                else:
                    outl = affinity.translate(connector_outline(opt.get("shape", "Octagon"), size), px, py)
                hole = outl.buffer(clear) if clear > 0 else outl
                if kind == "Plug":
                    # macho preso a B, entra em A; furo em A com folga
                    A = A - _prism(hole, origin, n, -0.01, depth + clear, taper)
                    B = B + _prism(outl, origin, n, -0.5, depth, taper)
                elif kind == "Dowel":
                    A = A - _prism(hole, origin, n, -0.01, depth + clear)
                    B = B - _prism(hole, origin, n, -(depth + clear), 0.01)
                    d = from_manifold(_prism(outl, np.zeros(3), np.array([0, 0, 1.0]), 0, 2 * depth - 0.4))
                    extras.append(d)
                elif kind == "Flexi":
                    # pino flexível com fenda (encaixe por pressão)
                    pin = _prism(outl, origin, n, -0.5, depth, taper)
                    b = outl.bounds
                    slot = sbox(b[0] - 1, (b[1] + b[3]) / 2 - max(size * 0.12, 0.4), b[2] + 1, (b[1] + b[3]) / 2 + max(size * 0.12, 0.4))
                    pin = pin - _prism(slot, origin, n, depth * 0.35, depth + 1)
                    A = A - _prism(hole, origin, n, -0.01, depth + clear, taper)
                    B = B + pin
    ch = opt.get("chamfer", 0.0)
    if ch > 0 and polys:
        steps = 6
        for poly in polys:
            for k in range(steps):
                t0, t1 = ch * k / steps, ch * (k + 1) / steps
                ring = poly.buffer(0.01).difference(poly.buffer(-(ch - t1)))
                if ring.is_empty:
                    continue
                A = A - _prism(ring, origin, n, t0 - 0.001, t1)
                B = B - _prism(ring, origin, n, -t1, -t0 + 0.001)
    a, b = from_manifold(A), from_manifold(B)
    if not opt.get("close", True):
        a, b = _open_cut(a, origin, n), _open_cut(b, origin, n)
    if label is not None and opt.get("number_parts") and polys:
        a, b = _engrave_label(a, polys, origin, n, f"{label}A", +1, size), _engrave_label(b, polys, origin, n, f"{label}B", -1, size)
    return a, b, extras


def _open_cut(m, origin, n):
    if not len(m.faces):
        return m
    d = (m.triangles_center - origin) @ n
    keep = ~((np.abs(d) < 1e-4) & (np.abs(m.face_normals @ n) > 0.999))
    out = m.copy(); out.update_faces(keep); out.remove_unreferenced_vertices()
    return out


def _engrave_label(m, polys, origin, n, text, side, size):
    try:
        from .text3d import text_polygons
        poly = max(polys, key=lambda p: p.area)
        h = max(min(math.sqrt(poly.area) * 0.15, 8.0), 2.0)
        g = text_polygons(text, size_mm=h, bold=True)
        gb = g.bounds
        p = poly.buffer(-h).representative_point() if not poly.buffer(-h).is_empty else poly.representative_point()
        g = affinity.translate(g, p.x - (gb[0] + gb[2]) / 2, p.y - (gb[1] + gb[3]) / 2 + size)
        if side < 0:
            g = affinity.scale(g, -1, 1, origin=(p.x, p.y))
        sl = _prism(g, origin, n, 0.0 if side > 0 else -0.6, 0.6 if side > 0 else 0.0)
        return from_manifold(to_manifold(m) - sl)
    except Exception:
        return m


# --------------------------------------------------------------------------
# Métodos
# --------------------------------------------------------------------------
def planar_planes(mesh, planes):
    return [plane_abcd(mesh, *p) for p in planes]


def run_planes(mesh, planes, opt):
    """Corta sequencialmente por vários planos (origem, normal)."""
    pieces = [mesh]
    extras = []
    for k, (o, n) in enumerate(planes):
        nxt = []
        for pc in pieces:
            d = (pc.vertices - o) @ n
            if d.min() < -1e-6 and d.max() > 1e-6:
                region = None
                if opt.get("regional"):
                    r = opt.get("region_size", 30.0)
                    reg = trimesh.creation.box(extents=(r, r, r)); reg.apply_translation(o + np.asarray(opt.get("region_offset", (0, 0, 0))))
                    region = reg
                a, b, ex = cut_pair(pc, o, n, opt, region, label=len(nxt) + 1 if opt.get("number_parts") else None)
                extras += ex
                nxt += [m for m in (b, a) if len(m.faces)]
            else:
                nxt.append(pc)
        pieces = nxt
    total = sum(abs(p.volume) for p in pieces if p.is_watertight) or 1.0
    pieces = [p for p in pieces if not p.is_watertight or abs(p.volume) > total * 1e-4]  # descarta lascas
    return pieces, extras


def grid_planes(mesh, nx, ny, nz, adjust_axis="X", adjust_index=0, adjust_offset=0.0):
    lo, hi = mesh.bounds
    planes = []
    for ax, cnt in enumerate((nx, ny, nz)):
        for i in range(1, int(cnt)):
            p = lo[ax] + (hi[ax] - lo[ax]) * i / cnt
            if "XYZ"[ax] == adjust_axis and i == int(adjust_index):
                p += adjust_offset
            o = np.zeros(3); o[ax] = p
            nrm = np.zeros(3); nrm[ax] = 1
            planes.append((o, nrm))
    return planes


def radial_planes(mesh, direction="Z", n_cuts=2, start_deg=0.0, cu=0.0, cv=0.0):
    """Planos que contêm o eixo (direção) e passam pelo centro deslocado (cu, cv ∈ [−1, 1])."""
    c, h = frame(mesh)
    ax = "XYZ".index(direction)
    others = [i for i in range(3) if i != ax]
    center = c.copy()
    center[others[0]] += cu * h[others[0]]; center[others[1]] += cv * h[others[1]]
    planes = []
    for k in range(int(n_cuts)):
        a = math.radians(start_deg) + math.pi * k / max(int(n_cuts), 1)
        nrm = np.zeros(3)
        nrm[others[0]] = -math.sin(a); nrm[others[1]] = math.cos(a)
        planes.append((center, nrm))
    return planes, center, ax


def run_radial(mesh, direction, n_cuts, start_deg, cu, cv, opt):
    """Cortes radiais: n_cuts planos pelo eixo dão 2·n_cuts setores; com 1 corte, 2 metades."""
    planes, center, ax = radial_planes(mesh, direction, n_cuts, start_deg, cu, cv)
    return run_planes(mesh, planes, opt)


def modular_counts(mesh, printer=(200, 150, 140), clearance=0.0, bed="Rectangle"):
    px, py, pz = printer
    if bed == "Circle":
        px = py = px / math.sqrt(2)  # quadrado inscrito na mesa circular
    avail = np.maximum(np.array([px, py, pz], float) - 2 * clearance, 1.0)
    ext = mesh.extents
    # tenta a orientação das peças em XY que dá menos peças
    a = [int(math.ceil(e / s)) for e, s in zip(ext, avail)]
    b = [int(math.ceil(ext[0] / avail[1])), int(math.ceil(ext[1] / avail[0])), a[2]]
    return a if np.prod(a) <= np.prod(b) else b


def run_modular(mesh, printer, clearance, bed, axial, opt):
    nx, ny, nz = modular_counts(mesh, printer, clearance, bed)
    if axial:  # só ao longo do eixo mais comprido
        counts = [1, 1, 1]
        L = int(np.argmax(mesh.extents))
        counts[L] = max(nx, ny, nz) if max(nx, ny, nz) > 1 else 1
        nx, ny, nz = counts
    pieces, extras = run_planes(mesh, grid_planes(mesh, nx, ny, nz), opt)
    return pieces, extras, (nx, ny, nz)


def run_conic(mesh, cx, cy, cz, dx, dy, dz, sides, vrot, d1, d2, height, tol):
    """Corte cónico: extrai do modelo um tronco de cone/pirâmide (p.ex. tampa de um frasco ou encaixe cónico)."""
    c, h = frame(mesh)
    lo = mesh.bounds[0]
    center = lo + np.array([cx, cy, cz]) * (h * 2)
    d = np.array([dx, dy, dz], float)
    if np.linalg.norm(d) < 1e-9:
        d = np.array([0, 0, 1.0])
    d /= np.linalg.norm(d)
    sides = max(int(sides), 3)
    ring = lambda r: Polygon([(r * math.cos(2 * math.pi * k / sides + math.radians(vrot * 360 / sides / 9)),
                               r * math.sin(2 * math.pi * k / sides + math.radians(vrot * 360 / sides / 9))) for k in range(sides)])
    scale = (d2 / max(d1, 1e-6))
    cutter = _prism(ring(d1 / 2), center - d * height / 2, d, 0, height, scale)
    hole = _prism(ring(d1 / 2 + max(-tol, 0)), center - d * height / 2, d, -0.01, height + 0.01, scale)
    M = to_manifold(mesh)
    inside = from_manifold(M ^ cutter)
    outside = from_manifold(M - hole)
    return [p for p in (outside, inside) if len(p.faces)], []


def run_staggered(mesh, nx, ny, nz, opt):
    from .mesh2 import staggered_cut
    lo, hi = mesh.bounds
    e = hi - lo
    return staggered_cut(mesh, e[0] / max(nx, 1), e[1] / max(ny, 1), e[2] / max(nz, 1)), []


def explode_parts(parts, gap):
    if gap <= 0 or len(parts) < 2:
        return parts
    c = np.mean([p.bounds.mean(axis=0) for p in parts], axis=0)
    for p in parts:
        d = p.bounds.mean(axis=0) - c
        L = np.linalg.norm(d)
        if L > 1e-9:
            p.apply_translation(d / L * gap)
    return parts


def cut(mesh, v):
    """Ponto de entrada: v = valores do painel. Devolve (peças, extras, informação)."""
    opt = {"connector": v["connector"], "profile": v["profile"], "shape": v["shape"], "number": v["number"],
           "depth_ratio": v["depth_ratio"], "size": v["size"], "tolerance": v["tolerance"], "lap": v["lap"],
           "lap_depth": v["lap_depth"], "chamfer": v["chamfer"], "close": v["close"], "number_parts": v["part_number"],
           "regional": v["regional"], "region_size": v["region_size"]}
    m = v["method"]
    info = ""
    if m in ("Planar cut", "Flexi cut"):
        if m == "Flexi cut":
            opt["connector"] = "Flexi" if v["connector"] != "None" else "None"
        planes = []
        for k in range(int(v["n_planes"])):
            planes.append(plane_abcd(mesh, *v["planes"][k]) if k < len(v["planes"]) else plane_abcd(mesh, 0, 0, 1, 0))
        parts, extras = run_planes(mesh, planes, opt)
    elif m == "Conic cut":
        parts, extras = run_conic(mesh, v["cx"], v["cy"], v["cz"], v["dx"], v["dy"], v["dz"], v["sides"], v["vrot"],
                                  v["d1"], v["d2"], v["cone_h"], v["tolerance"])
    elif m == "Grid cut":
        if v["staggered"]:
            parts, extras = run_staggered(mesh, v["gx"], v["gy"], v["gz"], opt)
        else:
            parts, extras = run_planes(mesh, grid_planes(mesh, v["gx"], v["gy"], v["gz"], v["adj_axis"], v["adj_index"], v["adj_offset"]), opt)
    elif m == "Radial cut":
        parts, extras = run_radial(mesh, v["r_dir"], v["r_cuts"], v["r_start"], v["r_cx"], v["r_cy"], opt)
    else:
        parts, extras, counts = run_modular(mesh, (v["px"], v["py"], v["pz"]), v["border"], v["bed"], v["axial"], opt)
        info = f"{counts[0]}×{counts[1]}×{counts[2]} módulos"
    # sem lascas: peças com menos de 0,1 % do volume total são descartadas (restos de cortes desfasados)
    vols = [abs(p.volume) if p.is_watertight else None for p in parts]
    tot = sum(x for x in vols if x) or 1.0
    parts = [p for p, x in zip(parts, vols) if x is None or x > tot * 1e-3]
    return parts, extras, info


def preview_planes(mesh, v):
    """Planos a desenhar na pré-visualização: lista de (origem, normal)."""
    m = v["method"]
    if m in ("Planar cut", "Flexi cut"):
        return [plane_abcd(mesh, *v["planes"][k]) for k in range(min(int(v["n_planes"]), len(v["planes"])))]
    if m == "Grid cut":
        return grid_planes(mesh, v["gx"], v["gy"], v["gz"], v["adj_axis"], v["adj_index"], v["adj_offset"])
    if m == "Radial cut":
        return radial_planes(mesh, v["r_dir"], v["r_cuts"], v["r_start"], v["r_cx"], v["r_cy"])[0]
    if m == "Modular cut":
        nx, ny, nz = modular_counts(mesh, (v["px"], v["py"], v["pz"]), v["border"], v["bed"])
        return grid_planes(mesh, nx, ny, nz)
    return []
