"""Métodos de conversão de corte em chapa: Stack, Hash, Star, Grill, Contour, Plate e Folding.
Cada um devolve as peças 2D (já encaixadas em chapas), a montagem 3D e as medidas (peças e tempos)."""
import math
import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, LineString, box as sbox, MultiPolygon
from shapely.ops import unary_union, polylabel
from shapely import affinity
from .meshops import to_manifold, from_manifold
from . import nesting

LASER_SPEED = 8.4       # mm/s, calibrado com o tempo de máquina (MDF 3 mm)
PIERCE_TIME = 0.5       # s por contorno
ASSEMBLY_PER_PART = 85  # s por peça (estimativa de montagem)


# --------------------------------------------------------------------------
# Utilitários
# --------------------------------------------------------------------------
def _axes(direction):
    a = "XYZ".index(direction)
    o = [i for i in range(3) if i != a]
    return a, o[0], o[1]


def section(mesh, axis, pos):
    """Secção como polígono no plano das duas outras coordenadas."""
    a, u, v = _axes("XYZ"[axis])
    n = np.zeros(3); n[a] = 1
    o = np.zeros(3); o[a] = pos
    sec = mesh.section(plane_origin=o, plane_normal=n)
    if sec is None:
        return None
    polys = []
    for ent in sec.discrete:
        P = np.asarray(ent)
        if len(P) >= 3:
            polys.append(Polygon(np.c_[P[:, u], P[:, v]]).buffer(0))
    if not polys:
        return None
    g = polys[0]
    for p in polys[1:]:
        g = g.symmetric_difference(p)
    g = g.buffer(0)
    return None if g.is_empty else g


def layer_positions(lo, hi, thickness=None, count=None):
    if count:
        return list(np.linspace(lo, hi, int(count) + 2)[1:-1])
    t = max(thickness, 0.1)
    return list(np.arange(lo + t / 2, hi, t))


def ring(g, width):
    """Contorno com largura 'width' (0 = peça cheia)."""
    if not width or width <= 0:
        return g
    inner = g.buffer(-width)
    return g if inner.is_empty else g.difference(inner).buffer(0)


def place_on_sheets(parts, sheet_w, sheet_h, gap=3.0, mode="Compact", rotate=True):
    """parts: [(nome, polígono)]. Devolve (formas posicionadas por chapa, n.º de chapas, não colocadas)."""
    items = [(k, g) for k, (n, g) in enumerate(parts) if g is not None and not g.is_empty]
    if mode == "Ordered":
        order = items
    elif mode == "Balanced":
        order = sorted(items, key=lambda it: -(it[1].bounds[3] - it[1].bounds[1]))
    else:
        order = sorted(items, key=lambda it: -it[1].area)
    rots = (0, 90, 180, 270) if rotate else (0,)
    plates, unplaced = nesting.nest(order, sheet_w, sheet_h, gap, 3.0, rots)
    placed = []
    for p_i, pl in enumerate(plates):
        for k, g, a in pl["placements"]:
            placed.append((parts[k][0], affinity.translate(g, p_i * (sheet_w + 40), 0), p_i))
    return placed, len(plates), [parts[k][0] for k in unplaced]


def lowpoly_target(mesh, lp):
    """'Low poly' é a fração de faces que se mantém (0 = automático: cerca de 100 faces)."""
    n = len(mesh.faces)
    if lp is None or lp <= 0:
        return int(min(n, 100))
    if lp <= 1:
        return int(max(8, round(n * lp)))
    return int(max(8, min(n, lp)))


def measures(parts, n_sheets, extra=None, asm_per_part=None, extra_cut=0.0):
    cut = sum(g.length for _, g in parts if g is not None)
    contours = sum(len(list(g.geoms)) if hasattr(g, "geoms") else 1 for _, g in parts if g is not None)
    mt = (cut + extra_cut) / LASER_SPEED + contours * PIERCE_TIME
    at = len(parts) * (asm_per_part or ASSEMBLY_PER_PART)

    def fmt(s):
        return f"{s / 3600:.1f} h" if s >= 3600 else f"{max(1, round(s / 60))} min"
    m = {"Peças": len(parts), "Chapas": n_sheets, "Tempo de máquina": fmt(mt), "Tempo de montagem": fmt(at), "Tempo total": fmt(mt + at)}
    m.update(extra or {})
    return m


def label_shape(text, size, at):
    """Texto (para gravação) centrado num ponto."""
    if not text or size <= 0:
        return None
    from .text3d import text_polygons
    g = text_polygons(str(text), size_mm=size, bold=False)
    b = g.bounds
    return affinity.translate(g, at[0] - (b[0] + b[2]) / 2, at[1] - (b[1] + b[3]) / 2)


def extrude_at(poly, axis, pos, t):
    """Peça 2D (nas coordenadas das outras duas) → sólido 3D centrado em 'pos' ao longo do eixo."""
    from .text3d import extrude
    m = extrude(poly, t)
    m.apply_translation((0, 0, pos - t / 2))
    a, u, v = _axes("XYZ"[axis])
    P = np.zeros_like(m.vertices)
    P[:, u], P[:, v], P[:, a] = m.vertices[:, 0], m.vertices[:, 1], m.vertices[:, 2]
    out = trimesh.Trimesh(P, m.faces, process=False)
    if out.volume < 0:
        out.invert()
    return out


def _result(parts_named, assembly, sheet, mode, rotate, extra=None, engrave=(), asm_per_part=None, extra_cut=0.0):
    placed, n_sheets, unplaced = place_on_sheets(parts_named, sheet[0], sheet[1], 3.0, mode, rotate)
    from .shapes2d import Shape
    shapes = [Shape("geo", {}, g, name) for name, g, _ in placed]
    for k in range(n_sheets):
        s = Shape("rc", dict(left=k * (sheet[0] + 40), bottom=0, right=k * (sheet[0] + 40) + sheet[0], top=sheet[1]), name=f"chapa {k + 1}")
        s.style.update(dash="tracejada", stroke="#888888")
        shapes.append(s)
    meas = measures(parts_named, n_sheets, extra, asm_per_part, extra_cut)
    if unplaced:
        meas["Não cabem na chapa"] = len(unplaced)
    return {"meshes": assembly, "shapes": shapes, "measure": meas}


# --------------------------------------------------------------------------
# Stack: camadas empilhadas com cavilhas de alinhamento e número gravado
# --------------------------------------------------------------------------
def stack(mesh, direction="Z", sheet=(600, 450), t=3.2, contour_w=15.0, dowel=5.0, text_size=5.0, mode="Compact", rotate=True, fast=False):
    a, u, v = _axes(direction)
    lo, hi = mesh.bounds
    pos = layer_positions(lo[a], hi[a], t)
    secs = [(p, section(mesh, a, p)) for p in pos]
    secs = [(p, g) for p, g in secs if g is not None]
    if not secs:
        raise ValueError("O modelo não tem secções nesta direção.")
    # furos de cavilha: dentro da interseção das camadas centrais
    core = None
    for p, g in secs[len(secs) // 4: max(len(secs) * 3 // 4, len(secs) // 4 + 1)]:
        core = g if core is None else core.intersection(g)
    holes = []
    if dowel > 0 and core is not None and not core.is_empty:
        inner = core.buffer(-(dowel + 2))
        if not inner.is_empty:
            c = polylabel(inner if inner.geom_type == "Polygon" else max(inner.geoms, key=lambda x: x.area), 0.5)
            b = inner.bounds
            d = min(b[2] - b[0], b[3] - b[1]) * 0.3
            for q in (Point(c.x - d, c.y), Point(c.x + d, c.y)):
                if inner.contains(q):
                    holes.append(q.buffer(dowel / 2, quad_segs=12))
    parts, asm = [], []
    for k, (p, g) in enumerate(secs):
        piece = ring(g, contour_w)
        for h in holes:
            if piece.buffer(-0.5).contains(h.centroid):
                piece = piece.difference(h)
        lab = label_shape(k + 1, text_size, polylabel(piece if piece.geom_type == "Polygon" else max(piece.geoms, key=lambda x: x.area), 1.0).coords[0]) if text_size > 0 and not fast else None
        parts.append((f"camada {k + 1}", piece.union(lab.boundary.buffer(0.05)) if False else piece))
        asm.append(extrude_at(piece, a, p, t * 0.98))
    assembly = [trimesh.util.concatenate(asm)]
    return _result(parts, assembly, sheet, mode, rotate, {"Camadas": len(parts), "Cavilhas": len(holes)})


# --------------------------------------------------------------------------
# Hash: grelha de fatias encaixadas (ortogonal) ou radial
# --------------------------------------------------------------------------
def hash_(mesh, sheet=(600, 450), t=3.2, layout="Orthogonal", surface=False, contour_w=15.0, nx=6, nz=6, tilt=0.0, tol=0.0,
          text_size=5.0, mode="Compact", rotate=True, fast=False):
    from .sheet import hash_slices, radial_slices
    m = mesh.copy()
    if tilt:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(tilt), [0, 0, 1], m.bounds.mean(axis=0)))
    parts, asm = [], []
    if layout == "Radial":
        res, base = radial_slices(m, max(2, int(nx)), t, max(tol, 0.05))
        for name, ang, g in res:
            parts.append((name, ring(g, contour_w) if surface else g))
            pc = extrude_at(g, 1, 0, t)
            pc.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 0, 1]))
            asm.append(pc)
        parts.append(("base", base))
    else:
        parts, asm = _hash_xz(m, int(nx), int(nz), t, max(tol, 0.0), contour_w if surface else 0)
    return _result(parts, [trimesh.util.concatenate(asm)] if asm else [], sheet, mode, rotate, asm_per_part=66)


def _intervals(geom, line):
    """Intervalos (ao longo da linha) onde a geometria tem material."""
    x = geom.intersection(line)
    out = []
    for g in getattr(x, "geoms", [x]):
        if g.geom_type == "LineString" and not g.is_empty:
            c = np.asarray(g.coords)
            out.append(c)
    return out


def _hash_xz(m, nx, nz, t, tol, contour_w):
    """Placas horizontais (Z) inteiras, com ranhuras curtas (2 × espessura) nos dois bordos em cada
    cruzamento; placas perpendiculares a X cortadas ao meio (centro do modelo em Y) — cada metade entra pelo seu lado,
    com ranhuras desde o centro até à ranhura da placa Z. Cada placa conta como as peças em que fica dividida."""
    from shapely.geometry import LineString
    lo, hi = m.bounds
    cy = (lo[1] + hi[1]) / 2
    xs = np.linspace(lo[0], hi[0], nx + 2)[1:-1]
    zs = np.linspace(lo[2], hi[2], nz + 2)[1:-1]
    X = {x: section(m, 0, x) for x in xs}     # coordenadas (y, z)
    Z = {z: section(m, 2, z) for z in zs}     # coordenadas (x, y)
    w = t + 2 * tol
    d = 2 * t
    BIG = float(np.max(hi - lo)) + 20
    cutX = {x: [] for x in xs}; cutZ = {z: [] for z in zs}
    for x, gx in X.items():
        if gx is None:
            continue
        for z, gz in Z.items():
            if gz is None:
                continue
            ia = [(c[:, 0].min(), c[:, 0].max()) for c in _intervals(gx, LineString([(lo[1] - 5, z), (hi[1] + 5, z)]))]
            ib = [(c[:, 1].min(), c[:, 1].max()) for c in _intervals(gz, LineString([(x, lo[1] - 5), (x, hi[1] + 5)]))]
            for y0, y1 in ia:
                for v0, v1 in ib:
                    ya, yb = max(y0, v0), min(y1, v1)
                    if yb - ya < 2 * d + 1:
                        continue
                    if yb > cy + d:   # lado +Y
                        cutZ[z].append(sbox(x - w / 2, yb - d, x + w / 2, yb + 1))
                        cutX[x].append(sbox(max(ya, cy) - 0.01, z - w / 2, yb - d, z + w / 2))
                    if ya < cy - d:   # lado −Y
                        cutZ[z].append(sbox(x - w / 2, ya - 1, x + w / 2, ya + d))
                        cutX[x].append(sbox(ya + d, z - w / 2, min(yb, cy) + 0.01, z + w / 2))
    parts, asm = [], []
    split = sbox(cy - max(tol, 0.0) / 2 - 0.05, -BIG * 2, cy + max(tol, 0.0) / 2 + 0.05, BIG * 2)
    for i, x in enumerate(xs):
        g = X[x]
        if g is None:
            continue
        piece = ring(g, contour_w) if contour_w else g
        piece = piece.difference(split)
        if cutX[x]:
            piece = piece.difference(unary_union(cutX[x]))
        piece = piece.buffer(0)
        comps = sorted([q for q in getattr(piece, "geoms", [piece]) if q.area > 0.5], key=lambda q: -q.area)
        for k, q in enumerate(comps):
            parts.append((f"X{i + 1}.1.{k + 1}", q))
            asm.append(extrude_at(q, 0, x, t))
    for i, z in enumerate(zs):
        g = Z[z]
        if g is None:
            continue
        piece = ring(g, contour_w) if contour_w else g
        piece = piece.difference(unary_union(cutZ[z])).buffer(0) if cutZ[z] else piece
        comps = sorted([q for q in getattr(piece, "geoms", [piece]) if q.area > 0.5], key=lambda q: -q.area)
        for k, q in enumerate(comps):
            parts.append((f"Z{i + 1}.{k + 1}", q))
            asm.append(extrude_at(q, 2, z, t))
    return parts, asm

# --------------------------------------------------------------------------
# Star: fatias radiais em estrela à volta de um eixo vertical
# --------------------------------------------------------------------------
def star(mesh, sheet=(600, 450), t=3.2, cx=0.0, cz=0.0, n=6, start=0.0, tol=0.0, text_size=5.0, mode="Compact", rotate=True, fast=False):
    c = mesh.bounds.mean(axis=0); h = mesh.extents / 2
    center = np.array([c[0] + cx * h[0], c[1] + cz * h[1], 0])
    m = mesh.copy(); m.apply_translation((-center[0], -center[1], 0))
    from .sheet import _section2d
    parts, asm = [], []
    n = max(2, int(n))
    slot_w = t + 2 * max(tol, 0.05)
    for k in range(n):
        ang = math.radians(start) + math.pi * k / n
        mm = m.copy(); mm.apply_transform(trimesh.transformations.rotation_matrix(-ang, [0, 0, 1]))
        g = _section2d(mm, 1, 0.0)  # plano XZ: coordenadas (x, z)
        if g is None:
            continue
        zlo, zhi = g.bounds[1], g.bounds[3]
        H = zhi - zlo
        # ranhuras centrais: fatia k recebe ranhura de baixo (metade inferior) ou de cima, alternando por nível
        frac0, frac1 = k / n, (k + 1) / n
        keep = sbox(-slot_w / 2, zlo + H * frac0, slot_w / 2, zlo + H * frac1)
        cut = sbox(-slot_w / 2, zlo - 1, slot_w / 2, zhi + 1).difference(keep)
        piece = g.difference(cut).buffer(0)
        parts.append((f"raio {k + 1}", piece))
        pc = extrude_at(piece, 1, 0, t)
        pc.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 0, 1])); pc.apply_translation((center[0], center[1], 0))
        asm.append(pc)
    return _result(parts, [trimesh.util.concatenate(asm)] if asm else [], sheet, mode, rotate, asm_per_part=40)


# --------------------------------------------------------------------------
# Grill: aletas paralelas encaixadas numa base
# --------------------------------------------------------------------------
def grill(mesh, sheet=(600, 450), t=3.2, direction="X", layout="Orthogonal", base_pos=0.0, base_t=5.0, layers=6, angle=0.0, tol=0.0,
          text_size=5.0, mode="Compact", rotate=True, fast=False):
    a, u, v = _axes(direction if direction in "XYZ" else "X")
    m = mesh.copy()
    if angle:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(angle), [0, 0, 1], m.bounds.mean(axis=0)))
    lo, hi = m.bounds
    # base: plano horizontal à altura base_pos ∈ [−1, 1]
    zb = (lo[2] + hi[2]) / 2 + base_pos * (hi[2] - lo[2]) / 2
    zb = min(max(zb, lo[2] + base_t / 2), hi[2] - base_t / 2)
    pos = layer_positions(lo[a], hi[a], count=max(1, int(layers)))
    if a == 2:
        a, u, v = 0, 1, 2
        pos = layer_positions(lo[0], hi[0], count=max(1, int(layers)))
    slot = t + 2 * max(tol, 0.05)
    parts, asm = [], []
    base_poly = section(m, 2, zb)
    if base_poly is None:
        raise ValueError("A base fica fora do modelo: ajuste 'Base [−1, 1]'.")
    base_cuts = []
    for k, p in enumerate(pos):
        g = section(m, a, p)  # coordenadas (u, z)
        if g is None:
            continue
        # aleta: ranhura de baixo até a meio da espessura da base
        fin_slot = sbox(-1e5, zb - base_t / 2 - 1, 1e5, zb)
        fin = g.difference(fin_slot.intersection(g).buffer(0).difference(sbox(-1e5, -1e5, 1e5, -1e5))) if False else g
        # a base recebe ranhura por cima até meio; a aleta, ranhura por baixo
        bx = sbox(p - slot / 2, -1e5, p + slot / 2, 1e5) if a == 0 else sbox(-1e5, p - slot / 2, 1e5, p + slot / 2)
        base_cuts.append(bx)
        parts.append((f"aleta {k + 1}", fin))
        asm.append(extrude_at(fin, a, p, t))
    base = base_poly.difference(unary_union(base_cuts).intersection(base_poly.buffer(-max(base_t, 3)))) if base_cuts else base_poly
    parts.append(("base", base))
    asm.append(extrude_at(base, 2, zb, base_t))
    return _result(parts, [trimesh.util.concatenate(asm)], sheet, mode, rotate)


# --------------------------------------------------------------------------
# Contour: contornos montados em hastes (armação triangular, quadrada…) — típico de candeeiros
# --------------------------------------------------------------------------
def contour(mesh, sheet=(600, 450), t=3.2, direction="Z", frame="Triangle", clearance=3.0, socket=False, layers=None,
            mode="Compact", rotate=True, fast=False):
    """Placas da forma escolhida (Triangle … Circle) empilhadas na direção indicada, uma por cada
    espessura de chapa; cada placa tem a secção do modelo recortada (o modelo aparece como vazio) e é numerada.
    A moldura é a menor forma regular que envolve todas as secções mais a folga."""
    a, u, v = _axes(direction)
    lo, hi = mesh.bounds
    # uma placa por espessura, desde a base até ao topo (⌈altura / espessura⌉ + 1 placas)
    n = int(math.floor((hi[a] - lo[a]) / t)) + 1
    pos = [min(lo[a] + k * t + 0.05, hi[a] - 0.05) for k in range(n)]
    secs = [(p, section(mesh, a, p)) for p in pos]
    secs = [(p, g) for p, g in secs if g is not None]
    if not secs:
        raise ValueError("Sem secções nesta direção.")
    allg = unary_union([g for _, g in secs])
    # moldura: polígono regular circunscrito ao círculo que envolve o modelo (+ folga)
    bb = allg.bounds
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    R = max(math.hypot(x - cx, y - cy) for x, y in allg.convex_hull.exterior.coords) + clearance
    nside = {"Triangle": 3, "Rectangle": 4, "Square": 4, "Pentagon": 5, "Hexagon": 6, "Octagon": 8}.get(frame, 0)
    if nside:
        Rv = R / math.cos(math.pi / nside)
        a0 = math.pi / 2 if nside % 2 else math.pi / nside
        outer = Polygon([(cx + Rv * math.cos(a0 + 2 * math.pi * i / nside), cy + Rv * math.sin(a0 + 2 * math.pi * i / nside)) for i in range(nside)])
    else:
        outer = Point(cx, cy).buffer(R, quad_segs=48)
    parts, asm = [], []
    oc = outer.centroid
    for k, (p, g) in enumerate(secs):
        piece = outer.difference(g).buffer(0)
        p = p - 0.05 + t / 2 if k < len(secs) - 1 else p
        if socket and k == 0:
            piece = piece.difference(Point(oc.x, oc.y).buffer(20, quad_segs=24))
        parts.append((f"{k + 1}", piece))
        asm.append(extrude_at(piece, a, p, t))
    return _result(parts, [trimesh.util.concatenate(asm)], sheet, mode, rotate, asm_per_part=88)


# --------------------------------------------------------------------------
# Plate: poliedro low-poly em placas com juntas em dente
# --------------------------------------------------------------------------
def plate(mesh, sheet=(600, 450), t=3.2, low_poly=1, finger=15.0, tol=0.0, text_size=5.0, mode="Compact", rotate=True, fast=False):
    target = lowpoly_target(mesh, low_poly)
    m = mesh.simplify_quadric_decimation(face_count=target) if len(mesh.faces) > target else mesh.copy()
    # agrupa faces quase complanares em placas
    from trimesh.graph import connected_components
    adj = m.face_adjacency
    ang = m.face_adjacency_angles
    same = adj[ang < math.radians(4)]
    groups = connected_components(same, nodes=np.arange(len(m.faces)), min_len=1)
    parts, asm = [], []
    for k, fids in enumerate(groups):
        fids = np.asarray(fids)
        n = m.face_normals[fids].mean(axis=0); n /= np.linalg.norm(n)
        o = m.triangles_center[fids].mean(axis=0)
        a_ = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n, a_); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        tris = [Polygon([((p - o) @ e1, (p - o) @ e2) for p in m.vertices[m.faces[f]]]) for f in fids]
        whole = unary_union([tr.buffer(0) for tr in tris]).buffer(0)
        # um grupo de faces no mesmo plano pode ter partes separadas: cada parte é uma placa
        subs = [g for g in getattr(whole, "geoms", [whole]) if g.geom_type == "Polygon" and g.area >= 1]
        for j, poly in enumerate(subs):
            # dentes ao longo do contorno (alternados), recuo de meia espessura
            shrunk = poly.buffer(-t / 2)
            if not shrunk.is_empty:
                shrunk = max(getattr(shrunk, "geoms", [shrunk]), key=lambda g: g.area)
                poly = shrunk
            teeth = []
            L = poly.exterior.length
            nt = max(1, int(L / max(finger, 3)))
            for i in range(0, nt, 2):
                s0 = poly.exterior.interpolate(i / nt, normalized=True); s1 = poly.exterior.interpolate((i + 1) / nt, normalized=True)
                seg = LineString([s0, s1]).buffer(t / 2 + tol, cap_style=2)
                teeth.append(seg)
            piece = poly.union(unary_union(teeth).difference(poly.buffer(-t / 2))).buffer(0) if teeth else poly
            parts.append((f"placa {k + 1}" + (f".{j + 1}" if len(subs) > 1 else ""), piece))
            T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = e1, e2, n, o - n * t / 2
            from .text3d import extrude
            pm = extrude(piece, t); pm.apply_transform(T)
            asm.append(pm)
    return _result(parts, [trimesh.util.concatenate(asm)] if asm else [m], sheet, mode, rotate, {"Faces do poliedro": len(m.faces)}, asm_per_part=151)


# --------------------------------------------------------------------------
# Folding: planificação (papercraft) com abas de colagem
# --------------------------------------------------------------------------
def folding(mesh, sheet=(600, 450), t=0.3, low_poly=1, max_fold=1, tab=5.0, mode="Compact", rotate=True, fast=False):
    target = lowpoly_target(mesh, low_poly)
    m = mesh.simplify_quadric_decimation(face_count=target) if len(mesh.faces) > target else mesh.copy()
    m.merge_vertices()
    F = m.faces; V = m.vertices
    nbr = {i: [] for i in range(len(F))}
    for (f1, f2), e in zip(m.face_adjacency, m.face_adjacency_edges):
        nbr[f1].append((f2, tuple(e))); nbr[f2].append((f1, tuple(e)))
    placed = {}
    pieces = []

    def place_first(f):
        a, b, c = V[F[f]]
        e1 = b - a; L = np.linalg.norm(e1); e1 /= L
        n = np.cross(b - a, c - a); n /= np.linalg.norm(n); e2 = np.cross(n, e1)
        return {F[f][0]: np.array([0.0, 0.0]), F[f][1]: np.array([L, 0.0]), F[f][2]: np.array([(c - a) @ e1, (c - a) @ e2])}

    def unfold_vertex(known, e, w):
        (i, j) = e
        pi, pj = known[i], known[j]
        A, B, C = V[i], V[j], V[w]
        ab = B - A; L = np.linalg.norm(ab)
        x = (C - A) @ ab / L
        y = np.linalg.norm(np.cross(ab, C - A)) / L
        d = (pj - pi) / np.linalg.norm(pj - pi); nrm = np.array([-d[1], d[0]])
        return pi + d * x, nrm, y

    remaining = set(range(len(F)))
    max_faces = max(4, int(len(F) / max(1, max_fold)))
    while remaining:
        start = min(remaining)
        coords = place_first(start)
        piece_faces = {start: coords}
        geom = Polygon([coords[v] for v in F[start]])
        queue = [start]; remaining.discard(start)
        cut_edges = []
        while queue and len(piece_faces) < max_faces:
            f = queue.pop(0)
            for g, e in nbr[f]:
                if g not in remaining:
                    continue
                c = piece_faces[f]
                w = [x for x in F[g] if x not in e][0]
                base, nrm, y = unfold_vertex(c, e, w)
                third = [x for x in F[f] if x not in e][0]
                s = np.sign((c[third] - c[e[0]]) @ nrm) or 1
                pw = base - nrm * y * s
                cg = {e[0]: c[e[0]], e[1]: c[e[1]], w: pw}
                tri = Polygon([cg[v] for v in F[g]])
                if tri.is_valid and tri.area > 1e-6 and geom.intersection(tri).area < 1e-6 * max(tri.area, 1):
                    geom = geom.union(tri)
                    piece_faces[g] = cg; remaining.discard(g); queue.append(g)
                else:
                    cut_edges.append((c[e[0]], c[e[1]]))
        # abas nas arestas de corte
        tabs = []
        for p0, p1 in cut_edges[::2]:
            d = p1 - p0; L = np.linalg.norm(d)
            if L < tab * 1.5:
                continue
            dn = d / L; nrm = np.array([-dn[1], dn[0]])
            mid = (p0 + p1) / 2
            if geom.contains(Point(*(mid + nrm * 0.1))):
                nrm = -nrm
            tp = Polygon([p0 + dn * tab * 0.3, p1 - dn * tab * 0.3, p1 - dn * tab + nrm * tab, p0 + dn * tab + nrm * tab]).buffer(0)
            if tp.is_valid and not tp.is_empty:
                tabs.append(tp)
        try:
            piece = geom.buffer(0).union(unary_union(tabs).buffer(0)) if tabs else geom
        except Exception:
            piece = geom
        pieces.append((f"planificação {len(pieces) + 1}", piece.buffer(0)))
    # linhas de dobra (arestas interiores) também passam pelo laser (vincar), como no tempo de máquina
    fold_len = float(np.sum(m.edges_unique_length)) * 1.3
    return _result(pieces, [m], sheet, mode, rotate, {"Faces": len(F)}, asm_per_part=260, extra_cut=fold_len)
