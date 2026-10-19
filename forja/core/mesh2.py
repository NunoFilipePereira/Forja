"""Mais operações 3D: vista explodida, virar/juntar, caixa mínima, cortes radiais/modulares/naturais/por região/
desfasados, encaixes com cavilhas e ímanes, relevo projetado, embutir, mapa de altura, impressora."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from .meshops import to_manifold, from_manifold, _box_mesh, _pin, cut_plane, split_by_planes, densify, drop_to_floor


# --------------------------------------------------------------------------
# Posição / orientação
# --------------------------------------------------------------------------
def explode(meshes, factor=0.5):
    """Afasta cada peça do centro comum (vista explodida). Devolve as translações aplicadas."""
    centers = np.array([m.bounds.mean(axis=0) for m in meshes])
    c = centers.mean(axis=0)
    moves = []
    for m, pc in zip(meshes, centers):
        d = (pc - c) * factor
        m.apply_translation(d); moves.append(d)
    return moves


def flip(mesh, axis):
    """Vira 180° em torno de um eixo que passa pelo centro e volta a pousar no chão."""
    c = mesh.bounds.mean(axis=0)
    ax = np.zeros(3); ax[axis] = 1
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, ax, c))
    return drop_to_floor(mesh)


def reset_coords(mesh):
    c = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-c)
    return mesh


def oriented_box(mesh):
    """Caixa envolvente mínima (orientada). Devolve (transformação que alinha, dimensões)."""
    try:
        T, ext = trimesh.bounds.oriented_bounds(mesh.convex_hull if len(mesh.vertices) > 3 else mesh)
    except Exception:
        # malha degenerada (plana, linha…): caixa alinhada com os eixos
        lo, hi = mesh.bounds
        T = np.eye(4); T[:3, 3] = -(lo + hi) / 2; ext = hi - lo
    return T, np.sort(ext)[::-1]


def align_to_min_box(mesh):
    T, ext = oriented_box(mesh)
    mesh.apply_transform(T)
    return drop_to_floor(mesh), ext


def join_along(a, b, axis=0, gap=0.0, union=True):
    """Encosta B a A ao longo de um eixo (B do lado positivo) e junta-os."""
    b = b.copy()
    d = np.zeros(3)
    d[axis] = a.bounds[1][axis] - b.bounds[0][axis] + gap
    others = [i for i in range(3) if i != axis]
    ca, cb = a.bounds.mean(axis=0), b.bounds.mean(axis=0)
    for i in others:
        d[i] = ca[i] - cb[i]
    b.apply_translation(d)
    if union:
        try:
            return from_manifold(to_manifold(a) + to_manifold(b))
        except Exception:
            pass
    return trimesh.util.concatenate([a, b])


def bury(obj, base, depth=1.0):
    """Assenta 'obj' no topo de 'base', enterrado 'depth' mm, e junta-os num só corpo."""
    o = obj.copy()
    cb = base.bounds.mean(axis=0)
    co = o.bounds.mean(axis=0)
    o.apply_translation((cb[0] - co[0], cb[1] - co[1], base.bounds[1][2] - depth - o.bounds[0][2]))
    return from_manifold(to_manifold(base) + to_manifold(o))


# --------------------------------------------------------------------------
# Cortes
# --------------------------------------------------------------------------
def radial_cut(mesh, n=4, start_deg=0.0, center=None):
    """Corta em n fatias tipo 'bolo' em torno do eixo Z."""
    M = to_manifold(mesh)
    c = mesh.bounds.mean(axis=0) if center is None else np.asarray(center, float)
    if n == 2:
        a = math.radians(start_deg)
        nrm = (-math.sin(a), math.cos(a), 0.0)
        A, B = M.split_by_plane(nrm, float(np.dot(nrm, c)))
        return [from_manifold(A), from_manifold(B)]
    parts = []
    for k in range(n):
        a0 = math.radians(start_deg + 360 * k / n); a1 = math.radians(start_deg + 360 * (k + 1) / n)
        # meio-espaço à esquerda do raio a0 e à direita do raio a1
        n0 = (-math.sin(a0), math.cos(a0), 0.0)
        n1 = (math.sin(a1), -math.cos(a1), 0.0)
        P = M.trim_by_plane(n0, float(np.dot(n0, c))).trim_by_plane(n1, float(np.dot(n1, c)))
        m = from_manifold(P)
        if len(m.faces):
            parts.append(m)
    return parts


def section_areas(mesh, axis=2, samples=200):
    lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
    pos = np.linspace(lo, hi, samples + 2)[1:-1]
    n = np.zeros(3); n[axis] = 1
    areas = []
    for p in pos:
        o = np.zeros(3); o[axis] = p
        s = mesh.section(plane_origin=o, plane_normal=n)
        if s is None:
            areas.append(0.0); continue
        try:
            planar, _ = s.to_2D()
            areas.append(float(sum(pg.area for pg in planar.polygons_full)))
        except Exception:
            areas.append(0.0)
    return pos, np.array(areas)


def natural_cut_positions(mesh, axis=2, n_cuts=1, min_gap_frac=0.12):
    """Escolhe posições de corte nos 'pescoços' do modelo (secção mínima), afastadas entre si."""
    pos, A = section_areas(mesh, axis)
    L = pos[-1] - pos[0]
    margin = L * 0.08
    order = np.argsort(A)
    chosen = []
    for i in order:
        p = pos[i]
        if p < pos[0] + margin or p > pos[-1] - margin or A[i] <= 0:
            continue
        if all(abs(p - q) > L * min_gap_frac for q in chosen):
            chosen.append(p)
        if len(chosen) >= n_cuts:
            break
    return sorted(chosen), pos, A


def regional_cut(mesh, lo, hi):
    """Separa a região dentro de uma caixa do resto do modelo."""
    M = to_manifold(mesh); B = to_manifold(_box_mesh(lo, hi))
    return from_manifold(M ^ B), from_manifold(M - B)


def modular_counts(mesh, printer=(220, 220, 250), margin=5.0):
    ext = mesh.extents
    avail = np.maximum(np.asarray(printer, float) - margin, 1)
    return [int(max(1, math.ceil(e / a))) for e, a in zip(ext, avail)]


def staggered_cut(mesh, cell_x, cell_y, layer_h):
    """Partição em 'tijolos': camadas em Z, e em cada camada alterna o desfasamento da grelha XY."""
    lo, hi = mesh.bounds
    M = to_manifold(mesh)
    parts = []
    nz = max(1, int(math.ceil((hi[2] - lo[2]) / layer_h)))
    for k in range(nz):
        z0 = lo[2] + k * layer_h - (1 if k == 0 else 0); z1 = min(hi[2], lo[2] + (k + 1) * layer_h) + (1 if k == nz - 1 else 0)
        off = cell_x / 2 if k % 2 else 0.0
        x = lo[0] - off - 1
        while x < hi[0] + 1:
            y = lo[1] - 1
            while y < hi[1] + 1:
                P = M ^ to_manifold(_box_mesh([x, y, z0], [x + cell_x, y + cell_y, z1]))
                m = from_manifold(P)
                if len(m.faces):
                    parts.append(m)
                y += cell_y
            x += cell_x
    return parts


def cut_with_connectors(mesh, origin, normal, kind="cavilha", radius=3.0, depth=8.0, clearance=0.2, count=1):
    """Corta por um plano e acrescenta encaixes nas duas faces de corte.
    kind: 'pino' (macho numa parte), 'cavilha' (furos nas duas + cavilhas soltas), 'íman' (alojamentos nas duas)."""
    from shapely.ops import polylabel
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    pos, neg = cut_plane(mesh, origin, normal)
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return pos, neg, []
    planar, to3d = sec.to_2D()
    polys = sorted(planar.polygons_full, key=lambda p: -p.area)
    P, N = to_manifold(pos), to_manifold(neg)
    extras = []
    placed = 0
    for poly in polys:
        if placed >= count:
            break
        pts = [polylabel(poly, 0.5)]
        if count > 1:
            # distribui ao longo do eixo maior do polígono
            mr = poly.minimum_rotated_rectangle
            c = np.array(mr.exterior.coords[:4])
            e1, e2 = c[1] - c[0], c[2] - c[1]
            ax = e1 if np.linalg.norm(e1) > np.linalg.norm(e2) else e2
            cen = np.array(poly.centroid.coords[0])
            pts = []
            for t in np.linspace(-0.3, 0.3, count):
                from shapely.geometry import Point
                q = Point(*(cen + ax * t))
                if poly.contains(q) and poly.exterior.distance(q) > radius * 1.5:
                    pts.append(q)
        for pt in pts:
            if poly.exterior.distance(pt) < radius * 1.5 or placed >= count:
                continue
            p3 = (to3d @ np.array([pt.x, pt.y, 0, 1.0]))[:3]
            axis_vec = normal
            def cyl(r, h, center):
                c = trimesh.creation.cylinder(radius=r, height=h, sections=32)
                c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis_vec))
                c.apply_translation(center)
                return to_manifold(c)
            if kind == "pino":
                P = P - cyl(radius + clearance, depth + 2 * clearance, p3)
                N = N + cyl(radius, depth, p3)
            elif kind == "cavilha":
                hole = cyl(radius + clearance, 2 * depth + 2 * clearance, p3)
                P, N = P - hole, N - hole
                d = trimesh.creation.cylinder(radius=radius, height=2 * depth - 1, sections=32)
                extras.append(d)
            else:  # íman
                sink = cyl(radius + clearance, 2 * depth, p3)
                P, N = P - sink, N - sink
            placed += 1
    if extras:
        for k, d in enumerate(extras):
            d.apply_translation((k * radius * 3, 0, 0))
        extras = [drop_to_floor(trimesh.util.concatenate(extras))]
    return from_manifold(P), from_manifold(N), extras


# --------------------------------------------------------------------------
# Relevo projetado (tatuagem / gravado / embutido)
# --------------------------------------------------------------------------
def project_relief(mesh, gray, direction="cima", center=(0, 0), size=(40, 40), depth=1.0, max_edge=None, invert=False):
    """Projeta uma imagem sobre a superfície e desloca os vértices ao longo da normal.
    depth > 0: relevo (tatuagem); depth < 0: gravado/embutido."""
    g = np.asarray(gray, float)
    if g.ndim == 3:
        g = g.mean(2)
    g = (g - g.min()) / max(g.max() - g.min(), 1e-9)
    if not invert:
        g = 1 - g  # escuro = alto
    w, h = size
    target = max_edge or max(min(w, h) / 120, 0.2)
    m = mesh.copy()
    # refina a malha inteira (mantém-na fechada; subdividir só uma parte cria junções em T e abre a malha)
    try:
        from .meshops import to_manifold, from_manifold
        target = max(target, float(np.sqrt(m.area / (2e6 * 0.43))))
        m = from_manifold(to_manifold(m).refine_to_length(target))
    except Exception:
        for _ in range(8):
            if len(m.faces) > 1_500_000:
                break
            m = m.subdivide()
    V = m.vertices.copy()
    N = m.vertex_normals
    if direction == "cima":
        u, v, dvec = V[:, 0] - (center[0] - w / 2), (center[1] + h / 2) - V[:, 1], np.array([0, 0, 1.0])
    elif direction == "frente":
        u, v, dvec = V[:, 0] - (center[0] - w / 2), (center[1] + h / 2) - V[:, 2], np.array([0, -1.0, 0])
    else:  # lado direito
        u, v, dvec = V[:, 1] - (center[0] - w / 2), (center[1] + h / 2) - V[:, 2], np.array([1.0, 0, 0])
    facing = N @ dvec > 0.25
    inside = (u >= 0) & (u <= w) & (v >= 0) & (v <= h) & facing
    # só a camada visível (o vértice mais exterior na direção da projeção)
    rows, cols = g.shape
    px = np.clip((u / w * (cols - 1)).astype(int), 0, cols - 1)
    py = np.clip((v / h * (rows - 1)).astype(int), 0, rows - 1)
    val = np.where(inside, g[py, px], 0.0)
    V = V + N * (val * depth)[:, None]
    out = trimesh.Trimesh(V, m.faces, process=False)
    return out


# --------------------------------------------------------------------------
# Malha → mapa de altura
# --------------------------------------------------------------------------
def mesh_heightmap(mesh, px=0.5, direction="cima", max_px=800):
    lo, hi = mesh.bounds
    if direction == "cima":
        a, b, up = 0, 1, 2
    elif direction == "frente":
        a, b, up = 0, 2, 1
    else:
        a, b, up = 1, 2, 0
    W = hi[a] - lo[a]; H = hi[b] - lo[b]
    px = max(px, max(W, H) / max_px)
    xs = np.arange(lo[a] + px / 2, hi[a], px); ys = np.arange(hi[b] - px / 2, lo[b], -px)
    X, Y = np.meshgrid(xs, ys)
    n = X.size
    O = np.zeros((n, 3)); D = np.zeros((n, 3))
    O[:, a] = X.ravel(); O[:, b] = Y.ravel()
    sign = -1 if direction == "frente" else 1
    if direction == "frente":
        O[:, up] = lo[up] - 1; D[:, up] = 1
    else:
        O[:, up] = hi[up] + 1; D[:, up] = -1
    loc, idx, _ = mesh.ray.intersects_location(O, D, multiple_hits=False)
    Z = np.full(n, np.nan)
    if direction == "frente":
        Z[idx] = hi[up] - loc[:, up]
    else:
        Z[idx] = loc[:, up] - lo[up]
    return Z.reshape(X.shape), px


# --------------------------------------------------------------------------
# Impressora: caber, encaixar em mesas, estimativas
# --------------------------------------------------------------------------
def fits_printer(mesh, printer=(220, 220, 250)):
    ext = np.sort(mesh.extents)[::-1]
    pr = np.sort(np.asarray(printer, float))[::-1]
    direct = np.all(mesh.extents <= np.asarray(printer, float))
    rotated = np.all(ext <= pr)
    return bool(direct), bool(rotated)


def print_estimate(mesh, infill=0.2, wall=1.2, filament_d=1.75, density=1.24, flow=8.0, layer=0.2):
    """Estimativa grosseira: volume das paredes + enchimento → filamento, peso e tempo."""
    V = abs(mesh.volume) if mesh.is_watertight else mesh.convex_hull.volume
    shell = min(mesh.area * wall, V)
    used = shell + (V - shell) * infill
    L = used / (math.pi * (filament_d / 2) ** 2)          # mm de filamento
    g = used / 1000 * density
    layers = mesh.extents[2] / layer
    t = used / flow + layers * 2.0                        # s: extrusão + mudança de camada
    return {"volume do modelo (cm³)": V / 1000, "material usado (cm³)": used / 1000, "filamento (m)": L / 1000,
            "peso (g)": g, "camadas": int(layers), "tempo estimado (h)": t / 3600}


def pack_on_plates(meshes, bed=(220, 220), gap=5.0, order="volume"):
    """Distribui peças por várias mesas de impressão (encaixe 2D das pegadas). Devolve [(índice, mesa, dx, dy)]."""
    from shapely.geometry import MultiPoint
    from .nesting import nest
    key = {"volume": lambda m: -abs(m.volume) if m.is_watertight else -m.convex_hull.volume,
           "área": lambda m: -m.area,
           "irregularidade": lambda m: -(1 - (abs(m.volume) / max(m.convex_hull.volume, 1e-9)) if m.is_watertight else 0)}[order]
    idx = sorted(range(len(meshes)), key=lambda i: key(meshes[i]))
    items = [(i, MultiPoint(meshes[i].vertices[:, :2]).convex_hull) for i in idx]
    plates, unplaced = nest(items, bed[0], bed[1], gap, 0, rotations=(0, 90))
    res = []
    for p, pl in enumerate(plates):
        for i, g, a in pl["placements"]:
            res.append((i, p, g, a))
    return res, unplaced, plates
