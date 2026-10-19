"""Métodos de conversão para impressão 3D: Block, Clicker, Cocoon, Grid, Helix, Mold, Pipe, Relief, Revolving, Slide,
Spinner, Spring, Thermal, Tiling, Tumbler, Wave e Wireframe. Os parâmetros seguem os do painel."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from scipy import ndimage
from shapely.geometry import Point, Polygon, box as sbox
from shapely import affinity
from .meshops import to_manifold, from_manifold, drop_to_floor, center_on_plate, heightmap_to_solid
from . import generators as gn


def _frame(mesh):
    lo, hi = mesh.bounds
    return lo, hi, (lo + hi) / 2, (hi - lo) / 2


def _size_meas(prefix, m):
    e = m.extents
    return {f"{prefix} X": f"{e[0]:.2f}", f"{prefix} Y": f"{e[1]:.2f}", f"{prefix} Z": f"{e[2]:.2f}"}


def _union(meshes):
    ms = [to_manifold(m) for m in meshes if len(m.faces)]
    if not ms:
        return trimesh.Trimesh()
    return from_manifold(mf.Manifold.batch_boolean(ms, mf.OpType.Add))


def _shell(mesh, wall):
    """Casca com espessura 'wall' (para os métodos que trabalham sobre a superfície)."""
    from .meshops import hollow
    try:
        return hollow(mesh, wall, resolution=110)
    except Exception:
        return mesh


def _cyl_between(p, q, r, sections=10):
    L = np.linalg.norm(q - p)
    if L < 1e-6:
        return None
    c = trimesh.creation.cylinder(radius=r, height=L, sections=sections)
    c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], (q - p) / L))
    c.apply_translation((p + q) / 2)
    return c


# --------------------------------------------------------------------------
# Block: o modelo feito de blocos (tipo peças de construção), ocos com parede
# --------------------------------------------------------------------------
def block(mesh, block_size=10.0, wall=5, fast=False):
    """O modelo em cubos de 'block_size'; 'wall' é a espessura da parede em número de blocos
    (o interior mais fundo do que isso fica oco)."""
    # grelha: centros dos blocos em lo + k·bloco; um bloco fica cheio se tocar na superfície do modelo
    lo, hi = mesh.bounds
    bs = block_size
    n = np.floor((hi - lo) / bs + 0.5).astype(int) + 1
    origin = lo - bs / 2
    pts = trimesh.sample.sample_surface_even(mesh, int(min(1e6, mesh.area / (bs / 25) ** 2) + 20000))[0]
    pts = np.vstack([pts, mesh.vertices])
    ijk = np.clip(np.floor((pts - origin) / bs).astype(int), 0, n - 1)
    mat = np.zeros(tuple(n), bool)
    mat[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    mat = ndimage.binary_fill_holes(mat)
    w = int(round(wall))
    if w > 0:
        inner = ndimage.binary_erosion(mat, iterations=w)
        mat = mat & ~inner
    idx = np.argwhere(mat)
    if not len(idx):
        raise ValueError("O bloco é maior do que o modelo.")
    pts = origin + (idx + 0.5) * bs
    cubes = []
    for p in pts:
        b = trimesh.creation.box(extents=(block_size, block_size, block_size)); b.apply_translation(p); cubes.append(b)
    M = _union(cubes) if len(cubes) < 4000 else trimesh.util.concatenate(cubes)
    return {"meshes": [drop_to_floor(M)], "measure": {"Blocos": len(cubes), **_size_meas("Tamanho", M)}}


def _shrink(m, d):
    s = m.copy()
    c = s.bounds.mean(axis=0); e = np.maximum(s.extents, 1e-9)
    k = np.maximum((e - 2 * d) / e, 0.01)
    s.apply_translation(-c); s.apply_scale(k) if False else None
    s.vertices = s.vertices * k
    s.apply_translation(c)
    return s


# --------------------------------------------------------------------------
# Clicker: o modelo vira um botão que se carrega (haste numa base com mola de encaixe)
# --------------------------------------------------------------------------
def clicker(mesh, shape="Circle", size_xy=15.0, pressed=25.0, base_xy=10.0, base_len=10.0, position=0.0, tol=0.0, fast=False):
    """Corta o modelo na horizontal (posição −1..1 na altura); a parte de cima ganha uma manga que
    entra numa cavidade da parte de baixo, e as duas partes têm os furos para um mecanismo de clique de caneta:
    em cima Ø size_xy × (pressed − base_len), em baixo Ø base_xy × base_len. As peças ficam montadas."""
    from shapely.geometry import Polygon as _P
    lo, hi = mesh.bounds
    cz = (lo[2] + hi[2]) / 2 + position * (hi[2] - lo[2]) / 2
    M = to_manifold(mesh)
    big = float(np.max(hi - lo)) * 4
    top = M.trim_by_plane((0, 0, 1), cz)
    bot = M.trim_by_plane((0, 0, -1), -cz)
    # secção no plano de corte → raio máximo da manga (cabe nas duas partes)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    r_in = None
    up_len0 = max(pressed - base_len, 1.0)
    span_up = np.linspace(0.0, max((hi[2] - cz) * 0.6, 1.0), 9)
    span_dn = -np.linspace(0.5, up_len0 / 2 + 1.0, 4)
    for k, dz in enumerate(np.r_[span_up, span_dn]):  # a manga tem de caber na parte de cima (secção mais estreita) e na de baixo
        sec = mesh.section([0, 0, 1], [0, 0, cz + dz])
        if sec is None:
            continue
        polys = [_P(np.asarray(e)[:, :2]).buffer(0) for e in sec.discrete if len(e) >= 3]
        if not polys:
            continue
        g = max(polys, key=lambda q: q.area)
        if k == 0:
            c = g.centroid; cx, cy = c.x, c.y
        lo_r, hi_r = 0.0, max(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1])
        for _ in range(20):  # maior círculo centrado que cabe na secção
            r = (lo_r + hi_r) / 2
            if g.contains(Point(cx, cy).buffer(r, quad_segs=24)):
                lo_r = r
            else:
                hi_r = r
        r_in = lo_r if r_in is None else min(r_in, lo_r)
    r_in = r_in if r_in is not None else size_xy
    rs = max(r_in - 0.6, size_xy / 2 + 1.2)
    up_len = max(pressed - base_len, 1.0)
    sl = up_len / 2  # comprimento da manga abaixo do plano

    def prism(r, h, z0):
        prof = Point(0, 0).buffer(r, quad_segs=32) if shape == "Circle" else sbox(-r, -r, r, r)
        return gn.extrude(prof, h).translate((cx, cy, z0))
    top = top + prism(rs, sl + 0.01, cz - sl)
    top = top - prism(size_xy / 2, up_len, cz - sl - 0.01)
    gap = max(tol, 0.0)
    bot = bot - prism(rs + gap, sl + 1.0 + 0.01, cz - sl - 1.0)
    bot = bot - prism(base_xy / 2, base_len + 0.01, cz - sl - 1.0 - base_len)
    a, b_ = from_manifold(top), from_manifold(bot)
    return {"meshes": [a, b_], "names": ["Botão (parte de cima)", "Base (parte de baixo)"],
            "measure": {"Peças": 2, "Plano de corte (Z)": f"{cz:.1f} mm", "Manga": f"Ø{2 * rs:.1f} × {sl:.1f} mm",
                        "Mecanismo": f"Ø{size_xy:g} × {up_len:g} mm + Ø{base_xy:g} × {base_len:g} mm"}}


# --------------------------------------------------------------------------
# Cocoon: o modelo dentro de um casulo (caixa, cilindro, polígono, esfera ou forma própria) em duas metades
# --------------------------------------------------------------------------
def cocoon(mesh, shape="Cube", clearance=0.0, ox=0.0, oy=0.0, oz=0.0, wall=2.0, fast=False):
    """Uma só peça — a forma envolvente (Cube, Cylinder, Polygon, Sphere ou Native) à volta do modelo,
    com o modelo como cavidade. As folgas X/Y/Z (−1..1) deslocam a forma em relação ao modelo."""
    lo, hi = mesh.bounds
    e = hi - lo + 2 * max(clearance, 0.0)
    c = (lo + hi) / 2 + np.array([ox, oy, oz]) * (hi - lo) / 2
    if shape == "Cylinder":
        r = math.hypot(e[0], e[1]) / 2
        outer = mf.Manifold.cylinder(e[2], r, r, 96).translate((c[0], c[1], c[2] - e[2] / 2))
    elif shape == "Polygon":
        r = math.hypot(e[0], e[1]) / 2 / math.cos(math.pi / 8)
        outer = mf.Manifold.cylinder(e[2], r, r, 8).rotate((0, 0, 22.5)).translate((c[0], c[1], c[2] - e[2] / 2))
    elif shape == "Sphere":
        outer = mf.Manifold.sphere(float(np.linalg.norm(e)) / 2, 96).translate(tuple(c))
    elif shape == "Native":
        outer = to_manifold(mesh.convex_hull)
        if clearance > 0:
            outer = to_manifold(trimesh.Trimesh(*_offset_hull(mesh.convex_hull, clearance)))
    else:
        outer = mf.Manifold.cube(tuple(e), True).translate(tuple(c))
    M = outer - to_manifold(mesh)
    out = from_manifold(M)
    return {"meshes": [out], "measure": _size_meas("Casulo", out)}


def _offset_hull(h, d):
    pts = np.vstack([h.vertices + n * d for n in (np.eye(3).tolist() + (-np.eye(3)).tolist())])
    hh = trimesh.convex.convex_hull(pts)
    return hh.vertices, hh.faces


# --------------------------------------------------------------------------
# Grid: superfície em grelha (células triangulares, retangulares, hexagonais ou circulares)
# --------------------------------------------------------------------------
def grid(mesh, dual=False, cell="Rectangle", cx=10.0, cy=10.0, line=1.0, ax=0.0, ay=0.0, az=0.0, fast=False):
    """O modelo atravessado por uma grelha 2D (células Triangle, Rectangle, Hexagon ou Circle,
    com o tamanho e a largura da linha indicados) extrudida ao longo de Z e rodada pelos ângulos X/Y/Z.
    Dual = o complemento (as células cheias em vez das linhas)."""
    lo, hi = mesh.bounds
    c = (lo + hi) / 2
    R = float(np.linalg.norm(hi - lo)) / 2 + max(cx, cy)
    lines = []
    w = line / 2
    if cell == "Triangle":
        for ang in (0, 60, 120):
            for k in np.arange(-R, R + cy, cy * math.sqrt(3) / 2):
                lines.append(affinity.rotate(sbox(-R, k - w, R, k + w), ang, origin=(0, 0)))
    elif cell == "Hexagon":
        s_ = cx / math.sqrt(3)  # lado do hexágono (cx = distância entre lados opostos)
        cells = []
        dx, dy = cx, 1.5 * s_
        for j, y in enumerate(np.arange(-R, R + dy, dy)):
            for x in np.arange(-R, R + dx, dx):
                xx = x + (dx / 2 if j % 2 else 0)
                hexp = Polygon([(xx + s_ * math.cos(math.radians(30 + 60 * i)), y + s_ * math.sin(math.radians(30 + 60 * i))) for i in range(6)])
                cells.append(hexp.exterior.buffer(w))
        lines = cells
    elif cell == "Circle":
        for y in np.arange(-R, R + cy, cy):
            for x in np.arange(-R, R + cx, cx):
                lines.append(Point(x, y).buffer(min(cx, cy) / 2, quad_segs=16).exterior.buffer(w))
    else:
        # linhas a meio-passo do centro (x = c ± passo/2, ± 3·passo/2, …)
        nx, ny = int(R / cx) + 1, int(R / cy) + 1
        for k in range(-nx, nx):
            x = (k + 0.5) * cx
            lines.append(sbox(x - w, -R, x + w, R))
        for k in range(-ny, ny):
            y = (k + 0.5) * cy
            lines.append(sbox(-R, y - w, R, y + w))
    from shapely.ops import unary_union
    pat = unary_union(lines)
    if dual:
        pat = sbox(-R, -R, R, R).difference(pat)
    P = mf.Manifold.extrude(gn.cs(pat), 2 * R).translate((0, 0, -R))
    P = P.rotate((ax, ay, az)).translate(tuple(c))
    out = from_manifold(to_manifold(mesh) ^ P)
    return {"meshes": [out], "measure": {"Volume": f"{abs(out.volume) / 1000:.1f} cm³", **_size_meas("Tamanho", out)}}


# --------------------------------------------------------------------------
# Helix: braços helicoidais à volta de um núcleo, com base sólida
# --------------------------------------------------------------------------
def helix(mesh, cxn=0.0, czn=0.0, arms=6, arm_w=5.0, diameter=100.0, rotation=60.0, solid_base=True, tol=0.0, wall=1.6, fast=False):
    """O modelo é dividido em duas peças por uma estrela de braços torcida em hélice —
    Helix_inner (a estrela: braços com a largura indicada até ao diâmetro, a rodar 'rotation' graus ao longo da
    altura) e Helix_outer (as cunhas helicoidais entre os braços, unidas por uma base se 'solid_base')."""
    lo, hi, c, h = _frame(mesh)
    center = np.array([c[0] + cxn * h[0], c[1] + czn * h[1]])
    H = hi[2] - lo[2]
    R = diameter / 2
    bh = arm_w if solid_base else 0.0
    star = unary_union_polys([affinity.rotate(sbox(0, -arm_w / 2, R, arm_w / 2), 360.0 * k / arms, origin=(0, 0))
                              for k in range(int(arms))] + [Point(0, 0).buffer(arm_w / 2 * 1.2, quad_segs=16)])
    n_div = max(int(abs(rotation) / 3), 8)
    S = mf.Manifold.extrude(gn.cs(star), H + 2, n_div, rotation).translate((center[0], center[1], lo[2] - 1))
    M = to_manifold(mesh)
    above = mf.Manifold.cube((h[0] * 4 + 10, h[1] * 4 + 10, H + 10), True).translate((c[0], c[1], lo[2] + bh + (H + 10) / 2))
    inner = (M ^ S) ^ above
    gap = mf.Manifold.extrude(gn.cs(star.buffer(max(tol, 0.0)) if tol > 0 else star), H + 2, n_div, rotation).translate(
        (center[0], center[1], lo[2] - 1)) ^ above
    outer = M - gap
    a, b_ = from_manifold(inner), from_manifold(outer)
    return {"meshes": [a, b_], "names": ["Helix_inner", "Helix_outer"],
            "measure": {"Braços": int(arms), "Volume interior": f"{abs(a.volume) / 1000:.1f} cm³",
                        "Volume exterior": f"{abs(b_.volume) / 1000:.1f} cm³"}}


def unary_union_polys(ps):
    from shapely.ops import unary_union
    return unary_union(ps)


# --------------------------------------------------------------------------
# Mold: molde em várias partes (cortes radiais à volta do eixo vertical)
# --------------------------------------------------------------------------
def offset_solid(mesh, d, pitch=None):
    """Sólido = pontos a distância ≤ d do modelo (alargamento 3D), por campo de distâncias numa grelha de voxels."""
    from skimage import measure
    lo, hi = mesh.bounds
    pitch = pitch or max(float(np.max(hi - lo)) / 140, 0.5)
    pad = d + 3 * pitch
    org = lo - pad
    n = np.ceil((hi - lo + 2 * pad) / pitch).astype(int) + 1
    vox = mesh.voxelized(pitch).fill()
    inside = np.zeros(tuple(n), bool)
    idx = np.round((vox.points - org) / pitch).astype(int)
    ok = np.all((idx >= 0) & (idx < n), 1)
    inside[tuple(idx[ok].T)] = True
    dist = ndimage.distance_transform_edt(~inside) * pitch
    verts, faces, _, _ = measure.marching_cubes(dist, level=d)
    m = trimesh.Trimesh(verts * pitch + org, faces[:, ::-1], process=True)
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return m


def mold(mesh, cuts=2, cxn=0.0, czn=0.0, gap=10.0, thickness=5.0, opening=1.0, tol=0.0, fast=False):
    """Molde-mãe para silicone — cilindro à volta do modelo com uma cavidade igual ao modelo alargado
    'gap' mm (espaço para o silicone), abertura em cima para verter, cortado em 2·cuts partes por planos verticais
    pelo centro, e uma base quadrada onde o modelo assenta. Devolve o objeto, as partes e a base."""
    lo, hi, c, h = _frame(mesh)
    cx, cy = c[0] + cxn * h[0], c[1] + czn * h[1]
    rmax = float(np.max(np.hypot(mesh.vertices[:, 0] - cx, mesh.vertices[:, 1] - cy)))
    R = rmax + gap + thickness + 1.5
    z0 = lo[2] + gap
    ztop = hi[2] + gap + thickness / 2
    cav = to_manifold(offset_solid(mesh, gap, None if not fast else max(float(np.max(hi - lo)) / 60, 1.0)))
    body = mf.Manifold.cylinder(ztop - z0, R, R, 128).translate((cx, cy, z0))
    ro = max(opening * gap * 2.1, 2.0)
    body = body - cav - mf.Manifold.cylinder(thickness * 4, ro, ro, 64).translate((cx, cy, hi[2]))
    n = max(int(cuts), 1)
    parts = []
    for k in range(2 * n):
        a0 = math.pi * k / n
        a1 = math.pi * (k + 1) / n
        wedge = Polygon([(cx, cy)] + [(cx + 3 * R * math.cos(t_), cy + 3 * R * math.sin(t_)) for t_ in np.linspace(a0, a1, 12)])
        W = gn.extrude(wedge.buffer(-max(tol, 0.0) / 2) if tol > 0 else wedge, ztop - z0 + 2).translate((0, 0, z0 - 1))
        parts.append(from_manifold(body ^ W))
    base = mf.Manifold.cube((2 * R, 2 * R, z0 - lo[2] + thickness - 0.5 + 0.8), False).translate((cx - R, cy - R, lo[2] - thickness + 0.5))
    base = base - cav
    out = [mesh.copy()] + parts + [from_manifold(base)]
    names = ["Object"] + [f"Part_{k + 1}" for k in range(2 * n)] + ["Base"]
    return {"meshes": out, "names": names, "measure": {"Partes": 2 * n, "Diâmetro": f"{2 * R:.1f} mm",
                                                      "Silicone (aprox.)": f"{(abs(cav.volume()) - abs(mesh.volume)) / 1000:.0f} cm³"}}


# --------------------------------------------------------------------------
# Pipe: estrutura de tubos (arestas) com conectores impressos (nós)
# --------------------------------------------------------------------------
def pipe(mesh, low_poly=1, cross="Triangle", face="Triangle", outer=6.0, inner=4.0, connector="Male", solid="Solid",
         contact=5.0, contact_t=2.0, chamfer=0.5, tol=0.0, fast=False):
    from .conv_sheet import lowpoly_target
    target = lowpoly_target(mesh, low_poly)
    m = mesh.simplify_quadric_decimation(face_count=target) if len(mesh.faces) > target else mesh.copy()
    m.merge_vertices()
    V = m.vertices
    edges = m.edges_unique
    if face == "Quadrangle":
        # junta pares de triângulos complanares: remove a diagonal mais longa
        angs = m.face_adjacency_angles
        flat = m.face_adjacency_edges[angs < math.radians(15)]
        drop = {tuple(sorted(e)) for e in flat}
        edges = np.array([e for e in edges if tuple(sorted(e)) not in drop]) if drop else edges
    elif face == "Voronoi":
        # rede dual: liga centros de faces vizinhas
        V = m.triangles_center
        edges = m.face_adjacency
    sides = {"Triangle": 3, "Square": 4, "Pentagon": 5, "Hexagon": 6, "Octagon": 8, "Circle": 24}[cross]
    pipes, nodes = [], []
    r_o, r_i = outer / 2, inner / 2
    for a, b in edges:
        p, q = V[a], V[b]
        d = q - p; L = np.linalg.norm(d)
        if L < 2 * contact + 1:
            continue
        u = d / L
        c = _cyl_between(p + u * contact, q - u * contact, r_o, sides)
        if c is None:
            continue
        if r_i > 0 and solid == "Hollow":
            ci = _cyl_between(p + u * (contact - 1), q - u * (contact - 1), r_i, sides)
            try:
                c = from_manifold(to_manifold(c) - to_manifold(ci))
            except Exception:
                pass
        pipes.append(c)
    deg = np.bincount(np.asarray(edges).ravel(), minlength=len(V))
    for i, p in enumerate(V):
        if deg[i] == 0:
            continue
        s = trimesh.creation.icosphere(2, r_o + contact_t); s.apply_translation(p)
        nodes.append(s)
        if connector == "Male":
            for a, b in edges:
                if a == i or b == i:
                    q = V[b] if a == i else V[a]
                    u = (q - p) / max(np.linalg.norm(q - p), 1e-9)
                    pin = _cyl_between(p, p + u * (contact + r_o), r_i if r_i > 0 else r_o * 0.6, sides)
                    if pin is not None:
                        nodes.append(pin)
    pm = trimesh.util.concatenate(pipes) if pipes else trimesh.Trimesh()
    nm = trimesh.util.concatenate(nodes) if nodes else trimesh.Trimesh()
    return {"meshes": [x for x in (pm, nm) if len(x.faces)], "measure": {"Tubos": len(pipes), "Conectores": int((deg > 0).sum())}}


# --------------------------------------------------------------------------
# Relief: baixo-relevo do modelo visto de uma direção, sobre base com moldura
# --------------------------------------------------------------------------
def relief(mesh, direction="+Z", sides="Single-sided", polarity="Positive", thickness=10.0, bury="0", base="Rectangle",
           base_t=5.0, border=5.0, border_t=10.0, clearance=3.0, res=0.5, fast=False):
    m = mesh.copy()
    ax = "XYZ".index(direction[1]); sign = 1 if direction[0] == "+" else -1
    rot = {0: trimesh.transformations.rotation_matrix(-sign * math.pi / 2, [0, 1, 0]),
           1: trimesh.transformations.rotation_matrix(sign * math.pi / 2, [1, 0, 0]),
           2: np.eye(4) if sign > 0 else trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])}[ax]
    m.apply_transform(rot)
    from .mesh2 import mesh_heightmap
    px = max(res * (2 if fast else 1), max(m.extents[:2]) / 400)
    Z, px = mesh_heightmap(m, px, "cima")
    Z = np.where(np.isfinite(Z), Z, np.nan)
    zmax = np.nanmax(Z) if np.isfinite(Z).any() else 1
    b = float(str(bury).rstrip("%") or 0) / 100
    H = np.nan_to_num(Z, nan=0) / max(zmax, 1e-9) * thickness
    H = np.clip(H - b * thickness, 0, None)
    if polarity == "Negative":
        H = np.where(np.isfinite(Z), thickness - H, 0)
    rows, cols = H.shape
    W, D = cols * px, rows * px
    m3 = heightmap_to_solid(H, W, D, base=base_t)
    if base == "Rectangle":
        outline = sbox(-border - clearance, -border - clearance, W + border + clearance, D + border + clearance)
    else:
        # polígono regular circunscrito ao círculo que envolve o relevo (+ folga e moldura)
        R = math.hypot(W, D) / 2 + clearance + border * 1.5
        ns = {"Triangle": 3, "Square": 4, "Pentagon": 5, "Hexagon": 6, "Octagon": 8}.get(base, 0)
        if ns:
            Rv = R / math.cos(math.pi / ns)
            a0 = 0.0 if ns % 2 else math.pi / ns  # vértice para +u (fica para cima depois de pôr o relevo de pé)
            outline = Polygon([(W / 2 + Rv * math.cos(a0 + 2 * math.pi * i / ns), D / 2 + Rv * math.sin(a0 + 2 * math.pi * i / ns)) for i in range(ns)])
        else:
            outline = Point(W / 2, D / 2).buffer(R, quad_segs=48)
    plate = gn.extrude(outline, base_t)
    frame = gn.extrude(outline.difference(outline.buffer(-border)), base_t + border_t) if border > 0 else None
    M = plate + to_manifold(m3)
    M = M ^ gn.extrude(outline, 1e4).translate((0, 0, -1))
    if frame is not None:
        M = M + frame
    out = from_manifold(M)
    if sides == "Double-sided":
        back = out.copy(); back.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
        back.apply_translation((0, 0, out.bounds[0][2] - back.bounds[1][2] + 0.01))
        out = _union([out, back])
    # o relevo fica de pé, virado para a direção escolhida
    out.apply_transform(np.linalg.inv(rot))
    out = drop_to_floor(out)
    return {"meshes": [out], "measure": _size_meas("Relevo", out)}


# --------------------------------------------------------------------------
# Revolving: o modelo em relevo à volta de uma forma de revolução (cilindro, cone, vaso, esfera, espiral)
# --------------------------------------------------------------------------
def revolving(mesh, shape="Cylinder", repeat=2, side="+X", inner_d=50.0, arc=60.0, thick=20.0, cap_line=0.0, cap_t=5.0,
              base_line=2.0, base_t=5.0, fast=False):
    """'repeat' cópias do modelo dispostas à volta de um anel. Cada cópia fica com a espessura radial
    'thick' (eixo indicado em 'side' virado para fora) e é curvada ao longo de um arco de 'arc' graus; a forma de
    revolução (Cylinder, Cone, Vase, Sphere, Spiral) define como o raio varia com a altura. Anel de base e, se
    'cap_line' > 0, anel no topo."""
    m = mesh.copy()
    ax = {"X": 0, "Y": 1}[side[1]]
    sgn = 1 if side[0] == "+" else -1
    lo, hi = m.bounds
    H = hi[2] - lo[2]
    V = m.vertices - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    rad_src = V[:, ax] * sgn                     # eixo do modelo que fica radial
    tan_src = V[:, 1 - ax] * (sgn if ax == 0 else -sgn)
    e_rad = max(np.ptp(rad_src), 1e-9); e_tan = max(np.ptp(tan_src), 1e-9)
    r0 = inner_d / 2
    rm = r0 + thick / 2
    L = math.radians(arc) * rm                  # comprimento do arco a meia espessura
    z = V[:, 2]
    f = z / max(H, 1e-9)
    if shape == "Cone":
        rz = lambda zz: zz / max(H, 1e-9) * H * 0.35
    elif shape == "Vase":
        rz = lambda zz: rm * 0.35 * np.sin(np.pi * zz / max(H, 1e-9))
    elif shape == "Sphere":
        rz = lambda zz: -rm * (1 - np.sqrt(np.clip(1 - ((zz / max(H, 1e-9)) * 2 - 1) ** 2, 0, 1))) * 0.6
    else:
        rz = lambda zz: 0.0 * zz
    parts = []
    ring_top = 0.02 * H
    for k in range(int(repeat)):
        extra = (rm * 0.3 * k) if shape == "Spiral" else 0.0
        r = r0 + extra + (rad_src - rad_src.min()) / e_rad * thick + rz(z)
        t = (tan_src - (tan_src.min() + tan_src.max()) / 2) / e_tan * L
        th = t / (rm + extra) + 2 * math.pi * k / repeat
        P = np.c_[r * np.cos(th), r * np.sin(th), z]
        c = trimesh.Trimesh(P, m.faces, process=False)
        if c.volume < 0:
            c.invert()
        C_ = to_manifold(c).trim_by_plane((0, 0, 1), ring_top)
        parts.append(from_manifold(C_))
    rw = max(base_line * 3.7, 1.0)
    out = parts
    if base_t > 0 and base_line > 0:
        ring = mf.Manifold.cylinder(base_t, rm + rw / 2, rm + rw / 2, 128) - mf.Manifold.cylinder(base_t + 2, rm - rw / 2, rm - rw / 2, 128).translate((0, 0, -1))
        out = out + [from_manifold(ring.translate((0, 0, ring_top - base_t)))]
    if cap_line > 0 and cap_t > 0:
        cw = max(cap_line * 3.7, 1.0)
        rt = rm + float(rz(np.array([H]))[0]) if shape != "Cylinder" else rm
        capm = mf.Manifold.cylinder(cap_t, rt + cw / 2, rt + cw / 2, 128) - mf.Manifold.cylinder(cap_t + 2, rt - cw / 2, rt - cw / 2, 128).translate((0, 0, -1))
        out = out + [from_manifold(capm.translate((0, 0, H - 0.02 * H)))]
    zmin = min(p.bounds[0][2] for p in out)
    for p in out:
        p.apply_translation((0, 0, -zmin))
    allm = trimesh.util.concatenate(out)
    return {"meshes": [allm], "measure": {"Cópias": int(repeat), "Diâmetro": f"{2 * (r0 + thick):.1f} mm",
                                         "Altura": f"{allm.extents[2]:.1f} mm"}}


# --------------------------------------------------------------------------
# Slide: fatias que rodam num eixo (escultura cinética)
# --------------------------------------------------------------------------
def slide(mesh, width=4.0, axle_d=5.0, ay=0.0, az=0.0, tol=0.0, gap=0.0, fast=False):
    """O modelo é cortado em fatias de 'width' ao longo de X, alternadas entre duas peças
    (Shape_1 e Shape_2). As fatias de cada peça ficam ligadas por um eixo ao longo de X: um varão numa peça e um
    tubo concêntrico na outra, para as fatias rodarem umas em relação às outras."""
    lo, hi, c, h = _frame(mesh)
    yc, zc = c[1] + ay * h[1], c[2] + az * h[2]
    L = hi[0] - lo[0] + 2
    M = to_manifold(mesh)
    r_to = axle_d / 2 + 0.3          # tubo (peça 2)
    r_ti = r_to - 0.6
    r_r = r_ti - 0.4 - max(tol, 0)   # varão (peça 1)

    def along_x(r_out, r_in=0.0):
        t = mf.Manifold.cylinder(L, r_out, r_out, 32)
        if r_in > 0:
            t = t - mf.Manifold.cylinder(L + 2, r_in, r_in, 32).translate((0, 0, -1))
        return t.rotate((0, 90, 0)).translate((lo[0] - 1, yc, zc))
    slabs = [[], []]
    n = int(math.ceil((hi[0] - lo[0]) / width))
    for k in range(n):
        x0 = lo[0] + k * width
        box = mf.Manifold.cube((width, h[1] * 4 + 10, h[2] * 4 + 10), True).translate((x0 + width / 2, c[1], c[2]))
        slabs[k % 2].append(M ^ box)
    A = mf.Manifold.batch_boolean(slabs[0], mf.OpType.Add) if slabs[0] else mf.Manifold()
    B = mf.Manifold.batch_boolean(slabs[1], mf.OpType.Add) if slabs[1] else mf.Manifold()
    inside = M  # o eixo só existe dentro do modelo
    rod = along_x(r_r) ^ inside
    tube = along_x(r_to, r_ti) ^ inside
    A = (A - along_x(r_to + max(tol, 0) + 0.2)) + rod
    B = (B - along_x(r_ti, 0) ) + tube
    a, b_ = from_manifold(A), from_manifold(B)
    return {"meshes": [a, b_], "names": ["Shape_1", "Shape_2"], "measure": {"Fatias": n, "Eixo": f"Ø{2 * r_to:.1f} mm"}}


# --------------------------------------------------------------------------
# Spinner: pião (o modelo assenta numa ponta que roda)
# --------------------------------------------------------------------------
def spinner(mesh, top_d=4.0, base_d=8.0, top_h=20.0, base_h=20.0, top_off=5.0, base_off=5.0, fast=False):
    """Pega em cima (Ø top_d, comprimento top_h a partir de top_off abaixo do topo) e ponta em baixo
    (Ø base_d, comprimento base_h a partir de base_off acima da base), ambas com a ponta arredondada e ligadas ao
    modelo por uma transição cónica até à secção do modelo nesse plano."""
    lo, hi, c, h = _frame(mesh)
    cx, cy = c[0], c[1]
    M = to_manifold(mesh)

    def sec_r(z):
        s_ = mesh.section([0, 0, 1], [cx, cy, z])
        if s_ is None:
            return 0.0
        try:
            P = s_.to_2D()[0]
            return max(math.sqrt(sum(q.area for q in P.polygons_full) / math.pi), 0.0)
        except Exception:
            return 0.0

    def stem(r, length, r_sec, z_plane, up):
        """Haste com ponta esférica + cone de transição (z cresce para fora do modelo se up)."""
        rr = max(r_sec, r)
        fl = max((rr - r) / 1.25, 0.01)
        L = max(length - r, fl + 0.1)
        cyl = mf.Manifold.cylinder(L, r, r, 48)
        tip = mf.Manifold.sphere(r, 48).translate((0, 0, L))
        cone = mf.Manifold.cylinder(fl, rr, r, 96)
        part = cyl + tip + cone
        if not up:
            part = part.mirror((0, 0, 1))
        return part.translate((cx, cy, z_plane))
    zt = hi[2] - top_off
    zb = lo[2] + base_off
    parts = [M, stem(top_d / 2, top_h, sec_r(zt), zt, True), stem(base_d / 2, base_h, sec_r(zb), zb, False)]
    out = from_manifold(mf.Manifold.batch_boolean(parts, mf.OpType.Add))
    return {"meshes": [out], "measure": _size_meas("Pião", out)}


# --------------------------------------------------------------------------
# Spring: o modelo cortado em espiral para encolher como uma mola
# --------------------------------------------------------------------------
def spring(mesh, layer_w=10.0, thickness=5.0, separation=0.4, fast=False):
    """A casca do modelo (espessura 'thickness') cortada por uma rampa helicoidal contínua de
    espessura 'separation' com passo 'layer_w' — uma só mola que estica e encolhe."""
    from .cimg import grid_solid
    lo, hi, c, h = _frame(mesh)
    shell = _shell(mesh, thickness)
    H = hi[2] - lo[2]
    pitch = layer_w + separation
    R = float(np.max(np.hypot(mesh.vertices[:, 0] - c[0], mesh.vertices[:, 1] - c[1]))) + 5
    turns = H / pitch + 2
    nt = int(turns * (48 if fast else 96))
    nr = 6
    t = np.linspace(0, turns * 2 * math.pi, nt)
    r = np.linspace(0.05, R, nr)
    T, Rr = np.meshgrid(t, r)
    X = c[0] + Rr * np.cos(T); Y = c[1] + Rr * np.sin(T)
    Z = lo[2] - pitch + pitch * T / (2 * math.pi)
    Pi = np.dstack([X, Y, Z]); Po = np.dstack([X, Y, Z + separation])
    cutter = to_manifold(grid_solid(Po, Pi))
    S = to_manifold(shell) - cutter
    out = from_manifold(S)
    k = int(H / pitch)
    return {"meshes": [out], "measure": {"Voltas": k, "Peças": len(out.split(only_watertight=False)), **_size_meas("Mola", out)}}


# --------------------------------------------------------------------------
# Thermal: relevo com padrão de células (aparência de mapa térmico)
# --------------------------------------------------------------------------
def thermal(mesh, direction="+X", thickness=10.0, bury="0", base_t=5.0, border=5.0, cell="Triangle", cell_size=10.0,
            line=1.0, cell_t=2.0, fast=False):
    """Termoformagem: três peças empilhadas na direção escolhida — o molde macho (relevo do modelo
    sobre uma base), a grelha de células (espessura 'cell_t', linhas 'line', células Triangle/Rectangle/Hexagon/Circle)
    e o molde fêmea (o negativo do relevo)."""
    ax = "XYZ".index(direction[1]); sign = 1 if direction[0] == "+" else -1
    rot = {0: trimesh.transformations.rotation_matrix(-sign * math.pi / 2, [0, 1, 0]),
           1: trimesh.transformations.rotation_matrix(sign * math.pi / 2, [1, 0, 0]),
           2: np.eye(4) if sign > 0 else trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])}[ax]
    r = relief(mesh, direction, "Single-sided", "Positive", thickness, bury, "Rectangle", base_t, border, 0.0, 0.0, fast=fast)
    male = r["meshes"][0]
    male.apply_transform(rot)  # volta ao referencial plano (relevo para +Z)
    male.apply_translation((0, 0, -male.bounds[0][2]))
    lo, hi = male.bounds
    outline = sbox(lo[0], lo[1], hi[0], hi[1])
    holes = []
    if cell != "Solid":
        nx = int((hi[0] - lo[0]) / cell_size) + 2; ny = int((hi[1] - lo[1]) / cell_size) + 2
        for i in range(nx):
            for j in range(ny):
                x, y = lo[0] + i * cell_size, lo[1] + j * cell_size
                if cell == "Rectangle":
                    g = sbox(x + line / 2, y + line / 2, x + cell_size - line / 2, y + cell_size - line / 2)
                elif cell == "Circle":
                    g = Point(x + cell_size / 2, y + cell_size / 2).buffer(cell_size / 2 - line / 2, quad_segs=8)
                elif cell == "Hexagon":
                    rr = cell_size / 2 - line / 2
                    g = Polygon([(x + cell_size / 2 + rr * math.cos(math.radians(a_)), y + cell_size / 2 + rr * math.sin(math.radians(a_))) for a_ in range(0, 360, 60)])
                else:
                    g = Polygon([(x + line, y + line), (x + cell_size - line, y + line), (x + cell_size / 2, y + cell_size - line)])
                holes.append(g)
    from shapely.ops import unary_union
    grid2d = outline.difference(unary_union(holes)).intersection(outline.buffer(-0.01)).union(
        outline.difference(outline.buffer(-line))) if holes else outline
    gap = 10.0
    H = hi[2] - lo[2]
    grid_m = from_manifold(gn.extrude(grid2d, cell_t).translate((0, 0, hi[2] + gap)))
    block = mf.Manifold.cube((hi[0] - lo[0], hi[1] - lo[1], H), False).translate((lo[0], lo[1], lo[2]))
    cav = (to_manifold(male) ^ mf.Manifold.cube((hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, H), False).translate(
        (lo[0] - 1, lo[1] - 1, lo[2] + base_t))).translate((0, 0, -base_t))
    female = block - cav
    fem = from_manifold(female)
    # o molde fêmea fica virado para baixo, sobre a grelha
    fem.apply_transform(trimesh.transformations.reflection_matrix([0, 0, (lo[2] + hi[2]) / 2], [0, 0, 1]))
    fem.invert()
    fem.apply_translation((0, 0, hi[2] + gap + cell_t + 8 - fem.bounds[0][2]))
    parts = [male, grid_m, fem]
    inv = np.linalg.inv(rot)
    for p_ in parts:
        p_.apply_transform(inv)
    z0 = min(p_.bounds[0][2] for p_ in parts)
    for p_ in parts:
        p_.apply_translation((0, 0, -z0))
    return {"meshes": parts, "names": ["Molde macho", "Grelha", "Molde fêmea"],
            "measure": {"Peças": 3, **_size_meas("Relevo", male)}}


# --------------------------------------------------------------------------
# Tiling: textura de imagem em relevo sobre a superfície
# --------------------------------------------------------------------------
def tiling(mesh, image=None, direction="Z", min_raise=0.0, max_raise=1.0, angle=0.0, res=0.5, fast=False):
    from .mesh2 import project_relief
    if image is None:
        # sem imagem de textura o modelo fica igual
        return {"meshes": [mesh.copy()], "measure": {"Aviso": "abra uma imagem (separador Imagem) para a textura", "Triângulos": len(mesh.faces)}}
    img = ndimage.rotate(np.asarray(image, float), angle, reshape=False, mode="wrap") if angle else np.asarray(image, float)
    e = mesh.extents
    a = "XYZ".index(direction)
    dir_name = {2: "cima", 1: "frente", 0: "lado"}[a]
    c = mesh.bounds.mean(axis=0)
    u = c[0] if a != 0 else c[1]
    v = c[1] if a == 2 else c[2]
    size = (e[0], e[1]) if a == 2 else ((e[0], e[2]) if a == 1 else (e[1], e[2]))
    # repete a imagem para cobrir a peça
    reps = max(1, int(max(size) / max(res * img.shape[1], 1)))
    tile = np.tile(img, (reps, reps))
    m = project_relief(mesh, tile, dir_name, (u, v), size, max_raise - min_raise, max_edge=max(res * (2 if fast else 1), 0.3))
    if min_raise:
        m.vertices += m.vertex_normals * min_raise
    if not m.is_watertight:
        m.merge_vertices()
    return {"meshes": [m], "measure": {"Triângulos": len(m.faces)}}


# --------------------------------------------------------------------------
# Tumbler: sempre-em-pé (base redonda com peso em baixo)
# --------------------------------------------------------------------------
def tumbler(mesh, base="Semi-cylinder", base_off=5.0, thickness=3.0, tol=0.0, fast=False):
    """Object = o modelo com um tampão quadrado em baixo; Base = caixa oca (paredes
    'thickness') com o fundo em meio-cilindro (ou hemisfério) para baloiçar, onde o tampão encaixa. A cavidade pode
    levar lastro (areia, plasticina)."""
    lo, hi, c, h = _frame(mesh)
    cx, cy = c[0], c[1]
    t = thickness
    zc = lo[2] + base_off * 0.4          # plano onde começa o objeto
    ztop = lo[2] + base_off               # topo da base
    sec = mesh.section([0, 0, 1], [cx, cy, ztop + 2])
    rs = 10.0
    if sec is not None:
        P = sec.to_2D()[0]
        rs = math.sqrt(max(sum(q.area for q in P.polygons_full), 1) / math.pi)
    s_in = 2 * rs                          # lado do tampão / cavidade
    s_out = s_in + 2 * t
    M = to_manifold(mesh).trim_by_plane((0, 0, 1), zc)
    plug = mf.Manifold.cube((s_in - 2 * max(tol, 0), s_in - 2 * max(tol, 0), ztop - zc + 2), False).translate(
        (cx - s_in / 2 + max(tol, 0), cy - s_in / 2 + max(tol, 0), zc))
    obj = M + (plug ^ mf.Manifold.cube((s_out * 4, s_out * 4, ztop - zc + 2), False).translate((cx - s_out * 2, cy - s_out * 2, zc)))
    R = s_out / 2

    def shape(r, side):
        if base == "Hemisphere":
            low = mf.Manifold.sphere(r, 96).translate((cx, cy, zc))
        else:
            low = mf.Manifold.cylinder(side, r, r, 96).rotate((90, 0, 0)).translate((cx, cy + side / 2, zc))
        low = low.trim_by_plane((0, 0, -1), -zc)
        box = mf.Manifold.cube((side, side, ztop - zc), False).translate((cx - side / 2, cy - side / 2, zc))
        return low + box
    outer = shape(R, s_out)
    inner = shape(R - t, s_in) + mf.Manifold.cube((s_in, s_in, 5), False).translate((cx - s_in / 2, cy - s_in / 2, ztop - 0.5))
    bm = from_manifold(outer - inner)
    om = from_manifold(obj)
    return {"meshes": [om, bm], "names": ["Object", "Base"],
            "measure": {"Base": f"{s_out:.1f} × {s_out:.1f} mm", "Cavidade": f"{s_in:.1f} mm",
                        "Volume da cavidade (lastro)": f"{abs((inner ^ outer).volume()) / 1000:.1f} cm³"}}

# --------------------------------------------------------------------------
# Wave: ondas sobre a superfície (a partir de fontes ou ao longo de um eixo)
# --------------------------------------------------------------------------
def wave(mesh, oscillate="Source", sources=((-0.5, -0.5, 0.0),), amplitude=5.0, attenuation=0.0, wavelength=10.0, phase=0.0, fast=False):
    from .meshops import densify
    m = densify(mesh, max(wavelength / (4 if fast else 8), 0.4))
    V = m.vertices.copy(); N = m.vertex_normals
    lo, hi, c, h = _frame(mesh)
    disp = np.zeros(len(V))
    if oscillate in ("X", "Y", "Z"):
        a = "XYZ".index(oscillate)
        d = V[:, a] - lo[a]
        disp = amplitude * np.sin(2 * math.pi * d / wavelength + math.radians(phase)) * np.exp(-attenuation * d / max(h[a] * 2, 1e-9))
    else:
        for s in sources:
            p = c + np.asarray(s) * h
            d = np.linalg.norm(V - p, axis=1)
            disp += amplitude * np.sin(2 * math.pi * d / wavelength + math.radians(phase)) * np.exp(-attenuation * d / max(np.linalg.norm(h), 1e-9))
    m.vertices = V + N * disp[:, None]
    return {"meshes": [m], "measure": _size_meas("Onda", m)}


# --------------------------------------------------------------------------
# Wireframe: estrutura em arame (axial por camadas, arestas ou Voronoi), 2D ou 3D
# --------------------------------------------------------------------------
def wireframe(mesh, layout="Orthogonal", nx=6, ny=6, nz=6, tilt=0.0, kind="Axial", dim="3D", clearance=3.0, socket=False,
              strut=1.2, fast=False):
    from .conv_sheet import section
    m = mesh.copy()
    if tilt:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(tilt), [0, 0, 1], m.bounds.mean(axis=0)))
    lo, hi = m.bounds
    parts = []
    if kind == "Edge":
        from .generators import wireframe as wf
        out = wf(m, strut, max_edges=1500 if not fast else 600)
        return {"meshes": [out], "measure": {"Tipo": "arestas"}}
    if kind == "Voronoi":
        tgt = 80 if fast else 200
        s = m.simplify_quadric_decimation(face_count=tgt) if len(m.faces) > tgt else m
        C = s.triangles_center
        for a, b in s.face_adjacency:
            cy = _cyl_between(C[a], C[b], strut, 8)
            if cy is not None:
                parts.append(cy)
        for p in C:
            sp = trimesh.creation.icosphere(1, strut * 1.3); sp.apply_translation(p); parts.append(sp)
        return {"meshes": [trimesh.util.concatenate(parts)], "measure": {"Barras": len(s.face_adjacency)}}
    # Axial: anéis de contorno em camadas segundo X, Y e Z (ortogonal) ou radiais
    counts = (int(nx), int(ny), int(nz))
    for ax, n in enumerate(counts):
        if n <= 0:
            continue
        for p in np.linspace(lo[ax], hi[ax], n + 2)[1:-1]:
            g = section(m, ax, p)
            if g is None:
                continue
            geoms = g.geoms if hasattr(g, "geoms") else [g]
            for poly in geoms:
                ring = np.asarray(poly.exterior.coords)
                for i in range(len(ring) - 1):
                    a3 = np.zeros(3); b3 = np.zeros(3)
                    o = [i2 for i2 in range(3) if i2 != ax]
                    a3[ax] = p; b3[ax] = p
                    a3[o[0]], a3[o[1]] = ring[i]; b3[o[0]], b3[o[1]] = ring[i + 1]
                    cy = _cyl_between(a3, b3, strut, 6)
                    if cy is not None:
                        parts.append(cy)
    out = trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()
    res = {"meshes": [out], "measure": {"Barras": len(parts)}}
    if dim == "2D":
        # os contornos de cada secção desenhados lado a lado (para dobrar o arame), numerados
        from .shapes2d import Shape
        from shapely.geometry import Polygon as _P
        rings2d = []
        for ax, n in enumerate(counts):
            if n <= 0:
                continue
            for k, p in enumerate(np.linspace(lo[ax], hi[ax], n + 2)[1:-1]):
                g = section(m, ax, p)
                if g is None:
                    continue
                for j, poly in enumerate(g.geoms if hasattr(g, "geoms") else [g]):
                    rings2d.append((f"{'XYZ'[ax]}{k + 1}.{j + 1}", _P(poly.exterior.coords)))
        shapes, x, y, rowh = [], 0.0, 0.0, 0.0
        W = 300.0
        for name, poly in rings2d:
            b = poly.bounds
            w, h = b[2] - b[0], b[3] - b[1]
            if x + w > W and x > 0:
                x = 0.0; y += rowh + 10; rowh = 0.0
            pp = affinity.translate(poly, x - b[0], y - b[1])
            shapes.append(Shape("geo", {}, pp.exterior.buffer(0.15), name))
            x += w + 10; rowh = max(rowh, h)
        res["shapes"] = shapes
        res["measure"]["Contornos (arames)"] = len(rings2d)
        res["measure"]["Comprimento de arame"] = f"{sum(p.exterior.length for _, p in rings2d) / 1000:.2f} m"
    return res
