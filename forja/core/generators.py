"""Geradores de modelos 3D paramétricos (caixas, vasos, molas, cortadores, moldes, litofanias, puzzles…)."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from shapely.geometry import Polygon, Point, box as sbox, MultiPolygon
from shapely import affinity
from shapely.ops import unary_union
from .meshops import to_manifold, from_manifold, image_to_heightmap, heightmap_to_solid, drop_to_floor, center_on_plate


# --------------------------------------------------------------------------
# Utilitários
# --------------------------------------------------------------------------
def cs(geom):
    """shapely (Multi)Polygon → manifold CrossSection."""
    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    contours = []
    for p in polys:
        if p.is_empty:
            continue
        ext = np.array(p.exterior.coords)[:-1]
        if Polygon(ext).exterior.is_ccw is False:
            ext = ext[::-1]
        contours.append(ext)
        for h in p.interiors:
            hh = np.array(h.coords)[:-1]
            contours.append(hh if not Polygon(hh).exterior.is_ccw else hh[::-1])
    return mf.CrossSection(contours, mf.FillRule.EvenOdd)


def extrude(geom, h, twist=0.0, scale_top=1.0, divisions=0):
    return mf.Manifold.extrude(cs(geom), h, divisions or (int(abs(twist) / 3) + 1 if twist else 0), twist,
                               (scale_top, scale_top))


def rounded_rect(w, d, r=0.0):
    r = min(r, w / 2 - 1e-3, d / 2 - 1e-3)
    b = sbox(-w / 2 + r, -d / 2 + r, w / 2 - r, d / 2 - r)
    return b.buffer(r, quad_segs=16) if r > 0 else sbox(-w / 2, -d / 2, w / 2, d / 2)


def regular_polygon(n, r, star_inner=None, rot=90.0):
    pts = []
    k = 2 * n if star_inner else n
    for i in range(k):
        a = math.radians(rot + 360 * i / k)
        rr = (r if i % 2 == 0 else star_inner) if star_inner else r
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    return Polygon(pts)


SHAPES_2D = ["círculo", "quadrado", "triângulo", "pentágono", "hexágono", "octógono", "estrela", "coração", "losango", "forma D"]


def base_shape(kind, size):
    r = size / 2
    if kind == "círculo": return Point(0, 0).buffer(r, quad_segs=48)
    if kind == "quadrado": return sbox(-r, -r, r, r)
    if kind == "triângulo": return regular_polygon(3, r)
    if kind == "pentágono": return regular_polygon(5, r)
    if kind == "hexágono": return regular_polygon(6, r)
    if kind == "octógono": return regular_polygon(8, r, rot=22.5)
    if kind == "estrela": return regular_polygon(5, r, r * 0.45)
    if kind == "losango": return Polygon([(0, r), (r * 0.7, 0), (0, -r), (-r * 0.7, 0)])
    if kind == "forma D": return Point(0, 0).buffer(r, quad_segs=48).intersection(sbox(-r * 0.6, -r, r, r))
    if kind == "coração":
        t = np.linspace(0, 2 * np.pi, 200)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return Polygon(np.c_[x, y] * (r / 17))
    raise ValueError(kind)


def prism(kind, size, height, twist=0.0, scale_top=1.0):
    return from_manifold(extrude(base_shape(kind, size), height, twist, scale_top))


def pyramid(sides, size, height):
    return prism({3: "triângulo", 4: "quadrado", 5: "pentágono", 6: "hexágono", 8: "octógono"}.get(sides, "quadrado"),
                 size, height, 0, 0.001)


# --------------------------------------------------------------------------
# Caixas
# --------------------------------------------------------------------------
def box_with_lid(w=60, d=40, h=30, wall=2.0, bottom=2.0, radius=4.0, lid_h=8.0, lip=4.0, clearance=0.25,
                 lid_type="encaixe"):
    """Caixa oca + tampa. 'encaixe': a tampa tem um rebordo que entra na caixa; 'sobreposta': a tampa cobre por fora."""
    outer = rounded_rect(w, d, radius)
    inner = rounded_rect(w - 2 * wall, d - 2 * wall, max(radius - wall, 0))
    base = extrude(outer, h) - extrude(inner, h).translate((0, 0, bottom))
    top_t = bottom
    if lid_type == "encaixe":
        lid = extrude(outer, top_t)
        lip_o = rounded_rect(w - 2 * wall - 2 * clearance, d - 2 * wall - 2 * clearance, max(radius - wall - clearance, 0))
        lip_i = rounded_rect(w - 4 * wall - 2 * clearance, d - 4 * wall - 2 * clearance, max(radius - 2 * wall, 0))
        lid = lid + (extrude(lip_o, lip) - extrude(lip_i, lip)).translate((0, 0, top_t))
    else:
        o2 = rounded_rect(w + 2 * wall + 2 * clearance, d + 2 * wall + 2 * clearance, radius + wall)
        i2 = rounded_rect(w + 2 * clearance, d + 2 * clearance, radius + clearance)
        lid = extrude(o2, lid_h) - extrude(i2, lid_h).translate((0, 0, top_t))
    return from_manifold(base), from_manifold(lid)


def nested_boxes(n=3, w=80, d=60, h=40, wall=1.6, gap=1.0, radius=4.0):
    out = []
    for k in range(n):
        s = k * (2 * wall + 2 * gap)
        ww, dd, hh = w - s, d - s, h - k * (wall + gap)
        if ww <= 2 * wall or dd <= 2 * wall or hh <= wall:
            break
        o = rounded_rect(ww, dd, max(radius - k * wall, 0.5))
        i = rounded_rect(ww - 2 * wall, dd - 2 * wall, max(radius - (k + 1) * wall, 0))
        out.append(from_manifold(extrude(o, hh) - extrude(i, hh).translate((0, 0, wall))))
    return out


# --------------------------------------------------------------------------
# Rotação / vasos / molas
# --------------------------------------------------------------------------
def lathe(profile, segments=96, degrees=360.0):
    """Gira um perfil 2D (x = raio ≥ 0, y = altura) em torno do eixo Z."""
    g = profile.intersection(sbox(0, -1e6, 1e6, 1e6))
    m = mf.Manifold.revolve(cs(g), segments, degrees)
    return from_manifold(m)


def vase(height=120, r_bottom=30, r_mid=45, r_top=28, lobes=6, amplitude=0.08, twist=90.0, wall=1.6,
         bottom=2.0, segments=120, layers=80, hollow=True):
    """Vaso paramétrico: perfil curvo + ondulação em lóbulos + torção."""
    def radius(z):
        t = z / height
        return (r_bottom * (1 - t) ** 2 + 2 * r_mid * t * (1 - t) + r_top * t ** 2)

    def solid(offset, z0):
        th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        zs = np.linspace(z0, height, layers)
        V = []
        for z in zs:
            R = radius(z) * (1 + amplitude * np.cos(lobes * (th + np.radians(twist) * z / height))) - offset
            V.append(np.c_[R * np.cos(th), R * np.sin(th), np.full_like(th, z)])
        V = np.vstack(V)
        F = []
        n = segments
        for i in range(layers - 1):
            for j in range(n):
                a, b = i * n + j, i * n + (j + 1) % n
                c, d = a + n, b + n
                F += [[a, b, d], [a, d, c]]
        bc, tc = len(V), len(V) + 1
        V = np.vstack([V, [[0, 0, z0], [0, 0, height]]])
        for j in range(n):
            F.append([bc, (j + 1) % n, j])
            F.append([tc, (layers - 1) * n + j, (layers - 1) * n + (j + 1) % n])
        m = trimesh.Trimesh(V, F, process=True)
        if m.volume < 0:
            m.invert()
        return m

    outer = solid(0, 0)
    if not hollow:
        return outer
    inner = solid(wall, bottom)
    inner.apply_translation((0, 0, 0.01))
    inner_ext = to_manifold(inner) + mf.Manifold.cylinder(height * 0.1, radius(height) * 2, -1, 64).translate((0, 0, height - 0.05))
    return from_manifold(to_manifold(outer) - inner_ext)


def spring(radius=10, wire=1.5, pitch=6, turns=5, segments=24, flat_ends=True):
    n = int(turns * 48)
    t = np.linspace(0, turns * 2 * np.pi, n)
    z = pitch * t / (2 * np.pi)
    path = np.c_[radius * np.cos(t), radius * np.sin(t), z]
    circle = Point(0, 0).buffer(wire, quad_segs=max(4, segments // 4))
    m = trimesh.creation.sweep_polygon(circle, path)
    if flat_ends:
        M = to_manifold(m)
        M = M.trim_by_plane((0, 0, 1), wire * 0.5).trim_by_plane((0, 0, -1), -(z[-1] - wire * 0.5))
        m = from_manifold(M)
    return drop_to_floor(m)


def twisted_extrusion(geom, height, twist=90, scale_top=1.0):
    return from_manifold(extrude(geom, height, twist, scale_top))


# --------------------------------------------------------------------------
# Litofanias e candeeiros
# --------------------------------------------------------------------------
def _tube(H, R, height, closed=True, base=0.0):
    """Superfície exterior r = R + H(θ, z), interior r = R."""
    rows, cols = H.shape
    th = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    zs = np.linspace(height, 0, rows)
    out = (R + H)[..., None] * np.stack([np.cos(th), np.sin(th)], -1)[None]
    V_out = np.c_[out.reshape(-1, 2), np.repeat(zs, cols)]
    V_in = np.c_[(R * np.stack([np.cos(th), np.sin(th)], -1))[None].repeat(rows, 0).reshape(-1, 2), np.repeat(zs, cols)]
    n = rows * cols
    V = np.vstack([V_out, V_in])
    idx = np.arange(n).reshape(rows, cols)
    a = idx[:-1, :]; b = np.roll(idx, -1, 1)[:-1, :]; c = idx[1:, :]; d = np.roll(idx, -1, 1)[1:, :]
    Fo = np.vstack([np.c_[a.ravel(), c.ravel(), b.ravel()], np.c_[b.ravel(), c.ravel(), d.ravel()]])
    Fi = Fo[:, ::-1] + n
    top, bot = idx[0], idx[-1]
    tn, bn = np.roll(top, -1), np.roll(bot, -1)
    Ft = np.vstack([np.c_[top, tn, tn + n], np.c_[top, tn + n, top + n]])
    Fb = np.vstack([np.c_[bot, bot + n, bn + n], np.c_[bot, bn + n, bn]])
    m = trimesh.Trimesh(V, np.vstack([Fo, Fi, Ft[:, ::-1], Fb[:, ::-1]]), process=True)
    if m.volume < 0:
        m.invert()
    return m


def cylinder_lithophane(gray, diameter=80, height=None, t_min=0.8, t_max=3.0, max_px=500, base_h=0.0):
    H = image_to_heightmap(gray, t_max - t_min, max_px=max_px) + t_min
    rows, cols = H.shape
    R = diameter / 2 - t_max
    height = height or (np.pi * diameter * rows / cols)
    m = _tube(H, R, height)
    if base_h > 0:
        ring = mf.Manifold.cylinder(base_h, R + t_max, R + t_max, 128) - mf.Manifold.cylinder(base_h, R - 2, R - 2, 128)
        m = from_manifold(to_manifold(m).translate((0, 0, base_h)) + ring)
    return m


def sphere_lithophane(gray, diameter=100, t_min=0.8, t_max=3.0, max_px=400, opening=40.0):
    """Litofania em esfera (abertura em baixo para a lâmpada)."""
    H = image_to_heightmap(gray, t_max - t_min, max_px=max_px) + t_min
    rows, cols = H.shape
    R = diameter / 2 - t_max
    lat_min = -math.asin(min(opening / 2 / (diameter / 2), 0.99))
    lats = np.linspace(math.pi / 2 * 0.98, lat_min, rows)
    th = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    def surf(r):
        x = r[..., None] * np.cos(lats)[:, None, None] * np.stack([np.cos(th), np.sin(th)], -1)[None]
        z = r * np.sin(lats)[:, None]
        return np.c_[x.reshape(-1, 2), z.reshape(-1)]
    Vo = surf(R + H); Vi = surf(np.full(H.shape, R))
    n = rows * cols
    idx = np.arange(n).reshape(rows, cols)
    a = idx[:-1, :]; b = np.roll(idx, -1, 1)[:-1, :]; c = idx[1:, :]; d = np.roll(idx, -1, 1)[1:, :]
    Fo = np.vstack([np.c_[a.ravel(), c.ravel(), b.ravel()], np.c_[b.ravel(), c.ravel(), d.ravel()]])
    top, bot = idx[0], idx[-1]; tn, bn = np.roll(top, -1), np.roll(bot, -1)
    Ft = np.vstack([np.c_[top, tn, tn + n], np.c_[top, tn + n, top + n]])
    Fb = np.vstack([np.c_[bot, bot + n, bn + n], np.c_[bot, bn + n, bn]])
    m = trimesh.Trimesh(np.vstack([Vo, Vi]), np.vstack([Fo, Fo[:, ::-1] + n, Ft[:, ::-1], Fb[:, ::-1]]), process=True)
    if m.volume < 0:
        m.invert()
    trimesh.repair.fix_normals(m)
    return drop_to_floor(m)


def lamp_base(diameter=80, height=30, wall=2.0, socket_d=28.0, cable_hole=8.0, seat_h=4.0, seat_clear=0.4,
              shade_t=3.0):
    """Base de candeeiro: cilindro oco com furo para casquilho, ranhura para assentar a litofania e furo do cabo."""
    R = diameter / 2
    body = mf.Manifold.cylinder(height, R + wall, R + wall, 128)
    body = body - mf.Manifold.cylinder(height, R - wall, R - wall, 128).translate((0, 0, wall))
    body = body + (mf.Manifold.cylinder(wall, R - wall, R - wall, 128).translate((0, 0, height - wall))
                   - mf.Manifold.cylinder(wall, socket_d / 2, socket_d / 2, 64).translate((0, 0, height - wall)))
    groove = (mf.Manifold.cylinder(seat_h, R + seat_clear, R + seat_clear, 128)
              - mf.Manifold.cylinder(seat_h, R - shade_t - seat_clear, R - shade_t - seat_clear, 128))
    body = body + (mf.Manifold.cylinder(seat_h, R + wall, R + wall, 128) - groove).translate((0, 0, height))
    hole = mf.Manifold.cylinder(diameter * 2, cable_hole / 2, cable_hole / 2, 32).rotate((0, 90, 0)).translate((0, 0, wall + cable_hole))
    return from_manifold(body - hole)


def framed_relief(gray, width=100, relief=3.0, base=1.5, border=5.0, border_h=None, invert=False, max_px=400,
                  hole_d=0.0):
    H = image_to_heightmap(gray, relief, invert=invert, max_px=max_px)
    rows, cols = H.shape
    hgt = width * rows / cols
    m = heightmap_to_solid(H, width, hgt, base=base)
    M = to_manifold(m)
    if border > 0:
        bh = border_h or (base + relief + 0.5)
        frame = (extrude(sbox(-border, -border, width + border, hgt + border), bh)
                 - extrude(sbox(0.01, 0.01, width - 0.01, hgt - 0.01), bh))
        M = M + frame
    if hole_d > 0:
        cx = width / 2; cy = hgt + border + hole_d
        tab = extrude(Point(cx, cy).buffer(hole_d + 3, quad_segs=24).union(sbox(cx - hole_d - 3, hgt, cx + hole_d + 3, cy)), base)
        M = (M + tab) - extrude(Point(cx, cy).buffer(hole_d / 2, quad_segs=24), base * 3).translate((0, 0, -base))
    return from_manifold(M)


def keychain(text_geom=None, shape="retângulo arredondado", w=50, d=20, t=3.0, text_h=1.5, hole_d=4.0):
    if shape == "retângulo arredondado":
        plate = rounded_rect(w, d, d / 2.5)
    else:
        plate = base_shape(shape, max(w, d))
    hb = plate.bounds
    hx = hb[0] + hole_d * 1.4
    plate = plate.union(Point(hx, 0).buffer(hole_d * 1.2, quad_segs=24))
    M = extrude(plate, t) - extrude(Point(hx, 0).buffer(hole_d / 2, quad_segs=24), t * 3).translate((0, 0, -t))
    if text_geom is not None and not text_geom.is_empty:
        tb = text_geom.bounds
        avail_w = (hb[2] - hb[0]) - hole_d * 3.5
        k = min(avail_w / max(tb[2] - tb[0], 1e-6), (hb[3] - hb[1]) * 0.7 / max(tb[3] - tb[1], 1e-6))
        g = affinity.scale(text_geom, k, k, origin=(0, 0))
        gb = g.bounds
        g = affinity.translate(g, (hx + hole_d * 1.5 + hb[2]) / 2 - (gb[0] + gb[2]) / 2, -(gb[1] + gb[3]) / 2)
        M = M + extrude(g, text_h).translate((0, 0, t))
    return center_on_plate(from_manifold(M))


# --------------------------------------------------------------------------
# Cortador de bolachas / carimbo
# --------------------------------------------------------------------------
def cookie_cutter(shape_geom, height=15, blade=0.8, flange_w=4.0, flange_t=1.6, size=None):
    g = shape_geom
    if size:
        b = g.bounds; k = size / max(b[2] - b[0], b[3] - b[1])
        g = affinity.scale(g, k, k, origin="centroid")
    g = unary_union(g.buffer(0))
    blade_ring = g.buffer(blade, join_style="round").difference(g)
    outer = g.buffer(blade + flange_w, join_style="round")
    # uma única secção com o perfil em L evita superfícies coincidentes entre lâmina e aba
    M = extrude(outer.difference(g), flange_t) + extrude(blade_ring, height - flange_t).translate((0, 0, flange_t - 0.01))
    return center_on_plate(from_manifold(M))


def stamp(shape_geom, depth=2.0, base_t=4.0, margin=3.0, handle_h=15.0, mirror=True):
    g = affinity.scale(shape_geom, -1, 1, origin="centroid") if mirror else shape_geom
    b = g.bounds
    base = sbox(b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)
    M = extrude(base, base_t) + extrude(g, depth).translate((0, 0, -depth))
    c = base.centroid
    hr = min(b[2] - b[0], b[3] - b[1]) / 4 + margin
    M = M + mf.Manifold.cylinder(handle_h, hr, hr * 0.7, 48).translate((c.x, c.y, base_t))
    m = from_manifold(M)
    return drop_to_floor(m)


# --------------------------------------------------------------------------
# Molde em duas metades
# --------------------------------------------------------------------------
def two_part_mold(mesh, wall=8.0, pin_r=3.0, pour_d=8.0, clearance=0.2, axis=2):
    lo, hi = mesh.bounds
    lo, hi = lo - wall, hi + wall
    blockM = to_manifold(trimesh.creation.box(extents=hi - lo)).translate(tuple((lo + hi) / 2))
    cavity = to_manifold(mesh)
    M = blockM - cavity
    c = (lo + hi) / 2
    n = [0, 0, 0]; n[axis] = 1
    # canal de vazamento: do topo até ao modelo
    top_axis = 2 if axis != 2 else 0
    pour = mf.Manifold.cylinder(hi[top_axis] - c[top_axis] + 1, pour_d / 2, pour_d / 2, 32)
    if top_axis == 0:
        pour = pour.rotate((0, 90, 0))
    pour = pour.translate(tuple(c))
    M = M - pour
    A, B = M.split_by_plane(n, float(c[axis]))
    # pinos de registo nos cantos
    ax2 = [i for i in range(3) if i != axis]
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            p = c.copy()
            p[ax2[0]] += s1 * ((hi - lo)[ax2[0]] / 2 - wall / 2)
            p[ax2[1]] += s2 * ((hi - lo)[ax2[1]] / 2 - wall / 2)
            pin = mf.Manifold.cylinder(wall, pin_r, pin_r, 32)
            hole = mf.Manifold.cylinder(wall + 2 * clearance, pin_r + clearance, pin_r + clearance, 32)
            rot = (0, 0, 0) if axis == 2 else ((0, 90, 0) if axis == 0 else (-90, 0, 0))
            pin = pin.translate((0, 0, -wall / 2)).rotate(rot).translate(tuple(p))
            hole = hole.translate((0, 0, -(wall + 2 * clearance) / 2)).rotate(rot).translate(tuple(p))
            if s1 * s2 > 0:
                A, B = A + pin, B - hole
            else:
                B, A = B + pin, A - hole
    return from_manifold(A), from_manifold(B)


# --------------------------------------------------------------------------
# Puzzle
# --------------------------------------------------------------------------
def puzzle_pieces(w=150, h=100, cols=4, rows=3, tab=0.2, seed=1):
    """Peças de puzzle (polígonos 2D) com encaixes clássicos."""
    rng = np.random.default_rng(seed)
    cw, ch = w / cols, h / rows
    # sinais dos encaixes nas arestas internas
    hs = rng.choice([-1, 1], (rows, cols - 1))   # arestas verticais
    vs = rng.choice([-1, 1], (rows - 1, cols))   # arestas horizontais

    def edge(p0, p1, sign):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        if sign == 0:
            return [tuple(p0)]
        d = p1 - p0; L = np.linalg.norm(d); u = d / L; n = np.array([-u[1], u[0]]) * sign
        s = L * tab
        pts = [p0, p0 + u * L * 0.38, p0 + u * L * 0.38 + n * s * 0.2]
        t = np.linspace(np.pi * 1.25, -np.pi * 0.25, 20)
        cen = p0 + u * L * 0.5 + n * s * 1.05
        r = s * 0.85
        for a in t:
            pts.append(cen + r * (np.cos(a) * u + np.sin(a) * n) * np.array([1, 1]))
        pts += [p0 + u * L * 0.62 + n * s * 0.2, p0 + u * L * 0.62]
        return [tuple(p) for p in pts]

    pieces = []
    for r in range(rows):
        for c in range(cols):
            x0, y0, x1, y1 = c * cw, r * ch, (c + 1) * cw, (r + 1) * ch
            bottom = 0 if r == 0 else -vs[r - 1, c]
            right = 0 if c == cols - 1 else hs[r, c]
            top = 0 if r == rows - 1 else vs[r, c]
            left = 0 if c == 0 else -hs[r, c - 1]
            pts = edge((x0, y0), (x1, y0), -bottom) + edge((x1, y0), (x1, y1), -right) + \
                  edge((x1, y1), (x0, y1), -top) + edge((x0, y1), (x0, y0), -left)
            pieces.append(Polygon(pts).buffer(0))
    return pieces


# --------------------------------------------------------------------------
# Low poly, estrutura em arame, pilha, suportes, ímanes
# --------------------------------------------------------------------------
def low_poly(mesh, ratio=0.05):
    target = max(20, int(len(mesh.faces) * ratio))
    m = mesh.simplify_quadric_decimation(face_count=target)
    return trimesh.Trimesh(m.vertices, m.faces, process=True)


def wireframe(mesh, strut=1.0, max_edges=3000, node_scale=1.4):
    """Estrutura de barras sobre as arestas da malha (simplificada se necessário)."""
    m = mesh
    if len(m.edges_unique) > max_edges:
        m = mesh.simplify_quadric_decimation(face_count=max(20, int(max_edges / 1.5)))
    parts = []
    V = m.vertices
    for a, b in m.edges_unique:
        p, q = V[a], V[b]
        L = np.linalg.norm(q - p)
        if L < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=strut, height=L, sections=8)
        cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], (q - p) / L))
        cyl.apply_translation((p + q) / 2)
        parts.append(cyl)
    for p in V:
        s = trimesh.creation.icosphere(1, strut * node_scale); s.apply_translation(p); parts.append(s)
    try:
        return from_manifold(mf.Manifold.batch_boolean([to_manifold(x) for x in parts], mf.OpType.Add))
    except Exception:
        return trimesh.util.concatenate(parts)


def stack_copies(mesh, n=3, gap=0.3):
    h = mesh.extents[2]
    return trimesh.util.concatenate([mesh.copy().apply_translation((0, 0, k * (h + gap))) for k in range(n)])


def supports(mesh, angle=45.0, spacing=4.0, radius=0.8, tip=0.4):
    """Suportes verticais em pilar sob as zonas com saliência (raio para baixo até ao chão ou ao próprio modelo)."""
    nz = mesh.face_normals[:, 2]
    over = nz < -np.cos(np.radians(angle))
    if not over.any():
        return None
    centers = mesh.triangles_center[over]
    # amostragem em grelha para não criar pilares a mais
    keys = np.round(centers[:, :2] / spacing).astype(int)
    _, first = np.unique(keys, axis=0, return_index=True)
    pts = centers[first]
    z0 = mesh.bounds[0][2]
    pts = pts[pts[:, 2] > z0 + 0.5]
    parts = []
    origins = pts - np.array([0, 0, 0.05])
    dirs = np.tile([0, 0, -1.0], (len(pts), 1))
    hits, ray_idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    floor = {i: z0 for i in range(len(pts))}
    for h, i in zip(hits, ray_idx):
        floor[i] = max(floor[i], h[2])
    for i, p in enumerate(pts):
        bottom = floor[i]
        L = p[2] - bottom
        if L < 1.0:
            continue
        col = trimesh.creation.cone(radius=radius, height=L)  # largo em baixo, fino em cima
        col.apply_translation((p[0], p[1], bottom))
        parts.append(col)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def magnet_holes(mesh, diameter=6.2, depth=2.2, count=1, spacing=20.0):
    M = to_manifold(mesh)
    c = mesh.bounds.mean(axis=0); z0 = mesh.bounds[0][2]
    offs = [0.0] if count == 1 else np.linspace(-spacing * (count - 1) / 2, spacing * (count - 1) / 2, count)
    for dx in offs:
        M = M - mf.Manifold.cylinder(depth + 0.01, diameter / 2, diameter / 2, 48).translate((c[0] + dx, c[1], z0 - 0.01))
    return from_manifold(M)


def dowel(diameter=5.0, length=20.0, chamfer=0.6):
    r = diameter / 2
    prof = Polygon([(0, 0), (r - chamfer, 0), (r, chamfer), (r, length - chamfer), (r - chamfer, length), (0, length)])
    return lathe(prof, 48)


# --------------------------------------------------------------------------
# Acessórios para fotografias / litofanias
# --------------------------------------------------------------------------
def photo_stand(width=100, thickness=3.0, slot_clear=0.3, base_d=30, height=15, angle=10.0):
    """Base em pé (pilar) com ranhura para segurar uma litofania/fotografia plana."""
    w = width * 0.6
    base = rounded_rect(w, base_d, 4)
    M = extrude(base, 4)
    rib = extrude(rounded_rect(w, thickness + 8, 2), height).translate((0, 0, 3.99))
    slot = extrude(sbox(-w, -(thickness + slot_clear) / 2, w, (thickness + slot_clear) / 2), height + 10)
    slot = slot.rotate((-angle, 0, 0)).translate((0, 0, 4))
    return from_manifold((M + rib) - slot)


def hanger(width=30, thickness=3.0, hole_d=5.0, depth=6.0):
    """Gancho/argola para pendurar, colado nas costas de um quadro."""
    tab = Point(0, 0).buffer(width / 2, quad_segs=32).intersection(sbox(-width, -2, width, width))
    tab = tab.union(sbox(-width / 2, -depth, width / 2, 0))
    M = extrude(tab, thickness) - extrude(Point(0, width / 4).buffer(hole_d / 2, quad_segs=24), thickness * 3).translate((0, 0, -thickness))
    return from_manifold(M)


def claw_clip(thickness=3.0, length=25, opening=0.4, wall=2.0):
    """Garra em U para prender uma foto ou cartão a uma base."""
    gap = thickness + opening
    outer = sbox(-(gap / 2 + wall), 0, gap / 2 + wall, length)
    inner = sbox(-gap / 2, wall, gap / 2, length + 1)
    prof = outer.difference(inner)
    M = extrude(prof, 15)
    return drop_to_floor(from_manifold(M))


def pillar(diameter=20, height=80, flutes=12, flute_depth=1.5, base_h=6, cap_h=4):
    """Pilar decorativo com caneluras, base e capitel."""
    t = np.linspace(0, 2 * np.pi, flutes * 16, endpoint=False)
    r = diameter / 2 - flute_depth * (0.5 + 0.5 * np.cos(flutes * t))
    shaft = Polygon(np.c_[r * np.cos(t), r * np.sin(t)])
    M = extrude(shaft, height)
    M = M + extrude(rounded_rect(diameter * 1.4, diameter * 1.4, 2), base_h)
    M = M + extrude(rounded_rect(diameter * 1.3, diameter * 1.3, 2), cap_h).translate((0, 0, height - cap_h))
    return from_manifold(M)
