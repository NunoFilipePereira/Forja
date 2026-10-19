"""Corte de chapa (laser/CNC): caixas com juntas em dente, planificação de cones, fatias encaixadas,
divisão de formas grandes, pontes e entradas/saídas de corte."""
import math
import numpy as np
from shapely.geometry import Polygon, LineString, Point, box as sbox, MultiPolygon
from shapely import affinity
from shapely.ops import unary_union, split


# --------------------------------------------------------------------------
# Caixa com juntas em dente (finger joints)
# --------------------------------------------------------------------------
def _finger_edge(length, t, finger, start_out, tol):
    """Lista de segmentos ao longo de uma aresta: True = dente para fora (ocupa a espessura)."""
    n = max(3, int(round(length / finger)))
    if n % 2 == 0:
        n += 1
    w = length / n
    return [(i * w, (i + 1) * w, (i % 2 == 0) == start_out) for i in range(n)]


def _panel(W, H, t, finger, edges, tol=0.1):
    """Painel W×H; edges = (baixo, direita, cima, esquerda) com 'm' (dentes macho começam fora),
    'f' (fêmea), ou None (aresta lisa). Os dentes ficam dentro do retângulo W×H (espessura t na borda)."""
    poly = sbox(0, 0, W, H)
    cuts = []
    specs = [
        ("b", W, lambda a, b: sbox(a, 0, b, t)),
        ("r", H, lambda a, b: sbox(W - t, a, W, b)),
        ("t", W, lambda a, b: sbox(a, H - t, b, H)),
        ("l", H, lambda a, b: sbox(0, a, t, b)),
    ]
    for (side, L, mk), kind in zip(specs, edges):
        if kind is None:
            continue
        for a, b, out in _finger_edge(L, t, finger, kind == "m", tol):
            if not out:
                # remove a reentrância com folga (tolerância) para o dente da outra peça encaixar
                cuts.append(mk(a - tol if a > 0 else a - t, b + tol if b < L else b + t))
    if cuts:
        poly = poly.difference(unary_union(cuts))
    return poly.buffer(0)


def finger_box(w=100, d=80, h=60, t=3.0, finger=10.0, tol=0.1, open_top=False, gap=5.0):
    """Seis (ou cinco) painéis 2D de uma caixa com juntas em dente. Dimensões exteriores."""
    if tol >= t / 2:
        raise ValueError("A tolerância tem de ser menor que metade da espessura da chapa.")
    if finger < t:
        raise ValueError("A largura do dente tem de ser ≥ espessura da chapa.")
    top = None if open_top else "f"
    panels = {
        "fundo": _panel(w, d, t, finger, ("f", "f", "f", "f")),
        "frente": _panel(w, h, t, finger, ("m", "f", top and "m", "f")),
        "trás": _panel(w, h, t, finger, ("m", "f", top and "m", "f")),
        "esquerda": _panel(d, h, t, finger, ("m", "m", top and "m", "m")),
        "direita": _panel(d, h, t, finger, ("m", "m", top and "m", "m")),
    }
    if not open_top:
        panels["tampa"] = _panel(w, d, t, finger, ("f", "f", "f", "f"))
    # disposição lado a lado
    out, x, y, rowh = [], 0.0, 0.0, 0.0
    for name, p in panels.items():
        b = p.bounds
        if x + (b[2] - b[0]) > 2.2 * (w + d) and x > 0:
            x, y, rowh = 0.0, y + rowh + gap, 0.0
        out.append((name, affinity.translate(p, x - b[0], y - b[1])))
        x += b[2] - b[0] + gap
        rowh = max(rowh, b[3] - b[1])
    return out


# --------------------------------------------------------------------------
# Leque: planificação de um tronco de cone
# --------------------------------------------------------------------------
def unfan(d_bottom, d_top, height, segments=180, overlap=0.0):
    """Planificação (setor de coroa circular) de um tronco de cone. Devolve (polígono, info)."""
    r1, r2 = d_bottom / 2, d_top / 2
    if abs(r1 - r2) < 1e-9:
        L = 2 * math.pi * r1 + overlap
        return sbox(0, 0, L, height), {"tipo": "retângulo", "comprimento": L, "altura": height}
    big, small = max(r1, r2), min(r1, r2)
    s = math.hypot(height, big - small)            # geratriz
    R_out = s * big / (big - small)                # raio exterior do setor
    R_in = R_out - s
    ang = 2 * math.pi * big / R_out + (overlap / R_out if overlap else 0)
    t = np.linspace(-ang / 2, ang / 2, segments)
    outer = np.c_[R_out * np.sin(t), R_out * np.cos(t)]
    inner = np.c_[R_in * np.sin(t[::-1]), R_in * np.cos(t[::-1])]
    poly = Polygon(np.vstack([outer, inner]))
    return poly, {"raio exterior": R_out, "raio interior": R_in, "ângulo (°)": math.degrees(ang), "geratriz": s}


def fan_from_lines(l1, l2):
    """Dadas as duas arestas laterais de um leque desenhado (2 linhas), calcula o centro e os raios."""
    (a1, b1), (a2, b2) = np.array(l1, float), np.array(l2, float)
    d1, d2 = b1 - a1, b2 - a2
    A = np.array([d1, -d2]).T
    t, _ = np.linalg.solve(A, a2 - a1)
    c = a1 + t * d1
    rs = sorted([np.linalg.norm(p - c) for p in (a1, b1, a2, b2)])
    ang = math.degrees(math.acos(np.clip(abs(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)), -1, 1)))
    R_in, R_out = (rs[0] + rs[1]) / 2, (rs[2] + rs[3]) / 2
    # cone equivalente
    frac = ang / 360
    return {"centro": tuple(c), "raio interior": R_in, "raio exterior": R_out, "ângulo (°)": ang,
            "diâmetro maior do cone": 2 * R_out * frac, "diâmetro menor do cone": 2 * R_in * frac,
            "altura do cone": math.sqrt(max((R_out - R_in) ** 2 - ((R_out - R_in) * frac) ** 2, 0))}


# --------------------------------------------------------------------------
# Esculturas em fatias encaixadas (cardinal / "hash" e radial)
# --------------------------------------------------------------------------
def _section2d(mesh, axis, pos):
    """Secção da malha num plano normal ao eixo; devolve polígono em coordenadas (u, z) do plano."""
    n = np.zeros(3); n[axis] = 1
    o = np.zeros(3); o[axis] = pos
    sec = mesh.section(plane_origin=o, plane_normal=n)
    if sec is None:
        return None
    other = 1 if axis == 0 else 0
    polys = []
    for ent in sec.discrete:
        pts = np.asarray(ent)
        if len(pts) >= 3:
            polys.append(Polygon(np.c_[pts[:, other], pts[:, 2]]).buffer(0))
    if not polys:
        return None
    # regra par-ímpar para furos
    g = polys[0]
    for p in polys[1:]:
        g = g.symmetric_difference(p)
    return g.buffer(0)


def _overlap_z(gx, gy, x, y, lo, hi):
    """Intervalo de z onde as fatias X (em x) e Y (em y) se cruzam: a reta vertical (x, y)."""
    lx = gx.intersection(LineString([(y, lo[2] - 1), (y, hi[2] + 1)]))   # fatia X, coordenadas (y, z)
    ly = gy.intersection(LineString([(x, lo[2] - 1), (x, hi[2] + 1)]))   # fatia Y, coordenadas (x, z)
    if lx.is_empty or ly.is_empty:
        return None
    z0 = max(lx.bounds[1], ly.bounds[1]); z1 = min(lx.bounds[3], ly.bounds[3])
    return (z0, z1) if z1 > z0 else None


def hash_slices(mesh, nx=5, ny=5, t=3.0, tol=0.1, margin=0.1):
    """Fatias ortogonais (X e Y) com ranhuras a meia altura para encaixarem em grelha."""
    lo, hi = mesh.bounds
    xs = np.linspace(lo[0], hi[0], nx + 2)[1:-1]
    ys = np.linspace(lo[1], hi[1], ny + 2)[1:-1]
    X = {x: _section2d(mesh, 0, x) for x in xs}   # coordenadas (y, z)
    Y = {y: _section2d(mesh, 1, y) for y in ys}   # coordenadas (x, z)
    w = t + 2 * tol
    out = []
    for x, gx in X.items():
        if gx is None:
            continue
        cut = []
        for y, gy in Y.items():
            if gy is None:
                continue
            # extensão em z comum às duas fatias sobre a linha de interseção
            zr = _overlap_z(gx, gy, x, y, lo, hi)
            if zr is None:
                continue
            z0, z1 = zr
            zm = (z0 + z1) / 2
            cut.append(sbox(y - w / 2, zm, y + w / 2, z1 + 1))     # fatias X: ranhura de cima
        out.append(("X", x, gx.difference(unary_union(cut)) if cut else gx))
    for y, gy in Y.items():
        if gy is None:
            continue
        cut = []
        for x, gx in X.items():
            if gx is None:
                continue
            zr = _overlap_z(gx, gy, x, y, lo, hi)
            if zr is None:
                continue
            z0, z1 = zr
            zm = (z0 + z1) / 2
            cut.append(sbox(x - w / 2, z0 - 1, x + w / 2, zm))     # fatias Y: ranhura de baixo
        out.append(("Y", y, gy.difference(unary_union(cut)) if cut else gy))
    return out


def radial_slices(mesh, n=8, t=3.0, tol=0.1, hub=None):
    """Meias-fatias radiais em torno do eixo Z do objeto + disco de base com ranhuras."""
    c = mesh.bounds.mean(axis=0)
    m = mesh.copy(); m.apply_translation((-c[0], -c[1], 0))
    R = np.linalg.norm(m.vertices[:, :2], axis=1).max()
    hub = hub if hub is not None else max(t * 2, R * 0.08)
    out = []
    feet = []
    for k in range(n):
        a = math.pi * k / n * 2
        # roda o objeto para que o plano radial fique no plano XZ (x ≥ 0)
        mm = m.copy()
        import trimesh
        mm.apply_transform(trimesh.transformations.rotation_matrix(-a, [0, 0, 1]))
        g = _section2d(mm, 1, 0.0)
        if g is None:
            continue
        half = g.intersection(sbox(hub, -1e6, 1e6, 1e6))
        if half.is_empty:
            continue
        # pé de encaixe na base
        zb = half.bounds[1]
        xr = half.bounds[2]
        foot = sbox(hub, zb - t, min(xr, hub + (xr - hub) * 0.6), zb + 0.01)
        piece = unary_union([half, foot])
        out.append((f"radial {math.degrees(a):.0f}°", a, piece))
        feet.append((a, hub, min(xr, hub + (xr - hub) * 0.6)))
    base = Point(0, 0).buffer(R + 2 * t, quad_segs=48)
    slots = []
    for a, r0, r1 in feet:
        s = sbox(r0, -(t + 2 * tol) / 2, r1, (t + 2 * tol) / 2)
        slots.append(affinity.rotate(s, math.degrees(a), origin=(0, 0)))
    base = base.difference(unary_union(slots)) if slots else base
    return out, base


# --------------------------------------------------------------------------
# Divisão de formas 2D grandes em peças
# --------------------------------------------------------------------------
def grid_cut(geom, cell_w, cell_h, staggered=False, origin=None):
    x0, y0, x1, y1 = geom.bounds
    if origin:
        x0, y0 = origin
    pieces = []
    row = 0
    y = y0
    while y < y1:
        off = (cell_w / 2 if staggered and row % 2 else 0)
        x = x0 - off
        while x < x1:
            p = geom.intersection(sbox(x, y, x + cell_w, y + cell_h))
            if not p.is_empty and p.area > 1e-6:
                pieces.append(p)
            x += cell_w
        y += cell_h; row += 1
    return pieces


def radial_cut(geom, n=6, center=None, rotation=0.0):
    c = geom.centroid if center is None else Point(center)
    x0, y0, x1, y1 = geom.bounds
    R = 2 * max(x1 - x0, y1 - y0) + 10
    pieces = []
    for k in range(n):
        a0 = math.radians(rotation + 360 * k / n); a1 = math.radians(rotation + 360 * (k + 1) / n)
        ts = np.linspace(a0, a1, 32)
        sector = Polygon([(c.x, c.y)] + [(c.x + R * math.cos(t), c.y + R * math.sin(t)) for t in ts])
        p = geom.intersection(sector)
        if not p.is_empty and p.area > 1e-6:
            pieces.append(p)
    return pieces


def line_cut(geom, lines):
    """Divide por linhas desenhadas (corte 'natural' ao longo de linhas escolhidas)."""
    out = [geom]
    for ln in lines:
        nxt = []
        L = LineString(ln)
        # prolonga a linha para atravessar a forma
        (xa, ya), (xb, yb) = L.coords[0], L.coords[-1]
        d = np.array([xb - xa, yb - ya]); d /= (np.linalg.norm(d) or 1)
        big = 1e5
        L = LineString([(xa - d[0] * big, ya - d[1] * big), (xb + d[0] * big, yb + d[1] * big)])
        for g in out:
            polys = g.geoms if hasattr(g, "geoms") else [g]
            for p in polys:
                try:
                    nxt += list(split(p, L).geoms)
                except Exception:
                    nxt.append(p)
        out = nxt
    return [p for p in out if p.area > 1e-6]


def add_joints(pieces, kind="cauda de andorinha", size=6.0):
    """Acrescenta encaixes nas fronteiras partilhadas entre peças vizinhas (macho numa, fêmea noutra)."""
    from shapely.strtree import STRtree
    pieces = [p.buffer(0) for p in pieces]
    tree = STRtree(pieces)
    pairs = set()
    for i, p in enumerate(pieces):
        for j in tree.query(p.buffer(1e-3)):
            if j > i:
                pairs.add((i, int(j)))
    for i, j in sorted(pairs):
        shared = pieces[i].boundary.intersection(pieces[j].buffer(1e-3))
        if shared.is_empty or shared.length < 3 * size:
            continue
        mid = shared.interpolate(0.5, normalized=True) if hasattr(shared, "interpolate") else shared.centroid
        v = np.array([pieces[j].centroid.x - pieces[i].centroid.x, pieces[j].centroid.y - pieces[i].centroid.y])
        if np.linalg.norm(v) == 0:
            continue
        minx, miny, maxx, maxy = shared.bounds
        if (maxx - minx) > (maxy - miny):
            n = np.array([0, np.sign(v[1]) or 1.0]); tdir = np.array([1.0, 0])
        else:
            n = np.array([np.sign(v[0]) or 1.0, 0]); tdir = np.array([0, 1.0])
        m = np.array([mid.x, mid.y])
        if kind == "cauda de andorinha":
            tab = Polygon([m - tdir * size * 0.4, m + tdir * size * 0.4, m + tdir * size * 0.7 + n * size, m - tdir * size * 0.7 + n * size])
        else:  # botão redondo (puzzle)
            tab = Point(*(m + n * size * 0.7)).buffer(size * 0.5).union(
                Polygon([m - tdir * size * 0.25, m + tdir * size * 0.25, m + tdir * size * 0.25 + n * size * 0.5,
                         m - tdir * size * 0.25 + n * size * 0.5]))
        pieces[i] = pieces[i].union(tab).buffer(0)
        pieces[j] = pieces[j].difference(tab).buffer(0)
    return pieces


# --------------------------------------------------------------------------
# Pontes (micro-juntas) e entradas/saídas
# --------------------------------------------------------------------------
def add_bridges(paths, n=2, width=1.0, min_len=20.0):
    """Interrompe cada contorno fechado em n pontos (largura 'width') para a peça não cair da chapa."""
    out = []
    for p in paths:
        p = np.asarray(p)
        ls = LineString(p[:, :2])
        L = ls.length
        closed = np.allclose(p[0, :2], p[-1, :2])
        if not closed or L < min_len or n <= 0:
            out.append(p); continue
        gaps = [(L * (k + 0.5) / n - width / 2, L * (k + 0.5) / n + width / 2) for k in range(n)]
        starts = [g[1] for g in gaps]; ends = [g[0] for g in gaps[1:]] + [gaps[0][0] + L]
        for s, e in zip(starts, ends):
            ds = np.linspace(s, e, max(2, int((e - s) / 0.5)))
            pts = [ls.interpolate(d % L) for d in ds]
            out.append(np.array([(q.x, q.y) for q in pts]))
    return out


def add_lead(paths, length=2.0, mode="entrada e saída"):
    """Entrada/saída perpendicular ao contorno (evita marcas no ponto de início do corte)."""
    out = []
    for p in paths:
        p = np.asarray(p, float)
        if len(p) < 3 or not np.allclose(p[0, :2], p[-1, :2]):
            out.append(p); continue
        d = p[1, :2] - p[0, :2]; d /= (np.linalg.norm(d) or 1)
        nrm = np.array([d[1], -d[0]])
        # escolhe o lado de fora (afastado do centróide)
        c = p[:, :2].mean(0)
        if np.dot(p[0, :2] - c, nrm) < 0:
            nrm = -nrm
        seg = [p[:, :2]]
        if mode in ("entrada", "entrada e saída"):
            seg.insert(0, (p[0, :2] + nrm * length)[None])
        if mode in ("saída", "entrada e saída"):
            seg.append((p[-1, :2] + nrm * length)[None])
        out.append(np.vstack(seg))
    return out
