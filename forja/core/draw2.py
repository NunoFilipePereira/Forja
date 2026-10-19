"""Desenho 2D avançado: padrões de recorte, meio-tom vetorial, fotografia → vetor por níveis, formas básicas."""
import math
import numpy as np
from shapely.geometry import Polygon, Point, box as sbox, LineString
from shapely import affinity
from shapely.ops import unary_union

PATTERNS = ["hexágonos", "quadrados", "losangos", "círculos", "triângulos", "ranhuras"]


def pattern_cells(kind, bounds, cell=10.0, wall=2.0, angle=0.0):
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    R = math.hypot(x1 - x0, y1 - y0) / 2 + cell * 2
    cells = []
    if kind == "hexágonos":
        r = cell / 2
        dx, dy = 1.5 * r + wall * 0.866, math.sqrt(3) * r + wall
        hexa = Polygon([(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in range(0, 360, 60)])
        i = 0
        x = -R
        while x < R:
            y = -R + (dy / 2 if i % 2 else 0)
            while y < R:
                cells.append(affinity.translate(hexa, x, y)); y += dy
            x += dx; i += 1
    elif kind in ("quadrados", "losangos"):
        step = cell + wall
        sq = sbox(-cell / 2, -cell / 2, cell / 2, cell / 2)
        if kind == "losangos":
            sq = affinity.rotate(sq, 45)
            step = cell * 1.4142 / 2 + wall
            off = True
        else:
            off = False
        j = 0
        y = -R
        while y < R:
            x = -R + (step / 2 if off and j % 2 else 0)
            while x < R:
                cells.append(affinity.translate(sq, x, y)); x += step if not off else step * 2
            y += step if not off else step
            j += 1
    elif kind == "círculos":
        step = cell + wall
        c = Point(0, 0).buffer(cell / 2, quad_segs=12)
        j = 0
        y = -R
        while y < R:
            x = -R + (step / 2 if j % 2 else 0)
            while x < R:
                cells.append(affinity.translate(c, x, y)); x += step
            y += step * 0.866; j += 1
    elif kind == "triângulos":
        h = cell * 0.866
        up = Polygon([(-cell / 2, -h / 3), (cell / 2, -h / 3), (0, 2 * h / 3)])
        down = affinity.rotate(up, 180, origin=(0, 0))
        sx, sy = (cell + wall * 1.7) / 2, h + wall
        j = 0
        y = -R
        while y < R:
            x = -R
            k = 0
            while x < R:
                t = up if (k + j) % 2 == 0 else down
                cells.append(affinity.translate(affinity.scale(t, 1, 1), x, y + (0 if t is up else h / 3)))
                x += sx; k += 1
            y += sy; j += 1
    else:  # ranhuras
        step = wall + cell / 3
        y = -R
        while y < R:
            cells.append(sbox(-R, y, R, y + cell / 3)); y += step
    return [affinity.translate(affinity.rotate(c, angle, origin=(0, 0)), cx, cy) for c in cells]


def pattern_fill(geom, kind="hexágonos", cell=10.0, wall=2.0, border=4.0, angle=0.0):
    """Recorta um padrão dentro da forma, deixando uma moldura de 'border' mm (peças leves para laser)."""
    inner = geom.buffer(-border)
    if inner.is_empty:
        raise ValueError("A moldura é maior do que a forma.")
    holes = [c.intersection(inner) for c in pattern_cells(kind, inner.bounds, cell, wall, angle)]
    holes = [h for h in holes if not h.is_empty and h.area > (cell * cell) * 0.05]
    return geom.difference(unary_union(holes)).buffer(0), len(holes)


def vector_halftone(gray, width_mm=100.0, cell=2.0, kind="pontos", min_frac=0.05, max_frac=0.95):
    """Meio-tom em vetores (círculos ou linhas de espessura variável) para gravação laser."""
    g = np.asarray(gray, float)
    if g.ndim == 3:
        g = g.mean(2)
    if g.max() > 1:
        g = g / 255.0
    rows, cols = g.shape
    px = width_mm / cols
    nx = int(width_mm / cell); ny = int(rows * px / cell)
    out = []
    for j in range(ny):
        for i in range(nx):
            r0, r1 = int(j * cell / px), int((j + 1) * cell / px)
            c0, c1 = int(i * cell / px), int((i + 1) * cell / px)
            v = 1 - g[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)].mean()
            if v < min_frac:
                continue
            v = min(v, max_frac)
            cx, cy = (i + 0.5) * cell, (ny - j - 0.5) * cell
            if kind == "pontos":
                out.append(Point(cx, cy).buffer(cell / 2 * math.sqrt(v), quad_segs=6))
            else:
                hgt = cell * v
                out.append(sbox(i * cell, cy - hgt / 2, (i + 1) * cell, cy + hgt / 2))
    return [unary_union(out)] if out else []


def photo_to_vector(gray, levels=4, px_mm=0.25, min_area_px=30, simplify_px=0.8, smooth=1.0):
    """Fotografia → contornos por níveis de cinzento (posterização), do mais escuro ao mais claro."""
    import cv2
    from scipy import ndimage
    g = np.asarray(gray, float)
    if g.ndim == 3:
        g = g.mean(2)
    if smooth:
        g = ndimage.gaussian_filter(g, smooth)
    lo, hi = np.percentile(g, [1, 99])
    thresholds = np.linspace(lo, hi, levels + 1)[1:-1] if levels > 1 else [(lo + hi) / 2]
    h = g.shape[0]
    res = []
    for k, t in enumerate(sorted(thresholds)):
        b = (g <= t).astype(np.uint8)
        cs, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        polys = []
        if hier is not None:
            hier = hier[0]
            for i, c in enumerate(cs):
                if hier[i][3] != -1 or cv2.contourArea(c) < min_area_px or len(c) < 3:
                    continue
                ext = [(x * px_mm, (h - y) * px_mm) for x, y in c[:, 0, :]]
                holes = []
                j = hier[i][2]
                while j != -1:
                    if cv2.contourArea(cs[j]) >= min_area_px and len(cs[j]) >= 3:
                        holes.append([(x * px_mm, (h - y) * px_mm) for x, y in cs[j][:, 0, :]])
                    j = hier[j][0]
                p = Polygon(ext, holes).buffer(0).simplify(simplify_px * px_mm)
                if not p.is_empty:
                    polys.append(p)
        if polys:
            res.append((k + 1, float(t), unary_union(polys)))
    return res
