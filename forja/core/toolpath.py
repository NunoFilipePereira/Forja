"""Percursos de ferramenta 2D (contorno, bolsa, tramado) e exportação G-code."""
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely import affinity
from shapely.ops import unary_union
from .shapes2d import iter_paths


def contour(geom, tool_d, side="outside"):
    r = tool_d / 2
    if side == "outside":
        g = geom.buffer(r, join_style="round")
    elif side == "inside":
        g = geom.buffer(-r, join_style="round")
    else:
        g = geom
    return [p for p in iter_paths(g) if len(p) > 1]


def pocket(geom, tool_d, stepover=0.5):
    """Bolsa por offsets concêntricos (de fora para dentro)."""
    r = tool_d / 2
    step = tool_d * stepover
    paths = []
    g = geom.buffer(-r)
    while not g.is_empty:
        paths += [p for p in iter_paths(g) if len(p) > 1]
        g = g.buffer(-step)
    return paths


def hatch(geom, spacing=0.5, angle=0.0, zigzag=True):
    """Tramado (preenchimento por linhas) — típico em gravação laser."""
    c = geom.centroid
    g = affinity.rotate(geom, -angle, origin=c)
    x0, y0, x1, y1 = g.bounds
    paths = []
    flip = False
    for y in np.arange(y0 + spacing / 2, y1, spacing):
        seg = g.intersection(LineString([(x0 - 1, y), (x1 + 1, y)]))
        if seg.is_empty:
            continue
        segs = list(seg.geoms) if hasattr(seg, "geoms") else [seg]
        segs = [s for s in segs if s.geom_type == "LineString"]
        segs.sort(key=lambda s: s.bounds[0], reverse=flip)
        for s in segs:
            pts = np.array(s.coords)
            if flip:
                pts = pts[::-1]
            paths.append(pts)
        if zigzag:
            flip = not flip
    out = []
    for p in paths:
        ls = affinity.rotate(LineString(p), angle, origin=c)
        out.append(np.array(ls.coords))
    return out


def follow(geom):
    """Seguir a linha (gravação sobre o traço, sem compensação)."""
    return [p for p in iter_paths(geom) if len(p) > 1]


def path_length(paths):
    return sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() for p in paths if len(p) > 1)


def order_paths(paths):
    """Ordena caminhos pelo vizinho mais próximo para reduzir deslocações em vazio."""
    rest = list(paths)
    out = []
    cur = np.zeros(2)
    while rest:
        d = [np.linalg.norm(p[0, :2] - cur) for p in rest]
        i = int(np.argmin(d))
        p = rest.pop(i)
        out.append(p)
        cur = p[-1, :2]
    return out


def gcode(paths, mode="cnc", feed=800, plunge=300, safe_z=5.0, depth=1.0, passes=1,
          power=1000, spindle=12000, travel_feed=3000):
    """Gera G-code. mode='cnc' (fresa com Z) ou 'laser' (M3/M4 S potência)."""
    L = ["; Gerado por Forja", "G21 ; mm", "G90 ; absoluto", "G17"]
    paths = order_paths(paths)
    if mode == "laser":
        L += ["M5", f"G0 F{travel_feed}"]
        for _ in range(passes):
            for p in paths:
                L.append(f"G0 X{p[0, 0]:.3f} Y{p[0, 1]:.3f}")
                L.append(f"M4 S{power:g}")
                for x, y in p[1:, :2]:
                    L.append(f"G1 X{x:.3f} Y{y:.3f} F{feed:g}")
                L.append("M5")
        L += ["M5", "G0 X0 Y0", "M2"]
    else:
        L += [f"M3 S{spindle:g}", f"G0 Z{safe_z:.3f}"]
        zs = np.linspace(depth / passes, depth, passes)
        for p in paths:
            for z in zs:
                L.append(f"G0 X{p[0, 0]:.3f} Y{p[0, 1]:.3f}")
                L.append(f"G1 Z{-z:.3f} F{plunge:g}")
                for x, y in p[1:, :2]:
                    L.append(f"G1 X{x:.3f} Y{y:.3f} F{feed:g}")
                L.append(f"G0 Z{safe_z:.3f}")
        L += ["M5", "G0 X0 Y0", "M2"]
    return "\n".join(L) + "\n"


def estimate_time(paths, feed, passes=1, travel_feed=3000):
    cut = path_length(paths) * passes
    travel = 0.0
    cur = np.zeros(2)
    for p in order_paths(paths):
        travel += np.linalg.norm(p[0, :2] - cur)
        cur = p[-1, :2]
    return (cut / feed + travel * passes / travel_feed) * 60  # segundos
