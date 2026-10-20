"""Formas 2D, formato de texto de dados, importação/exportação DXF e SVG."""
import itertools
import math
import numpy as np
from shapely.geometry import (Point, LineString, Polygon, MultiPolygon, box, mapping,
                              GeometryCollection)
from shapely import affinity
from shapely.ops import unary_union

KINDS = {
    "pt": "Ponto", "ln": "Linha", "rc": "Retângulo", "rcr": "Retângulo rodado", "cir": "Círculo",
    "ell": "Elipse", "pol": "Polígono", "pl": "Polilinha", "spl": "Spline", "arr": "Seta",
    "txt": "Texto", "geo": "Geometria", "arc": "Arco", "fan": "Leque", "ff": "Mão livre",
}


class Shape:
    _ids = itertools.count(1)

    def __init__(self, kind, params, geom=None, name=None):
        self.id = next(Shape._ids)
        self.kind = kind
        self.params = params
        self.geom = geom if geom is not None else build(kind, params)
        self.name = name or f"{KINDS.get(kind, kind)} {self.id}"
        self.group = None
        self.hidden = False
        # estilo: traço, preenchimento, espessura (px), tipo de linha, setas
        self.style = {"stroke": None, "fill": None, "width": None, "dash": "contínua", "arrow_start": False, "arrow_end": False}

    def copy(self):
        s = Shape(self.kind, dict(self.params), self.geom, self.name)
        s.group = self.group; s.hidden = self.hidden; s.style = dict(self.style)
        return s

    @property
    def closed(self):
        return self.geom.geom_type in ("Polygon", "MultiPolygon")


def catmull_rom(points, samples=16, closed=False):
    P = np.asarray(points, float)
    if len(P) < 3:
        return P
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def arc_points(x1, y1, x2, y2, x3, y3, n=64):
    """Arco que passa por 3 pontos (início, ponto no arco, fim)."""
    A = np.array([[x1, y1], [x2, y2], [x3, y3]], float)
    (ax, ay), (bx, by), (cx, cy) = A
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return [(x1, y1), (x3, y3)]
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    a1, a2, a3 = (math.atan2(y - uy, x - ux) for x, y in A)
    def ccw(a, b):
        return (b - a) % (2 * math.pi)
    # sentido que passa pelo ponto intermédio
    if ccw(a1, a2) <= ccw(a1, a3):
        ts = a1 + np.linspace(0, ccw(a1, a3), n)
    else:
        ts = a1 - np.linspace(0, ccw(a3, a1), n)
    return [(ux + r * math.cos(t), uy + r * math.sin(t)) for t in ts]


def curve(kind, x0=0, y0=0, length=100, amplitude=10, period=20, turns=5, r0=2):
    """Curvas paramétricas: espiral, onda (seno), dente de serra, onda quadrada, triangular."""
    if kind == "espiral":
        t = np.linspace(0, turns * 2 * np.pi, int(turns * 90))
        r = r0 + (length / 2 - r0) * t / t[-1]
        return list(zip(x0 + r * np.cos(t), y0 + r * np.sin(t)))
    x = np.linspace(0, length, max(50, int(length / period * 40)))
    ph = (x / period) % 1
    y = {"onda": np.sin(2 * np.pi * x / period), "dente de serra": 2 * ph - 1,
         "onda quadrada": np.where(ph < 0.5, 1.0, -1.0), "triangular": 1 - 4 * np.abs(ph - 0.5)}[kind] * amplitude
    if kind == "onda quadrada":
        xs, ys = [], []
        for i in range(len(x) - 1):
            xs += [x[i], x[i + 1]]; ys += [y[i], y[i]]
        x, y = np.array(xs), np.array(ys)
    return list(zip(x0 + x, y0 + y))


def build(kind, p):
    if kind == "pt":
        return Point(p["x"], p["y"])
    if kind == "ln":
        return LineString([(p["x1"], p["y1"]), (p["x2"], p["y2"])])
    if kind == "rc":
        return box(p["left"], p["bottom"], p["right"], p["top"])
    if kind == "rcr":
        w, h = p["w"], p["h"]
        g = box(-w / 2, -h / 2, w / 2, h / 2)
        return affinity.translate(affinity.rotate(g, p["angle"]), p["cx"], p["cy"])
    if kind == "cir":
        return Point(p["cx"], p["cy"]).buffer(p["r"], quad_segs=32)
    if kind == "ell":
        g = Point(0, 0).buffer(1, quad_segs=32)
        g = affinity.scale(g, p["rx"], p["ry"])
        return affinity.translate(affinity.rotate(g, p.get("angle", 0)), p["cx"], p["cy"])
    if kind == "pol":
        return Polygon(p["pts"]).buffer(0)
    if kind == "pl":
        return LineString(p["pts"])
    if kind == "spl":
        pts = catmull_rom(p["pts"], closed=p.get("closed", False))
        return Polygon(pts).buffer(0) if p.get("closed") else LineString(pts)
    if kind == "arr":
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        L = math.hypot(x2 - x1, y2 - y1) or 1
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        hs = p.get("head", max(2.0, L * 0.15))
        w = p.get("width", hs * 0.35)
        bx, by = x2 - ux * hs, y2 - uy * hs
        nx, ny = -uy, ux
        pts = [(x1 + nx * w / 2, y1 + ny * w / 2), (bx + nx * w / 2, by + ny * w / 2),
               (bx + nx * hs / 2, by + ny * hs / 2), (x2, y2), (bx - nx * hs / 2, by - ny * hs / 2),
               (bx - nx * w / 2, by - ny * w / 2), (x1 - nx * w / 2, y1 - ny * w / 2)]
        return Polygon(pts)
    if kind == "arc":
        return LineString(arc_points(p["x1"], p["y1"], p["x2"], p["y2"], p["x3"], p["y3"]))
    if kind == "fan":
        a0, a1 = math.radians(p["a0"]), math.radians(p["a1"])
        t = np.linspace(a0, a1, max(8, int(abs(a1 - a0) / math.radians(2))))
        outer = [(p["cx"] + p["r2"] * math.cos(a), p["cy"] + p["r2"] * math.sin(a)) for a in t]
        inner = [(p["cx"] + p["r1"] * math.cos(a), p["cy"] + p["r1"] * math.sin(a)) for a in t[::-1]] if p["r1"] > 0 else [(p["cx"], p["cy"])]
        return Polygon(outer + inner).buffer(0)
    if kind == "ff":
        return LineString(p["pts"]).simplify(p.get("tol", 0.2))
    if kind == "txt":
        from .text3d import text_polygons
        g = text_polygons(p["text"], p.get("font", "Helvetica"), p.get("size", 10), p.get("bold", False))
        if p.get("angle"):
            g = affinity.rotate(g, p["angle"], origin=(0, 0))
        return affinity.translate(g, p["x"], p["y"])
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Formato de dados em texto
# --------------------------------------------------------------------------
DATA_HELP = """Formato (uma forma por linha, valores separados por espaço):
pt  x y
ln  x1 y1 x2 y2
rc  left bottom right top
rcr cx cy w h angle
cir cx cy r
ell cx cy rx ry [angle]
pol x1 y1 x2 y2 x3 y3 ...
pl  x1 y1 x2 y2 ...
arr x1 y1 x2 y2
txt x y tamanho texto...
"""


def to_data(shapes):
    lines = []
    for s in shapes:
        p, k = s.params, s.kind
        f = lambda *v: " ".join(f"{x:.4g}" for x in v)
        if k == "pt": lines.append(f"pt {f(p['x'], p['y'])}")
        elif k == "ln": lines.append(f"ln {f(p['x1'], p['y1'], p['x2'], p['y2'])}")
        elif k == "rc": lines.append(f"rc {f(p['left'], p['bottom'], p['right'], p['top'])}")
        elif k == "rcr": lines.append(f"rcr {f(p['cx'], p['cy'], p['w'], p['h'], p['angle'])}")
        elif k == "cir": lines.append(f"cir {f(p['cx'], p['cy'], p['r'])}")
        elif k == "ell": lines.append(f"ell {f(p['cx'], p['cy'], p['rx'], p['ry'], p.get('angle', 0))}")
        elif k in ("pol", "pl"): lines.append(k + " " + f(*np.ravel(p["pts"])))
        elif k == "arr": lines.append(f"arr {f(p['x1'], p['y1'], p['x2'], p['y2'])}")
        elif k == "txt": lines.append(f"txt {f(p['x'], p['y'], p.get('size', 10))} {p['text']}")
        else:
            g = s.geom
            if g.geom_type == "Polygon":
                lines.append("pol " + f(*np.ravel(np.array(g.exterior.coords)[:-1])))
    return "\n".join(lines)


def to_json(shapes):
    """Serialização completa (geometria em WKT + parâmetros + estilo) para projetos."""
    import json
    out = []
    for sh in shapes:
        params = {k: (list(map(list, v)) if k == "pts" else v) for k, v in sh.params.items()}
        out.append({"kind": sh.kind, "params": params, "wkt": sh.geom.wkt, "name": sh.name, "group": sh.group,
                    "hidden": sh.hidden, "style": sh.style})
    return json.dumps(out, ensure_ascii=False)


def from_json(text):
    import json
    from shapely import wkt
    shapes = []
    for d in json.loads(text):
        params = d.get("params", {})
        if "pts" in params:
            params["pts"] = [tuple(p) for p in params["pts"]]
        sh = Shape(d["kind"], params, wkt.loads(d["wkt"]), d.get("name"))
        sh.group = d.get("group"); sh.hidden = d.get("hidden", False); sh.style.update(d.get("style") or {})
        shapes.append(sh)
    return shapes


def from_data(text):
    shapes = []
    for raw in text.splitlines():
        t = raw.strip().split()
        if not t or t[0].startswith("#"):
            continue
        k = t[0].lower()
        if k == "txt":
            x, y, size = map(float, t[1:4])
            shapes.append(Shape("txt", {"x": x, "y": y, "size": size, "text": " ".join(t[4:])}))
            continue
        v = list(map(float, t[1:]))
        if k == "pt": p = dict(x=v[0], y=v[1])
        elif k in ("ln", "arr"): p = dict(x1=v[0], y1=v[1], x2=v[2], y2=v[3])
        elif k == "rc": p = dict(left=v[0], bottom=v[1], right=v[2], top=v[3])
        elif k == "rcr": p = dict(cx=v[0], cy=v[1], w=v[2], h=v[3], angle=v[4])
        elif k == "cir": p = dict(cx=v[0], cy=v[1], r=v[2])
        elif k == "ell": p = dict(cx=v[0], cy=v[1], rx=v[2], ry=v[3], angle=v[4] if len(v) > 4 else 0)
        elif k in ("pol", "pl", "spl"): p = dict(pts=[(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)])
        else:
            raise ValueError(f"Tipo desconhecido: {k}")
        shapes.append(Shape(k, p))
    return shapes


# --------------------------------------------------------------------------
# Transformações e operações
# --------------------------------------------------------------------------
def transformed(s, fn, kind="geo"):
    n = Shape(kind, {}, fn(s.geom), s.name)
    n.group = s.group; n.hidden = s.hidden; n.style = dict(s.style)
    return n


def move(s, dx, dy): return transformed(s, lambda g: affinity.translate(g, dx, dy))
def rotate(s, a, origin="centroid"): return transformed(s, lambda g: affinity.rotate(g, a, origin=origin))
def scale(s, k): return transformed(s, lambda g: affinity.scale(g, k, k, origin="centroid"))
def mirror_x(s): return transformed(s, lambda g: affinity.scale(g, -1, 1, origin="centroid"))
def offset(s, d): return transformed(s, lambda g: g.buffer(d, join_style="round"))


def boolean(shapes, op):
    gs = [s.geom for s in shapes if s.closed]
    if len(gs) < 2:
        raise ValueError("Selecione pelo menos duas formas fechadas.")
    if op == "union":
        g = unary_union(gs)
    elif op == "difference":
        g = gs[0].difference(unary_union(gs[1:]))
    elif op == "intersection":
        g = gs[0]
        for o in gs[1:]:
            g = g.intersection(o)
    elif op == "xor":
        g = gs[0]
        for o in gs[1:]:
            g = g.symmetric_difference(o)
    return Shape("geo", {}, g)


def align(shapes, mode):
    if not shapes:
        return shapes
    b = np.array([s.geom.bounds for s in shapes])
    out = []
    for s, (x0, y0, x1, y1) in zip(shapes, b):
        dx = dy = 0
        if mode == "left": dx = b[:, 0].min() - x0
        elif mode == "right": dx = b[:, 2].max() - x1
        elif mode == "bottom": dy = b[:, 1].min() - y0
        elif mode == "top": dy = b[:, 3].max() - y1
        elif mode == "hcenter": dx = (b[:, 0].min() + b[:, 2].max()) / 2 - (x0 + x1) / 2
        elif mode == "vcenter": dy = (b[:, 1].min() + b[:, 3].max()) / 2 - (y0 + y1) / 2
        out.append(move(s, dx, dy))
    return out


def shape_info(s):
    g = s.geom
    x0, y0, x1, y1 = g.bounds
    info = {"Tipo": KINDS.get(s.kind, s.kind), "Caixa": f"{x1 - x0:.3f} × {y1 - y0:.3f}",
            "Mín": f"({x0:.3f}, {y0:.3f})", "Máx": f"({x1:.3f}, {y1:.3f})"}
    if s.closed:
        info["Área"] = f"{g.area:.3f}"
        info["Perímetro"] = f"{g.length:.3f}"
        c = g.centroid
        info["Centróide"] = f"({c.x:.3f}, {c.y:.3f})"
    elif g.geom_type != "Point":
        info["Comprimento"] = f"{g.length:.3f}"
    return info


# --------------------------------------------------------------------------
# Geometria → listas de caminhos
# --------------------------------------------------------------------------
def iter_paths(g):
    """Devolve listas de coordenadas (contornos fechados e linhas)."""
    t = g.geom_type
    if t == "Polygon":
        yield np.array(g.exterior.coords)
        for i in g.interiors:
            yield np.array(i.coords)
    elif t in ("LineString", "LinearRing"):
        yield np.array(g.coords)
    elif t == "Point":
        return
    elif hasattr(g, "geoms"):
        for sub in g.geoms:
            yield from iter_paths(sub)


def export_svg(shapes, path, stroke="#000", extra_paths=()):
    allb = [s.geom.bounds for s in shapes] + [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
                                              for p in extra_paths if len(p)]
    if not allb:
        raise ValueError("Nada para exportar.")
    b = np.array(allb)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    W, H = x1 - x0, y1 - y0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.3f}mm" height="{H:.3f}mm" '
           f'viewBox="0 0 {W:.3f} {H:.3f}">']

    def d(p):
        return "M " + " L ".join(f"{x - x0:.3f},{y1 - y:.3f}" for x, y in p)

    dashes = {"contínua": "", "tracejada": "2,1", "pontilhada": "0.4,0.8", "traço-ponto": "2,0.8,0.4,0.8", "traço longo": "5,1.5"}
    for s in shapes:
        if getattr(s, "hidden", False):
            continue
        ps = list(iter_paths(s.geom))
        st = getattr(s, "style", {}) or {}
        if ps:
            da = dashes.get(st.get("dash", "contínua"), "")
            out.append(f'<path d="{" ".join(d(p) + (" Z" if s.closed else "") for p in ps)}" '
                       f'fill="{st.get("fill") or "none"}" fill-rule="evenodd" stroke="{st.get("stroke") or stroke}" '
                       f'stroke-width="{0.1 * (st.get("width") or 1):g}"' + (f' stroke-dasharray="{da}"' if da else "") + "/>")
    for p in extra_paths:
        out.append(f'<path d="{d(p)}" fill="none" stroke="#c00" stroke-width="0.1"/>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


def export_eps(shapes, path):
    b = np.array([s.geom.bounds for s in shapes])
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    k = 72 / 25.4  # mm → pontos
    out = ["%!PS-Adobe-3.0 EPSF-3.0", f"%%BoundingBox: 0 0 {int((x1 - x0) * k) + 1} {int((y1 - y0) * k) + 1}",
           "%%Creator: Forja 3D", "0.3 setlinewidth"]
    for s in shapes:
        for p in iter_paths(s.geom):
            pts = [((x - x0) * k, (y - y0) * k) for x, y in p[:, :2]]
            out.append(f"newpath {pts[0][0]:.3f} {pts[0][1]:.3f} moveto")
            out += [f"{x:.3f} {y:.3f} lineto" for x, y in pts[1:]]
            out.append("closepath stroke" if s.closed else "stroke")
    out.append("showpage\n%%EOF")
    open(path, "w").write("\n".join(out))


def export_dxf(shapes, path, extra_paths=()):
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for s in shapes:
        if s.geom.geom_type == "Point":
            msp.add_point((s.geom.x, s.geom.y))
            continue
        for p in iter_paths(s.geom):
            closed = s.closed
            pts = p[:-1] if closed and np.allclose(p[0], p[-1]) else p
            msp.add_lwpolyline([tuple(q[:2]) for q in pts], close=closed)
    if extra_paths:
        doc.layers.add("TOOLPATH", color=1)
        for p in extra_paths:
            msp.add_lwpolyline([tuple(q[:2]) for q in p], dxfattribs={"layer": "TOOLPATH"})
    doc.saveas(path)


def import_dxf(path):
    import ezdxf
    from ezdxf import path as ezpath
    doc = ezdxf.readfile(path)
    shapes = []
    for e in doc.modelspace():
        try:
            if e.dxftype() == "POINT":
                shapes.append(Shape("pt", {"x": e.dxf.location.x, "y": e.dxf.location.y}))
                continue
            if e.dxftype() == "CIRCLE":
                c = e.dxf.center
                shapes.append(Shape("cir", {"cx": c.x, "cy": c.y, "r": e.dxf.radius}))
                continue
            p = ezpath.make_path(e)
            pts = [(v.x, v.y) for v in p.flattening(0.05)]
            if len(pts) < 2:
                continue
            closed = np.allclose(pts[0], pts[-1]) or getattr(e, "closed", False) or getattr(e, "is_closed", False)
            if closed and len(pts) >= 3:
                shapes.append(Shape("pol", {"pts": pts}))
            else:
                shapes.append(Shape("pl", {"pts": pts}))
        except Exception:
            continue
    return shapes


def import_svg(path):
    from svgpathtools import svg2paths
    paths, _ = svg2paths(path)
    shapes = []
    for p in paths:
        for sub in p.continuous_subpaths():
            n = max(8, int(sub.length() / 0.5))
            pts = [(sub.point(t).real, -sub.point(t).imag) for t in np.linspace(0, 1, n)]
            if sub.isclosed() and len(pts) >= 3:
                shapes.append(Shape("pol", {"pts": pts}))
            else:
                shapes.append(Shape("pl", {"pts": pts}))
    return shapes
