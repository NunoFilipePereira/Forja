"""Litofanias completas: ajustes de imagem, formas (plana, arco, cilindro, esfera, recortes), moldura,
texto, furos, pé e orientação de impressão."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from scipy import ndimage
from shapely.geometry import Point, Polygon, box as sbox
from shapely import affinity
from shapely.ops import unary_union
from .meshops import to_manifold, from_manifold, heightmap_to_solid, drop_to_floor, center_on_plate

SHAPES = ["retângulo", "retângulo arredondado", "círculo", "oval", "coração", "hexágono", "estrela", "arco (curva)",
          "cilindro", "esfera"]


# --------------------------------------------------------------------------
# Imagem
# --------------------------------------------------------------------------
def prepare_image(gray, brightness=0.0, contrast=0.0, gamma=1.0, smooth=0.0, negative=False, mirror=False, rotate=0,
                  max_px=600):
    g = np.asarray(gray, float)
    if g.ndim == 3:
        g = g[..., :3] @ [0.299, 0.587, 0.114]
    if g.max() > 1.0:
        g = g / 255.0
    if rotate:
        g = np.rot90(g, -int(rotate) // 90)
    if mirror:
        g = g[:, ::-1]
    s = max(g.shape) / max_px
    if s > 1:
        g = ndimage.zoom(g, 1 / s, order=1)
    g = g + brightness / 100.0
    g = (g - 0.5) * (1 + contrast / 100.0) + 0.5
    g = np.clip(g, 0, 1) ** (1.0 / max(gamma, 1e-3))
    if smooth > 0:
        g = ndimage.gaussian_filter(g, smooth)
    if negative:
        g = 1 - g
    return np.clip(g, 0, 1)


def thickness_map(g, t_min, t_max):
    """Escuro = espesso (deixa passar menos luz)."""
    return t_min + (1 - g) * (t_max - t_min)


# --------------------------------------------------------------------------
# Formas de recorte
# --------------------------------------------------------------------------
def outline(shape, w, h, radius=5.0):
    cx, cy = w / 2, h / 2
    if shape in ("retângulo", "arco (curva)"):
        return sbox(0, 0, w, h)
    if shape == "retângulo arredondado":
        r = min(radius, w / 2 - 0.01, h / 2 - 0.01)
        return sbox(r, r, w - r, h - r).buffer(r, quad_segs=16)
    if shape == "círculo":
        return Point(cx, cy).buffer(min(w, h) / 2, quad_segs=64)
    if shape == "oval":
        return affinity.scale(Point(cx, cy).buffer(1, quad_segs=64), w / 2, h / 2)
    if shape == "coração":
        t = np.linspace(0, 2 * np.pi, 240)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        p = Polygon(np.c_[x, y])
        b = p.bounds
        p = affinity.scale(p, w / (b[2] - b[0]), h / (b[3] - b[1]), origin=(0, 0))
        b = p.bounds
        return affinity.translate(p, -b[0], -b[1])
    if shape == "hexágono":
        r = min(w, h) / 2
        return Polygon([(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(0, 360, 60)])
    if shape == "estrela":
        R = min(w, h) / 2
        pts = [(cx + (R if k % 2 == 0 else R * 0.5) * math.cos(math.radians(90 + 36 * k)),
                cy + (R if k % 2 == 0 else R * 0.5) * math.sin(math.radians(90 + 36 * k))) for k in range(10)]
        return Polygon(pts)
    return sbox(0, 0, w, h)


def _extrude(geom, h, z0=0.0):
    from .generators import cs
    return mf.Manifold.extrude(cs(geom), h).translate((0, 0, z0))


# --------------------------------------------------------------------------
# Construção
# --------------------------------------------------------------------------
def lithophane(gray, shape="retângulo", width=100.0, height=0.0, t_min=0.8, t_max=3.0, px_mm=0.2,
               brightness=0.0, contrast=0.0, gamma=1.0, smooth=0.6, negative=False, mirror=False, rotate=0,
               border=4.0, border_t=0.0, corner_r=6.0, arc_angle=120.0, sphere_angle=150.0, base_ring=3.0,
               text="", text_size=5.0, hole_d=0.0, hanger=False, stand=False, stand_depth=25.0, stand_h=4.0,
               vertical=False):
    """Devolve a malha da litofania com todas as opções."""
    if t_max <= t_min:
        raise ValueError("A espessura máxima tem de ser maior do que a mínima.")
    max_px = int(min(900, max(80, width / max(px_mm, 0.05))))
    g = prepare_image(gray, brightness, contrast, gamma, smooth, negative, mirror, rotate, max_px)
    rows, cols = g.shape
    w = width
    h = height if height > 0 else width * rows / cols
    if height > 0:  # ajusta a imagem à proporção pedida
        g = ndimage.zoom(g, (int(round(cols * h / w)) / rows, 1), order=1)
        rows = g.shape[0]
    T = thickness_map(g, t_min, t_max)
    bt = border_t if border_t > 0 else t_max + 0.4

    if shape == "cilindro":
        from .generators import _tube
        R = w / (2 * math.pi)
        H = T - t_min
        m = _tube(H + t_min, max(R - t_min, 1.0), h)
        M = to_manifold(m)
        if border > 0:
            Ro = max(R - t_min, 1.0) + bt
            Ri = max(R - t_min, 1.0)
            ov = 0.4  # sobreposição com o tubo para a união ficar bem fechada
            ring = mf.Manifold.cylinder(border + ov, Ro, Ro, 128) - mf.Manifold.cylinder(border + ov, Ri - 0.3, Ri - 0.3, 128)
            M = M.translate((0, 0, border)) + ring + ring.translate((0, 0, h + border - ov))
        if base_ring > 0:
            Ro = max(R - t_min, 1.0) + bt + 1
            M = M + (mf.Manifold.cylinder(base_ring, Ro, Ro, 128) - mf.Manifold.cylinder(base_ring, Ro - bt - 3, Ro - bt - 3, 128)).translate((0, 0, -base_ring + 0.01))
        return drop_to_floor(from_manifold(M))

    if shape == "esfera":
        from .generators import sphere_lithophane
        R = w / math.radians(max(sphere_angle, 10)) * 1.0
        return sphere_lithophane(np.clip((1 - (T - t_min) / (t_max - t_min)) * 255, 0, 255), diameter=2 * R,
                                 t_min=t_min, t_max=t_max, max_px=max_px, opening=2 * R * math.sin(math.radians(min(70, sphere_angle / 4))))

    # --- plana (e arco, que é uma plana dobrada) -----------------------------------
    flat = heightmap_to_solid(T - t_min, w, h, base=t_min)
    M = to_manifold(flat)
    out = outline(shape, w, h, corner_r)
    M = M ^ _extrude(out, t_max + 1, -0.5)
    if border > 0:
        frame = out.buffer(border, join_style="round" if shape != "retângulo" and shape != "arco (curva)" else "mitre").difference(out.buffer(-0.05))
        M = M + _extrude(frame, bt)
        outer = out.buffer(border, join_style="mitre" if shape in ("retângulo", "arco (curva)") else "round")
    else:
        outer = out
    ob = outer.bounds
    if text and border >= text_size * 0.6:
        from .text3d import text_polygons
        tp = text_polygons(text, size_mm=min(text_size, border * 0.85), bold=True)
        tb = tp.bounds
        tp = affinity.translate(tp, (ob[0] + ob[2]) / 2 - (tb[0] + tb[2]) / 2, ob[1] + border / 2 - (tb[1] + tb[3]) / 2)
        M = M + _extrude(tp, 0.8, bt - 0.01)
    if hole_d > 0 or hanger:
        d = hole_d if hole_d > 0 else 6.0
        ring_w = 2.5 if not hanger else 4.0
        cx, cy = (ob[0] + ob[2]) / 2, ob[3] + d / 2 + ring_w - 1.0
        tab = Point(cx, cy).buffer(d / 2 + ring_w, quad_segs=32).union(sbox(cx - d / 2 - ring_w, ob[3] - 2, cx + d / 2 + ring_w, cy))
        M = (M + _extrude(tab, bt)) - _extrude(Point(cx, cy).buffer(d / 2, quad_segs=32), bt * 3, -bt)
    if stand:
        # pé em T por baixo da aresta inferior: faz a litofania ficar de pé
        foot = sbox(ob[0], ob[1] - stand_h, ob[2], ob[1] + 0.01)
        F = _extrude(foot, stand_depth, -(stand_depth - bt) / 2)
        M = M + F
    m = from_manifold(M)
    if shape == "arco (curva)" and arc_angle > 1:
        m = bend(m, arc_angle)
    if vertical or stand:
        # de pé: a altura da imagem passa para Z
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    return center_on_plate(m)


def bend(mesh, angle_deg):
    """Dobra a peça plana em arco à volta de um eixo paralelo a Y (concavidade para trás)."""
    lo, hi = mesh.bounds
    W = hi[0] - lo[0]
    R = W / math.radians(angle_deg)
    M = to_manifold(mesh).refine_to_length(max(W / 100, 0.6))
    m = from_manifold(M)
    V = m.vertices.copy()
    x = V[:, 0] - (lo[0] + hi[0]) / 2
    z = V[:, 2]
    a = x / R
    r = R + z
    V[:, 0] = r * np.sin(a)
    V[:, 2] = r * np.cos(a) - R
    out = trimesh.Trimesh(V, m.faces, process=True)
    if out.volume < 0:
        out.invert()
    return out
