"""Utilitários comuns às ferramentas "Criar" (litofanias, Photo magic, projeções, objetos):
fontes de imagem (ficheiro, imagem atual da Forja ou exemplo), silhuetas vetoriais, texto, quantização de cores,
formas regulares e construção de sólidos fechados a partir de grelhas de pontos."""
import math
import os
import numpy as np
import trimesh
import manifold3d as mf
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon, Point, box as sbox
from shapely import affinity
from shapely.ops import unary_union

# imagem atual da Forja (definida pela janela quando se abre a ferramenta)
CURRENT = {"rgb": None}

SHAPES = ["Triangle", "Rectangle", "Square", "Pentagon", "Hexagon", "Octagon", "Circle"]


# --------------------------------------------------------------------------
# Imagens
# --------------------------------------------------------------------------
def demo_rgb(n=240):
    """Imagem de exemplo (céu, sol, montanhas e um lago) para pré-visualizar sem fotografia."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.zeros((n, n, 3))
    img[..., 0] = 0.35 + 0.5 * y; img[..., 1] = 0.55 + 0.3 * y; img[..., 2] = 0.95 - 0.2 * y
    sun = (x - 0.72) ** 2 + (y - 0.25) ** 2 < 0.012
    img[sun] = (1.0, 0.85, 0.2)
    mount = y > 0.55 + 0.25 * np.abs(np.sin(x * 5.5)) - 0.12 * x
    img[mount] = np.stack([0.25 + 0.2 * y, 0.45 + 0.1 * y, 0.2 + 0.0 * y], -1)[mount]
    snow = mount & (y < 0.62 + 0.25 * np.abs(np.sin(x * 5.5)) - 0.12 * x)
    img[snow] = (0.95, 0.95, 0.97)
    lake = y > 0.85
    img[lake] = np.stack([0.15 + 0 * x, 0.35 + 0.1 * np.sin(x * 40), 0.6 + 0 * x], -1)[lake]
    return np.clip(img, 0, 1)


def load_rgb(path="", max_px=400):
    """RGB em [0, 1]. Sem caminho: imagem atual da Forja; sem imagem: exemplo."""
    img = None
    if path and os.path.exists(path):
        import cv2
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is not None:
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            elif a.shape[2] == 4:
                alpha = a[..., 3:4] / 255.0
                a = (a[..., :3] * alpha + 255 * (1 - alpha))
            img = a[..., :3][..., ::-1].astype(float)
            img /= 65535.0 if img.max() > 255 else 255.0
    if img is None and CURRENT.get("rgb") is not None:
        img = np.asarray(CURRENT["rgb"], float)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        if img.max() > 1:
            img = img / 255.0
    if img is None:
        img = demo_rgb()
    s = max(img.shape[:2]) / max_px
    if s > 1:
        img = ndimage.zoom(img, (1 / s, 1 / s, 1), order=1)
    return np.clip(img, 0, 1)


def to_gray(rgb):
    return np.clip(np.asarray(rgb)[..., :3] @ [0.299, 0.587, 0.114], 0, 1)


def load_gray(path="", max_px=400):
    return to_gray(load_rgb(path, max_px))


def resample(g, rows, cols):
    """Reamostra uma matriz (2D ou 3D) para rows × cols."""
    g = np.asarray(g, float)
    z = (rows / g.shape[0], cols / g.shape[1]) + ((1,) if g.ndim == 3 else ())
    return ndimage.zoom(g, z, order=1)[:rows, :cols]


def grid_for(g, W, H, res):
    """Número de linhas/colunas para cobrir W × H mm com passo res (limitado)."""
    cols = int(np.clip(W / max(res, 0.05), 8, 700))
    rows = int(np.clip(H / max(res, 0.05), 8, 700))
    return rows, cols


def litho_T(g, t_min, t_max, positive=True):
    """Espessura da litofania: positivo = zonas escuras mais espessas."""
    g = np.clip(g, 0, 1)
    return t_min + ((1 - g) if positive else g) * (t_max - t_min)


def quantize(rgb, n, auto=True, user=None):
    """Reduz a imagem a n cores. Devolve (rótulos, paleta) com a paleta ordenada do mais claro para o mais escuro."""
    import cv2
    a = np.asarray(rgb, np.float32).reshape(-1, 3)
    if not auto and user:
        pal = np.array(user[:n], np.float32)
        lab = np.argmin(((a[:, None, :] - pal[None]) ** 2).sum(-1), 1)
    else:
        n = max(1, min(int(n), 16))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
        _, lab, pal = cv2.kmeans(a, n, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
    order = np.argsort(-(pal @ [0.299, 0.587, 0.114]))
    remap = np.empty_like(order); remap[order] = np.arange(len(order))
    return remap[lab].reshape(rgb.shape[:2]), np.asarray(pal)[order]


# --------------------------------------------------------------------------
# Silhuetas (vetoriais)
# --------------------------------------------------------------------------
def mask_to_geom(mask, simplify=0.8):
    """Máscara binária (linha 0 em cima) → (Multi)Polygon em píxeis (y para cima)."""
    import cv2
    m = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    m = np.pad(m, 2)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return Polygon()
    hier = hier[0]
    H = m.shape[0]
    polys = []
    for i, c in enumerate(cs):
        if hier[i][3] != -1 or len(c) < 3:
            continue
        ext = [(p[0][0] - 2, H - 1 - p[0][1] - 2) for p in c]
        holes = []
        k = hier[i][2]
        while k != -1:
            if len(cs[k]) >= 3:
                holes.append([(p[0][0] - 2, H - 1 - p[0][1] - 2) for p in cs[k]])
            k = hier[k][0]
        p = Polygon(ext, holes).buffer(0)
        if p.area > 4:
            polys.append(p.simplify(simplify))
    return unary_union(polys) if polys else Polygon()


def photo_mask(gray, positive=True, thr=None):
    """Silhueta da fotografia: objeto = zonas escuras (positivo) ou claras (negativo)."""
    g = ndimage.gaussian_filter(np.asarray(gray, float), 1.0)
    if thr is None:
        from skimage.filters import threshold_otsu
        try:
            thr = threshold_otsu(g)
        except Exception:
            thr = 0.5
    m = g < thr if positive else g >= thr
    m = ndimage.binary_opening(m, iterations=1)
    return m


def text_geom(text, font="Arial", size=20.0):
    from .text3d import text_polygons
    lines = [t for t in str(text).split("\n")] or [""]
    geoms = []
    for k, t in enumerate(lines):
        if not t.strip():
            continue
        g = text_polygons(t, font_family=font or "Arial", size_mm=size, bold=True)
        geoms.append(affinity.translate(g, 0, -k * size * 1.4))
    return unary_union(geoms) if geoms else Polygon()


def fit(geom, W, H=None, keep=True, center=True):
    """Ajusta a geometria a W × H mm (mantendo a proporção se keep) e centra-a na origem."""
    if geom.is_empty:
        return geom
    x0, y0, x1, y1 = geom.bounds
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    H = H or W
    if keep:
        k = min(W / w, H / h); kx = ky = k
    else:
        kx, ky = W / w, H / h
    g = affinity.scale(geom, kx, ky, origin=(x0, y0))
    g = affinity.translate(g, -x0, -y0)
    if center:
        b = g.bounds
        g = affinity.translate(g, -(b[0] + b[2]) / 2, -(b[1] + b[3]) / 2)
    return g


def silhouette(v, W, H=None, key="photo", positive=True):
    """Silhueta da fotografia ou do texto conforme o campo "source" (Photo/Text)."""
    if v.get("source", "Photo") == "Text" and str(v.get("text", "")).strip():
        g = text_geom(v["text"], v.get("font", "Arial"), 20)
    else:
        gray = load_gray(v.get(key, ""), 300)
        g = mask_to_geom(photo_mask(gray, positive), 0.6)
        if g.is_empty:
            g = sbox(0, 0, 10, 10)
    return fit(g, W, H)


def regular(shape, size, sy=None):
    """Forma regular centrada na origem com largura size (e altura sy)."""
    sy = sy or size
    if shape in ("Circle", "circle", "O"):
        return affinity.scale(Point(0, 0).buffer(0.5, quad_segs=48), size, sy)
    if shape in ("Rectangle", "Square"):
        return sbox(-size / 2, -sy / 2, size / 2, sy / 2)
    n = {"Triangle": 3, "Pentagon": 5, "Hexagon": 6, "Octagon": 8}.get(shape, 6)
    a0 = math.pi / 2 if n % 2 else math.pi / n
    p = Polygon([(math.cos(a0 + 2 * math.pi * k / n), math.sin(a0 + 2 * math.pi * k / n)) for k in range(n)])
    return fit(p, size, sy, keep=False)


def frame_ring(shape, inner, width, round_corner=True):
    """Moldura: forma regular que envolve a geometria interior, com a largura indicada."""
    if shape in ("No frame", None, "") or inner.is_empty:
        return Polygon(), inner
    x0, y0, x1, y1 = inner.bounds
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if shape == "Square":
        w = h = max(w, h)
    k = {"Triangle": 2.1, "Circle": 1.45, "Pentagon": 1.3, "Hexagon": 1.2, "Octagon": 1.12}.get(shape, 1.05)
    out_in = affinity.translate(regular(shape, w * k, h * k if shape not in ("Circle", "Square") or True else w * k), cx, cy)
    js = "round" if round_corner else "mitre"
    outer = out_in.buffer(width, join_style=js)
    return outer.difference(out_in), out_in


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def cs(geom):
    from .generators import cs as _cs
    return _cs(geom)


def to_tm(M):
    """manifold → trimesh sem fundir vértices por posição (evita juntar superfícies quase encostadas)."""
    mm = M.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def to_mf(m):
    from .meshops import to_manifold
    return to_manifold(m)


def ext(geom, h, z0=0.0):
    """Extrusão de uma geometria shapely (manifold)."""
    if geom.is_empty or h <= 0:
        return mf.Manifold()
    return mf.Manifold.extrude(cs(geom), h).translate((0, 0, z0))


def ext_tm(geom, h, z0=0.0):
    return to_tm(ext(geom, h, z0))


def grid_solid(Po, Pi, wrap=False):
    """Sólido fechado entre duas grelhas de pontos (R × C × 3): superfície exterior Po e interior Pi.
    wrap=True liga a última coluna à primeira (tubos, cones, esferas)."""
    Po = np.asarray(Po, float); Pi = np.asarray(Pi, float)
    R, C, _ = Po.shape
    n = R * C
    idx = np.arange(n).reshape(R, C)
    I = np.c_[idx, idx[:, :1]] if wrap else idx
    a = I[:-1, :-1].ravel(); b = I[:-1, 1:].ravel(); c = I[1:, :-1].ravel(); d = I[1:, 1:].ravel()
    fo = np.vstack([np.c_[a, c, b], np.c_[b, c, d]])
    fi = fo[:, ::-1] + n
    loops = [np.r_[idx[0, :]], np.r_[idx[-1, ::-1]]] if wrap else \
        [np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])]
    fs = []
    for L in loops:
        nb = np.roll(L, -1)
        fs.append(np.c_[L, nb, nb + n]); fs.append(np.c_[L, nb + n, L + n])
    F = np.vstack([fo, fi] + fs)
    m = trimesh.Trimesh(np.vstack([Po.reshape(-1, 3), Pi.reshape(-1, 3)]), F, process=True)
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return m


def plate_relief(T, W, H, base=0.0):
    """Placa W × H (centrada na origem, em XY) com relevo T (mm) por cima de uma base."""
    from .meshops import heightmap_to_solid
    m = heightmap_to_solid(np.asarray(T, float), W, H, base=base)
    m.apply_translation((-W / 2, -H / 2, 0))
    return m


def masked_relief(T, W, H, geom, base=0.0):
    """Relevo recortado pela geometria (centrada na origem)."""
    m = plate_relief(T, W, H, base)
    M = to_mf(m) ^ ext(geom, float(np.max(T)) + base + 2, -1)
    return to_tm(M)


def union(ms):
    ms = [m if isinstance(m, mf.Manifold) else to_mf(m) for m in ms if m is not None]
    ms = [m for m in ms if not m.is_empty()]
    if not ms:
        return trimesh.Trimesh()
    return to_tm(mf.Manifold.batch_boolean(ms, mf.OpType.Add) if len(ms) > 1 else ms[0])


def on_floor(m):
    if len(m.vertices):
        m.apply_translation((0, 0, -m.bounds[0][2]))
    return m


def group_floor(ms):
    """Assenta um conjunto de peças no chão mantendo as posições relativas."""
    ms = [m for m in ms if m is not None and len(m.vertices)]
    if ms:
        z = min(m.bounds[0][2] for m in ms)
        for m in ms:
            m.apply_translation((0, 0, -z))
    return ms


def rot(m, deg, axis):
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(deg), axis))
    return m


def fsize(meshes):
    """Tamanho aproximado do ficheiro STL binário."""
    n = sum(len(m.faces) for m in meshes if m is not None)
    b = 84 + 50 * n
    return f"{b / 1e6:.1f} MB" if b > 1e6 else f"{b / 1e3:.0f} KB" if b > 1e3 else f"{b} B"


def measure(meshes, extra=None):
    ms = [m for m in meshes if m is not None and len(getattr(m, "vertices", []))]
    d = {}
    if ms:
        lo = np.min([m.bounds[0] for m in ms], 0); hi = np.max([m.bounds[1] for m in ms], 0)
        e = hi - lo
        d["Tamanho"] = f"{e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm"
        vol = sum(abs(m.volume) for m in ms if m.is_volume) / 1000
        if vol:
            d["Volume"] = f"{vol:.1f} cm³"
        d["Peças"] = len(ms)
    d["Tamanho do ficheiro"] = fsize(ms)
    if extra:
        d.update(extra)
    return d


def result(meshes, names=None, colors=None, extra=None):
    meshes = [m for m in meshes if m is not None and len(getattr(m, "faces", []))]
    return {"meshes": meshes, "names": names, "colors": colors, "measure": measure(meshes, extra)}
