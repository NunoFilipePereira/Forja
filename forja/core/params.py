"""Geradores paramétricos: cada objeto criado guarda o gerador e os valores, para poder ser ajustado depois.

Cada parâmetro: (chave, rótulo, valor por omissão, opções ou intervalo, secção).
O construtor recebe (valores, fonte) e devolve uma malha ou uma lista de malhas.
A fonte guarda o que veio de fora: imagem ('image'), contorno 2D ('geom', WKT) ou texto."""
import numpy as np

REGISTRY = {}


def register(gid, name, params, needs=None):
    def deco(fn):
        REGISTRY[gid] = {"id": gid, "name": name, "params": params, "build": fn, "needs": needs}
        return fn
    return deco


def defaults(gid):
    return {p[0]: p[2] for p in REGISTRY[gid]["params"]}


def build(gid, values, src=None):
    g = REGISTRY[gid]
    v = defaults(gid); v.update(values or {})
    return g["build"](v, src or {})


def _geom(src):
    from shapely import wkt
    return wkt.loads(src["geom"]) if src.get("geom") else None


# ===========================================================================
# Primitivas
# ===========================================================================
from . import meshops as mo, generators as gn, litho as lt  # noqa: E402

SEG = (3, 512)


@register("caixa", "Caixa", [("x", "X (mm)", 20.0, (0.01, 1e5), "Dimensões"), ("y", "Y (mm)", 20.0, (0.01, 1e5), "Dimensões"),
                             ("z", "Z (mm)", 20.0, (0.01, 1e5), "Dimensões"), ("r", "Raio das arestas (mm)", 0.0, (0, 1e4), "Acabamento")])
def _box(v, s):
    if v["r"] > 0:
        g = gn.rounded_rect(v["x"], v["y"], v["r"])
        return mo.drop_to_floor(gn.from_manifold(gn.extrude(g, v["z"])))
    return mo.primitive("box", x=v["x"], y=v["y"], z=v["z"])


@register("esfera", "Esfera", [("r", "Raio (mm)", 10.0, (0.01, 1e5), "Dimensões"), ("sub", "Detalhe (1–6)", 4, (1, 6), "Qualidade"),
                               ("cut", "Cortar em baixo (fração, 0 = esfera inteira)", 0.0, (0, 0.95), "Forma")])
def _sphere(v, s):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=int(v["sub"]), radius=v["r"])
    if v["cut"] > 0:
        m, _ = mo.cut_plane(m, [0, 0, -v["r"] + 2 * v["r"] * v["cut"]], [0, 0, 1])
    return mo.drop_to_floor(m)


@register("cilindro", "Cilindro", [("r", "Raio (mm)", 10.0, (0.01, 1e5), "Dimensões"), ("r2", "Raio no topo (mm, 0 = igual)", 0.0, (0, 1e5), "Dimensões"),
                                   ("h", "Altura (mm)", 20.0, (0.01, 1e5), "Dimensões"), ("seg", "Segmentos", 64, SEG, "Qualidade"),
                                   ("hole", "Furo central (raio mm, 0 = sem)", 0.0, (0, 1e5), "Forma")])
def _cyl(v, s):
    import manifold3d as mf
    r2 = v["r2"] or v["r"]
    M = mf.Manifold.cylinder(v["h"], v["r"], r2, int(v["seg"]))
    if v["hole"] > 0:
        M = M - mf.Manifold.cylinder(v["h"] * 3, v["hole"], v["hole"], int(v["seg"])).translate((0, 0, -v["h"]))
    return gn.from_manifold(M)


@register("cone", "Cone", [("r", "Raio (mm)", 10.0, (0.01, 1e5), "Dimensões"), ("h", "Altura (mm)", 20.0, (0.01, 1e5), "Dimensões"),
                           ("seg", "Segmentos", 64, SEG, "Qualidade")])
def _cone(v, s):
    return mo.primitive("cone", r=v["r"], h=v["h"], seg=int(v["seg"]))


@register("toro", "Toro", [("R", "Raio maior (mm)", 20.0, (0.01, 1e5), "Dimensões"), ("r", "Raio menor (mm)", 5.0, (0.01, 1e5), "Dimensões")])
def _torus(v, s):
    return mo.primitive("torus", R=v["R"], r=min(v["r"], v["R"] * 0.99))


@register("capsula", "Cápsula", [("r", "Raio (mm)", 5.0, (0.01, 1e5), "Dimensões"), ("h", "Altura (mm)", 20.0, (0.01, 1e5), "Dimensões")])
def _capsule(v, s):
    return mo.primitive("capsule", r=v["r"], h=v["h"])


@register("prisma", "Prisma / forma extrudida", [("k", "Forma", "hexágono", gn.SHAPES_2D, "Forma"), ("s", "Tamanho (mm)", 30.0, (0.01, 1e5), "Dimensões"),
                                                ("h", "Altura (mm)", 20.0, (0.01, 1e5), "Dimensões"), ("tw", "Torção (°)", 0.0, (-3600, 3600), "Forma"),
                                                ("sc", "Escala no topo", 1.0, (0, 20), "Forma")])
def _prism(v, s):
    return mo.drop_to_floor(gn.prism(v["k"], v["s"], v["h"], v["tw"], v["sc"]))


@register("piramide", "Pirâmide", [("n", "Lados", 4, (3, 64), "Forma"), ("s", "Base (mm)", 40.0, (0.01, 1e5), "Dimensões"),
                                   ("h", "Altura (mm)", 30.0, (0.01, 1e5), "Dimensões"), ("top", "Topo (fração, 0 = bico)", 0.0, (0, 1), "Forma")])
def _pyr(v, s):
    from shapely.geometry import Polygon
    import math
    n = int(v["n"]); r = v["s"] / 2
    p = Polygon([(r * math.cos(2 * math.pi * k / n + math.pi / 2), r * math.sin(2 * math.pi * k / n + math.pi / 2)) for k in range(n)])
    return mo.drop_to_floor(gn.from_manifold(gn.extrude(p, v["h"], 0, max(v["top"], 0.001))))


# ===========================================================================
# Texto, relevos, litofania
# ===========================================================================
@register("texto3d", "Texto 3D", [("t", "Texto", "Forja", None, "Texto"), ("font", "Tipo de letra", "Helvetica", None, "Texto"),
                                  ("bold", "Negrito", True, None, "Texto"), ("size", "Altura das letras (mm)", 20.0, (0.1, 1e4), "Dimensões"),
                                  ("depth", "Espessura (mm)", 5.0, (0.01, 1e4), "Dimensões"), ("base", "Base por baixo (mm, 0 = sem)", 0.0, (0, 1e4), "Base"),
                                  ("margin", "Margem da base (mm)", 3.0, (0, 1e4), "Base"), ("rr", "Raio dos cantos da base (mm)", 2.0, (0, 1e4), "Base")])
def _text(v, s):
    from . import text3d
    g = text3d.text_polygons(v["t"] or " ", v["font"], v["size"], v["bold"])
    if g.is_empty:
        raise ValueError("Texto vazio.")
    M = gn.extrude(g, v["depth"])
    if v["base"] > 0:
        b = g.bounds
        w, d = b[2] - b[0] + 2 * v["margin"], b[3] - b[1] + 2 * v["margin"]
        plate = gn.rounded_rect(w, d, v["rr"])
        from shapely import affinity
        plate = affinity.translate(plate, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        M = gn.extrude(plate, v["base"]) + M.translate((0, 0, v["base"] - 0.01))
    return mo.center_on_plate(gn.from_manifold(M))


@register("litofania", "Litofania", [
    ("shape", "Forma", "retângulo", lt.SHAPES, "Forma"),
    ("width", "Largura (mm)", 100.0, (5, 2000), "Dimensões"), ("height", "Altura (mm, 0 = proporcional)", 0.0, (0, 2000), "Dimensões"),
    ("t_min", "Espessura mínima (mm)", 0.8, (0.2, 20), "Espessura"), ("t_max", "Espessura máxima (mm)", 3.0, (0.3, 50), "Espessura"),
    ("brightness", "Brilho", 0.0, (-100, 100), "Imagem"), ("contrast", "Contraste", 0.0, (-100, 300), "Imagem"),
    ("gamma", "Gama", 1.0, (0.1, 5), "Imagem"), ("smooth", "Suavização (px)", 0.6, (0, 10), "Imagem"),
    ("negative", "Negativo", False, None, "Imagem"), ("mirror", "Espelhar", False, None, "Imagem"),
    ("rotate", "Rodar imagem (°)", "0", ["0", "90", "180", "270"], "Imagem"),
    ("px_mm", "Resolução (mm por pixel)", 0.3, (0.05, 5), "Qualidade"),
    ("border", "Largura da moldura (mm, 0 = sem)", 4.0, (0, 200), "Moldura"), ("border_t", "Espessura da moldura (mm, 0 = automática)", 0.0, (0, 50), "Moldura"),
    ("corner_r", "Raio dos cantos (retângulo arredondado)", 6.0, (0, 500), "Moldura"),
    ("text", "Texto na moldura", "", None, "Moldura"), ("text_size", "Altura do texto (mm)", 5.0, (1, 100), "Moldura"),
    ("arc_angle", "Ângulo do arco (°)", 120.0, (5, 359), "Arco / cilindro / esfera"),
    ("sphere_angle", "Abertura da esfera (°)", 150.0, (20, 340), "Arco / cilindro / esfera"),
    ("base_ring", "Anel de base do cilindro (mm)", 3.0, (0, 50), "Arco / cilindro / esfera"),
    ("hole_d", "Furo porta-chaves (mm, 0 = sem)", 0.0, (0, 50), "Fixação"), ("hanger", "Argola para pendurar", False, None, "Fixação"),
    ("stand", "Pé para ficar de pé", False, None, "Fixação"), ("stand_depth", "Profundidade do pé (mm)", 25.0, (3, 500), "Fixação"),
    ("stand_h", "Altura do pé (mm)", 4.0, (1, 100), "Fixação"), ("vertical", "Orientar para imprimir na vertical", False, None, "Fixação"),
], needs="image")
def _litho(v, s):
    kw = dict(v); kw["rotate"] = int(kw["rotate"])
    return lt.lithophane(s["image"], **kw)


@register("relevo", "Relevo a partir de imagem", [
    ("w", "Largura (mm)", 100.0, (1, 5000), "Dimensões"), ("h", "Altura do relevo (mm)", 5.0, (0.01, 500), "Dimensões"),
    ("base", "Espessura da base (mm)", 2.0, (0, 500), "Dimensões"), ("inv", "Claro = alto", True, None, "Imagem"),
    ("brightness", "Brilho", 0.0, (-100, 100), "Imagem"), ("contrast", "Contraste", 0.0, (-100, 300), "Imagem"),
    ("gamma", "Gama", 1.0, (0.1, 5), "Imagem"), ("blur", "Suavização (px)", 1.0, (0, 20), "Imagem"),
    ("res", "Resolução máx. (px)", 300, (20, 2000), "Qualidade"),
    ("frame", "Moldura (mm, 0 = sem)", 0.0, (0, 200), "Moldura"), ("hole", "Furo para pendurar (mm, 0 = sem)", 0.0, (0, 50), "Moldura"),
], needs="image")
def _relief(v, s):
    g = lt.prepare_image(s["image"], v["brightness"], v["contrast"], v["gamma"], v["blur"], max_px=int(v["res"]))  # 1 = claro
    if v["frame"] > 0 or v["hole"] > 0:
        return gn.framed_relief(g * 255, v["w"], v["h"], v["base"], v["frame"], None, v["inv"], int(v["res"]), v["hole"])
    H = (g if v["inv"] else 1 - g) * v["h"]
    r, c = H.shape
    return mo.heightmap_to_solid(H, v["w"], v["w"] * r / c, base=v["base"])


@register("lito_cil", "Litofania cilíndrica (candeeiro)", [
    ("d", "Diâmetro (mm)", 80.0, (10, 1000), "Dimensões"), ("h", "Altura (mm, 0 = proporcional)", 0.0, (0, 2000), "Dimensões"),
    ("tmin", "Espessura mínima (mm)", 0.8, (0.2, 20), "Espessura"), ("tmax", "Espessura máxima (mm)", 3.0, (0.3, 50), "Espessura"),
    ("base", "Anel de base (mm, 0 = sem)", 3.0, (0, 100), "Base"), ("res", "Resolução (px)", 500, (50, 2000), "Qualidade"),
], needs="image")
def _lcyl(v, s):
    return gn.cylinder_lithophane(s["image"], v["d"], v["h"] or None, v["tmin"], v["tmax"], int(v["res"]), v["base"])


@register("lito_esf", "Litofania esférica", [
    ("d", "Diâmetro (mm)", 100.0, (10, 1000), "Dimensões"), ("tmin", "Espessura mínima (mm)", 0.8, (0.2, 20), "Espessura"),
    ("tmax", "Espessura máxima (mm)", 3.0, (0.3, 50), "Espessura"), ("op", "Abertura em baixo (mm)", 40.0, (0, 1000), "Forma"),
    ("res", "Resolução (px)", 400, (50, 2000), "Qualidade"),
], needs="image")
def _lsph(v, s):
    return gn.sphere_lithophane(s["image"], v["d"], v["tmin"], v["tmax"], int(v["res"]), v["op"])


# ===========================================================================
# Geradores
# ===========================================================================
@register("caixa_tampa", "Caixa com tampa", [
    ("w", "Largura (mm)", 60.0, (5, 5000), "Dimensões"), ("d", "Profundidade (mm)", 40.0, (5, 5000), "Dimensões"),
    ("h", "Altura (mm)", 30.0, (2, 5000), "Dimensões"), ("wall", "Parede (mm)", 2.0, (0.4, 50), "Paredes"),
    ("bottom", "Fundo/tampo (mm)", 2.0, (0.4, 50), "Paredes"), ("r", "Raio dos cantos (mm)", 4.0, (0, 500), "Paredes"),
    ("type", "Tipo de tampa", "encaixe", ["encaixe", "sobreposta"], "Tampa"), ("lip", "Rebordo da tampa (mm)", 4.0, (0.5, 100), "Tampa"),
    ("lh", "Altura tampa sobreposta (mm)", 8.0, (1, 500), "Tampa"), ("cl", "Folga (mm)", 0.25, (0, 5), "Tampa"),
    ("gap", "Distância entre caixa e tampa (mm)", 10.0, (0, 500), "Disposição"),
])
def _boxlid(v, s):
    base, lid = gn.box_with_lid(v["w"], v["d"], v["h"], v["wall"], v["bottom"], v["r"], v["lh"], v["lip"], v["cl"], v["type"])
    lid.apply_translation((v["w"] + v["gap"] + 2 * v["wall"], 0, 0))
    return [base, lid]


@register("caixas_enc", "Caixas encaixadas", [
    ("n", "Número", 3, (1, 20), "Dimensões"), ("w", "Largura exterior (mm)", 80.0, (5, 5000), "Dimensões"),
    ("d", "Profundidade (mm)", 60.0, (5, 5000), "Dimensões"), ("h", "Altura (mm)", 40.0, (2, 5000), "Dimensões"),
    ("wall", "Parede (mm)", 1.6, (0.4, 50), "Paredes"), ("gap", "Folga (mm)", 1.0, (0, 50), "Paredes"), ("r", "Raio dos cantos (mm)", 4.0, (0, 500), "Paredes"),
])
def _nested(v, s):
    ms = gn.nested_boxes(int(v["n"]), v["w"], v["d"], v["h"], v["wall"], v["gap"], v["r"])
    for k, m in enumerate(ms):
        m.apply_translation((k * (v["w"] + 10), 0, 0))
    return ms


@register("vaso", "Vaso", [
    ("h", "Altura (mm)", 120.0, (5, 2000), "Perfil"), ("rb", "Raio na base (mm)", 30.0, (1, 1000), "Perfil"),
    ("rm", "Raio a meio (mm)", 45.0, (1, 1000), "Perfil"), ("rt", "Raio no topo (mm)", 28.0, (1, 1000), "Perfil"),
    ("lobes", "Lóbulos", 6, (0, 60), "Padrão"), ("amp", "Ondulação (fração)", 0.08, (0, 0.5), "Padrão"),
    ("tw", "Torção (°)", 90.0, (-3600, 3600), "Padrão"), ("wall", "Parede (mm)", 1.6, (0.4, 50), "Paredes"),
    ("bottom", "Fundo (mm)", 2.0, (0.4, 50), "Paredes"), ("hollow", "Oco", True, None, "Paredes"),
    ("seg", "Segmentos", 120, (12, 720), "Qualidade"), ("layers", "Camadas", 80, (4, 800), "Qualidade"),
])
def _vase(v, s):
    return gn.vase(v["h"], v["rb"], v["rm"], v["rt"], int(v["lobes"]), v["amp"], v["tw"], v["wall"], v["bottom"], int(v["seg"]), int(v["layers"]), v["hollow"])


@register("mola", "Mola", [
    ("r", "Raio da mola (mm)", 10.0, (0.5, 1000), "Dimensões"), ("w", "Raio do arame (mm)", 1.5, (0.1, 100), "Dimensões"),
    ("p", "Passo (mm)", 6.0, (0.2, 1000), "Dimensões"), ("n", "Voltas", 5.0, (0.5, 200), "Dimensões"), ("flat", "Pontas planas", True, None, "Forma"),
])
def _spring(v, s):
    if v["p"] <= 2 * v["w"]:
        raise ValueError("O passo tem de ser maior do que o diâmetro do arame.")
    return gn.spring(v["r"], v["w"], v["p"], v["n"], flat_ends=v["flat"])


@register("porta_chaves", "Porta-chaves", [
    ("t", "Texto", "Forja", None, "Texto"), ("shape", "Forma", "retângulo arredondado", ["retângulo arredondado"] + gn.SHAPES_2D, "Forma"),
    ("w", "Largura (mm)", 50.0, (5, 1000), "Dimensões"), ("d", "Altura (mm)", 20.0, (5, 1000), "Dimensões"),
    ("th", "Espessura (mm)", 3.0, (0.4, 50), "Dimensões"), ("tt", "Relevo do texto (mm)", 1.5, (0, 20), "Texto"),
    ("hole", "Furo (mm)", 4.0, (0.5, 50), "Furo"),
])
def _keychain(v, s):
    from . import text3d
    tg = text3d.text_polygons(v["t"], bold=True) if v["t"] else None
    return gn.keychain(tg, v["shape"], v["w"], v["d"], v["th"], v["tt"], v["hole"])


@register("cortador", "Cortador de bolachas", [
    ("shape", "Forma (se não vier do desenho 2D)", "coração", gn.SHAPES_2D, "Forma"), ("fit", "Tamanho máximo (mm, 0 = manter)", 70.0, (0, 2000), "Dimensões"),
    ("h", "Altura (mm)", 15.0, (1, 200), "Dimensões"), ("blade", "Espessura da lâmina (mm)", 0.8, (0.2, 10), "Lâmina"),
    ("fw", "Largura da aba (mm)", 4.0, (0, 50), "Aba"), ("ft", "Espessura da aba (mm)", 1.6, (0.2, 20), "Aba"),
])
def _cutter(v, s):
    g = _geom(s) or gn.base_shape(v["shape"], 60)
    return gn.cookie_cutter(g, v["h"], v["blade"], v["fw"], v["ft"], v["fit"] or None)


@register("carimbo", "Carimbo", [
    ("t", "Texto (se não vier do desenho 2D)", "FORJA", None, "Texto"), ("size", "Altura do texto (mm)", 15.0, (1, 500), "Texto"),
    ("d", "Relevo (mm)", 2.0, (0.2, 50), "Dimensões"), ("b", "Base (mm)", 4.0, (0.5, 50), "Dimensões"),
    ("m", "Margem (mm)", 3.0, (0, 100), "Dimensões"), ("hh", "Pega (mm)", 15.0, (0, 300), "Pega"), ("mirror", "Espelhar (para carimbar)", True, None, "Texto"),
])
def _stamp(v, s):
    from . import text3d
    g = _geom(s) or text3d.text_polygons(v["t"], size_mm=v["size"], bold=True)
    return gn.stamp(g, v["d"], v["b"], v["m"], v["hh"], v["mirror"])


@register("candeeiro_base", "Base de candeeiro", [
    ("d", "Diâmetro da litofania (mm)", 80.0, (10, 1000), "Dimensões"), ("h", "Altura (mm)", 30.0, (5, 500), "Dimensões"),
    ("wall", "Parede (mm)", 2.0, (0.4, 20), "Dimensões"), ("sock", "Furo do casquilho (mm)", 28.0, (1, 200), "Casquilho"),
    ("cab", "Furo do cabo (mm)", 8.0, (0, 50), "Casquilho"), ("t", "Espessura máx. da litofania (mm)", 3.0, (0.5, 20), "Encaixe"),
    ("seat", "Altura do encaixe (mm)", 4.0, (0.5, 50), "Encaixe"), ("cl", "Folga (mm)", 0.4, (0, 5), "Encaixe"),
])
def _lamp(v, s):
    return gn.lamp_base(v["d"], v["h"], v["wall"], v["sock"], v["cab"], v["seat"], v["cl"], v["t"])


@register("relevo_moldura", "Relevo com moldura", [
    ("w", "Largura (mm)", 100.0, (5, 2000), "Dimensões"), ("rel", "Altura do relevo (mm)", 3.0, (0.1, 100), "Dimensões"),
    ("base", "Base (mm)", 1.5, (0.2, 50), "Dimensões"), ("b", "Moldura (mm, 0 = sem)", 5.0, (0, 200), "Moldura"),
    ("bh", "Altura da moldura (mm, 0 = automática)", 0.0, (0, 200), "Moldura"), ("inv", "Claro = alto", True, None, "Imagem"),
    ("hole", "Furo para pendurar (mm, 0 = sem)", 4.0, (0, 50), "Moldura"), ("res", "Resolução (px)", 400, (20, 2000), "Qualidade"),
], needs="image")
def _framed(v, s):
    return gn.framed_relief(s["image"], v["w"], v["rel"], v["base"], v["b"], v["bh"] or None, v["inv"], int(v["res"]), v["hole"])


@register("cavilha", "Cavilhas", [("d", "Diâmetro (mm)", 5.0, (0.5, 100), "Dimensões"), ("l", "Comprimento (mm)", 20.0, (1, 1000), "Dimensões"),
                                  ("c", "Chanfro (mm)", 0.6, (0, 10), "Dimensões"), ("n", "Quantidade", 4, (1, 200), "Quantidade")])
def _dowels(v, s):
    import trimesh
    ms = []
    for k in range(int(v["n"])):
        m = gn.dowel(v["d"], v["l"], min(v["c"], v["d"] / 2.2)); m.apply_translation(((k % 10) * v["d"] * 2, (k // 10) * v["d"] * 2, 0)); ms.append(m)
    return trimesh.util.concatenate(ms)


@register("suporte_foto", "Suporte em pé para foto", [("w", "Largura da foto (mm)", 100.0, (10, 1000), "Dimensões"),
                                                       ("t", "Espessura da foto (mm)", 3.0, (0.3, 50), "Dimensões"),
                                                       ("a", "Inclinação (°)", 10.0, (0, 45), "Forma"), ("h", "Altura (mm)", 15.0, (3, 200), "Dimensões"),
                                                       ("bd", "Profundidade da base (mm)", 30.0, (5, 300), "Dimensões")])
def _stand(v, s):
    return gn.photo_stand(v["w"], v["t"], 0.3, v["bd"], v["h"], v["a"])


@register("gancho", "Gancho", [("w", "Largura (mm)", 30.0, (5, 500), "Dimensões"), ("t", "Espessura (mm)", 3.0, (0.5, 50), "Dimensões"),
                               ("h", "Furo (mm)", 5.0, (0.5, 100), "Dimensões"), ("d", "Comprimento da patilha (mm)", 6.0, (0, 200), "Dimensões")])
def _hook(v, s):
    return gn.hanger(v["w"], v["t"], v["h"], v["d"])


@register("garra", "Garra / clip em U", [("t", "Espessura a prender (mm)", 3.0, (0.2, 100), "Dimensões"), ("l", "Comprimento (mm)", 25.0, (2, 500), "Dimensões"),
                                         ("o", "Folga (mm)", 0.4, (0, 10), "Dimensões"), ("w", "Parede (mm)", 2.0, (0.4, 20), "Dimensões")])
def _clip(v, s):
    return gn.claw_clip(v["t"], v["l"], v["o"], v["w"])


@register("pilar", "Pilar decorativo", [("d", "Diâmetro (mm)", 20.0, (2, 1000), "Dimensões"), ("h", "Altura (mm)", 80.0, (2, 5000), "Dimensões"),
                                        ("f", "Caneluras", 12, (0, 120), "Forma"), ("fd", "Profundidade das caneluras (mm)", 1.5, (0, 50), "Forma"),
                                        ("bh", "Altura da base (mm)", 6.0, (0, 200), "Forma"), ("ch", "Altura do capitel (mm)", 4.0, (0, 200), "Forma")])
def _pillar(v, s):
    return gn.pillar(v["d"], v["h"], int(v["f"]), v["fd"], v["bh"], v["ch"])


@register("puzzle", "Puzzle", [("w", "Largura (mm)", 150.0, (10, 5000), "Dimensões"), ("h", "Altura (mm)", 100.0, (10, 5000), "Dimensões"),
                               ("c", "Colunas", 4, (1, 50), "Peças"), ("r", "Linhas", 3, (1, 50), "Peças"), ("tab", "Tamanho dos encaixes (fração)", 0.2, (0.05, 0.35), "Peças"),
                               ("t", "Espessura (mm)", 3.0, (0.3, 50), "Dimensões"), ("gap", "Folga (mm)", 0.15, (0, 2), "Peças"),
                               ("seed", "Variante", 1, (0, 9999), "Peças")])
def _puzzle(v, s):
    from .text3d import extrude
    pieces = gn.puzzle_pieces(v["w"], v["h"], int(v["c"]), int(v["r"]), v["tab"], int(v["seed"]))
    out = []
    for k, p in enumerate(pieces):
        m = extrude(p.buffer(-v["gap"] / 2), v["t"])
        m.apply_translation(((k % int(v["c"])) * 4, (k // int(v["c"])) * 4, 0)); out.append(m)
    return out


@register("extrusao_torcida", "Extrusão com torção", [("shape", "Forma (se não vier do desenho 2D)", "estrela", gn.SHAPES_2D, "Forma"),
                                                      ("size", "Tamanho (mm)", 40.0, (1, 5000), "Forma"), ("h", "Altura (mm)", 60.0, (0.1, 5000), "Dimensões"),
                                                      ("t", "Torção (°)", 120.0, (-3600, 3600), "Torção"), ("sc", "Escala no topo", 0.7, (0, 20), "Torção")])
def _twist(v, s):
    g = _geom(s) or gn.base_shape(v["shape"], v["size"])
    return mo.center_on_plate(gn.twisted_extrusion(g, v["h"], v["t"], v["sc"]))


@register("revolucao", "Sólido de revolução", [("seg", "Segmentos", 96, (3, 720), "Qualidade"), ("deg", "Ângulo (°)", 360.0, (1, 360), "Forma")], needs="geom")
def _lathe(v, s):
    return mo.drop_to_floor(gn.lathe(_geom(s), int(v["seg"]), v["deg"]))


@register("extrusao", "Extrusão de formas 2D", [("h", "Altura (mm)", 5.0, (0.01, 5000), "Dimensões"), ("tw", "Torção (°)", 0.0, (-3600, 3600), "Forma"),
                                                ("sc", "Escala no topo", 1.0, (0, 20), "Forma"), ("ch", "Chanfro no topo (mm)", 0.0, (0, 50), "Acabamento")], needs="geom")
def _extr(v, s):
    g = _geom(s)
    if v["ch"] > 0 and v["tw"] == 0 and v["sc"] == 1:
        import manifold3d as mf
        body = gn.extrude(g, max(v["h"] - v["ch"], 0.01))
        top = mf.Manifold.batch_hull([gn.extrude(g, 0.01).translate((0, 0, v["h"] - v["ch"])),
                                      gn.extrude(g.buffer(-v["ch"]), 0.01).translate((0, 0, v["h"] - 0.01))]) if not g.buffer(-v["ch"]).is_empty else None
        if top is not None and g.convex_hull.area - g.area < 1e-6:
            return gn.from_manifold(body + top)
    return gn.twisted_extrusion(g, v["h"], v["tw"], v["sc"])
