"""Ferramenta "Converter": converte o modelo em 24 formas diferentes (mesmos métodos e campos)."""
import numpy as np
from ..core import conv_sheet as LS, conv3d as L3

METHODS = ["Block", "Clicker", "Cocoon", "Contour", "Folding", "Grid", "Grill", "Hash", "Helix", "Mold", "Pipe", "Plate", "Relief",
           "Revolving", "Slide", "Spinner", "Spring", "Stack", "Star", "Thermal", "Tiling", "Tumbler", "Wave", "Wireframe"]
SHEET = {"Contour", "Folding", "Grill", "Hash", "Plate", "Stack", "Star"}
NEST = ["Balanced", "Compact", "Ordered"]
YN = ("radio", ["Sim", "Não"])

PT = {"Block": "Blocos", "Clicker": "Botão (clicker)", "Cocoon": "Casulo", "Contour": "Contornos", "Folding": "Dobragem (papercraft)",
      "Grid": "Grelha", "Grill": "Grelha de aletas", "Hash": "Fatias cruzadas", "Helix": "Hélice", "Mold": "Molde", "Pipe": "Tubos",
      "Plate": "Placas", "Relief": "Relevo", "Revolving": "Revolução", "Slide": "Fatias rotativas", "Spinner": "Pião", "Spring": "Mola",
      "Stack": "Camadas empilhadas", "Star": "Estrela", "Thermal": "Térmico", "Tiling": "Textura", "Tumbler": "Sempre-em-pé",
      "Wave": "Onda", "Wireframe": "Arame"}


def F(key, label, default, opts, *methods, group="Saída"):
    return (key, label, default, opts, group, set(methods))


FIELDS = [
    # chapa (comum)
    F("sheet_w", "Largura da chapa (mm)", 600.0, (10, 1e5), *SHEET, group="Chapa"),
    F("sheet_h", "Altura da chapa (mm)", 450.0, (10, 1e5), *SHEET, group="Chapa"),
    F("sheet_t", "Espessura da chapa (mm)", 3.2, (0.1, 100), *SHEET, group="Chapa"),
    F("low_poly", "Low poly (fração das faces; 0 = automático ≈ 100 faces)", 0.0, (0, 1e5), "Folding", "Plate", "Pipe"),
    # Block
    F("block_size", "Tamanho do bloco (mm)", 10.0, (0.5, 1e4), "Block"),
    F("wall", "Espessura da parede (n.º de blocos)", 5, (0, 1000), "Block"),
    # Clicker
    F("click_shape", "Forma", "Circle", ("radio", ["Circle", "Square"]), "Clicker"),
    F("click_xy", "Tamanho do botão XY (mm)", 15.0, (1, 1e4), "Clicker"),
    F("pressed", "Curso ao carregar (mm)", 25.0, (1, 1e4), "Clicker"),
    F("base_xy", "Tamanho da base XY (mm)", 10.0, (1, 1e4), "Clicker"),
    F("base_len", "Comprimento da base (mm)", 10.0, (1, 1e4), "Clicker"),
    F("position", "Posição (−1, 1)", 0.0, (-1, 1), "Clicker"),
    # Cocoon
    F("cocoon_shape", "Forma", "Cube", ["Cube", "Cylinder", "Polygon", "Sphere", "Native"], "Cocoon"),
    F("ox", "Desvio X", 0.0, (-1, 1), "Cocoon"), F("oy", "Desvio Y", 0.0, (-1, 1), "Cocoon"), F("oz", "Desvio Z", 0.0, (-1, 1), "Cocoon"),
    # Direção (vários)
    F("direction", "Direção", "Z", ("radio", ["X", "Y", "Z"]), "Contour", "Stack", "Grill"),
    F("frame", "Armação", "Triangle", ["Triangle", "Rectangle", "Square", "Pentagon", "Hexagon", "Octagon", "Circle"], "Contour"),
    F("socket", "Casquilho de lâmpada na base", False, None, "Contour", "Wireframe"),
    # Folding
    F("max_fold", "Dobras máximas (peças)", 1, (1, 50), "Folding"),
    F("tab", "Tamanho da aba (mm)", 5.0, (0, 100), "Folding"),
    # Grid
    F("dual", "Dual", "Não", YN, "Grid"),
    F("cell", "Célula", "Rectangle", ["Triangle", "Rectangle", "Hexagon", "Circle"], "Grid"),
    F("cell_x", "Tamanho da célula X (mm)", 10.0, (1, 1e4), "Grid"), F("cell_y", "Tamanho da célula Y (mm)", 10.0, (1, 1e4), "Grid"),
    F("line_w", "Largura da linha (mm)", 1.0, (0.1, 100), "Grid", "Thermal"),
    F("ang_x", "Ângulo X", 0.0, (-180, 180), "Grid"), F("ang_y", "Ângulo Y", 0.0, (-180, 180), "Grid"), F("ang_z", "Ângulo Z", 0.0, (-180, 180), "Grid"),
    # Grill / Hash / Star / Wireframe
    F("layout", "Disposição", "Orthogonal", ("radio", ["Orthogonal", "Radial"]), "Grill", "Hash", "Wireframe"),
    F("base_pos", "Base [−1, 1]", 0.0, (-1, 1), "Grill"),
    F("base_t", "Espessura da base (mm)", 5.0, (0, 1e4), "Grill", "Relief", "Thermal", "Revolving"),
    F("layers", "Camadas", 6, (1, 500), "Grill"),
    F("angle", "Ângulo (°)", 0.0, (-360, 360), "Grill", "Tiling"),
    F("surface", "Superfície (só contorno)", "Não", YN, "Hash"),
    F("contour_w", "Largura do contorno (mm)", 15.0, (0, 1e4), "Hash", "Stack"),
    F("x_layer", "Camadas X", 6, (0, 500), "Hash", "Star", "Wireframe"),
    F("z_layer", "Camadas Z", 6, (0, 500), "Hash", "Wireframe"),
    F("y_layer", "Camadas Y", 6, (0, 500), "Wireframe"),
    F("tilt", "Inclinação (°)", 0.0, (-180, 180), "Hash", "Wireframe"),
    F("text_size", "Tamanho do texto (mm)", 5.0, (0, 100), "Grill", "Hash", "Plate", "Stack", "Star"),
    F("dowel", "Tamanho da cavilha (mm)", 5.0, (0, 100), "Stack"),
    F("finger", "Largura do dente (mm)", 15.0, (1, 1e4), "Plate"),
    # centro / eixos
    F("cx", "Centro X [−1, 1]", 0.0, (-1, 1), "Helix", "Mold", "Star"),
    F("cz", "Centro Z [−1, 1]", 0.0, (-1, 1), "Helix", "Mold", "Star"),
    F("start", "Ângulo inicial (°)", 0.0, (-360, 360), "Star"),
    # Helix
    F("arms", "Número de braços", 6, (1, 100), "Helix"), F("arm_w", "Largura do braço (mm)", 5.0, (0.2, 1e4), "Helix"),
    F("diameter", "Diâmetro (mm)", 100.0, (0, 1e4), "Helix"), F("rotation", "Rotação (°)", 60.0, (-3600, 3600), "Helix"),
    F("solid_base", "Base sólida", "Sim", YN, "Helix"),
    # Mold
    F("cuts", "Número de cortes", 2, (1, 24), "Mold"), F("gap", "Folga (mm)", 10.0, (0, 1e4), "Mold"),
    F("mold_t", "Espessura (mm)", 5.0, (0.4, 1e4), "Mold", "Spring"),
    F("mold_t", "Espessura (mm)", 3.0, (0.4, 1e4), "Tumbler"),
    F("opening", "Tamanho da abertura", 1.0, (0, 100), "Mold"),
    # Pipe
    F("cross", "Secção", "Triangle", ["Triangle", "Square", "Pentagon", "Hexagon", "Octagon", "Circle"], "Pipe"),
    F("face", "Faces", "Triangle", ["Triangle", "Quadrangle", "Polygon", "Voronoi"], "Pipe"),
    F("outer", "Diâmetro exterior (mm)", 6.0, (0.5, 1e4), "Pipe"), F("inner", "Diâmetro interior (mm)", 4.0, (0, 1e4), "Pipe"),
    F("connector", "Conector", "Male", ("radio", ["Male", "Female"]), "Pipe"),
    F("solid", "Tubo", "Solid", ("radio", ["Solid", "Hollow"]), "Pipe"),
    F("contact", "Comprimento de contacto (mm)", 5.0, (0, 1e4), "Pipe"), F("contact_t", "Espessura de contacto (mm)", 2.0, (0, 1e4), "Pipe"),
    F("chamfer", "Chanfro (mm)", 0.5, (0, 100), "Pipe"),
    # Relief / Thermal
    F("rdir", "Direção", "+X", ("radio", ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]), "Relief"),
    F("rdir", "Direção", "+X", ("radio", ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]), "Thermal"),
    F("sides", "Faces", "Single-sided", ("radio", ["Single-sided", "Double-sided"]), "Relief"),
    F("polarity", "Polaridade", "Positive", ("radio", ["Positive", "Negative"]), "Relief"),
    F("rthick", "Espessura do relevo (mm)", 10.0, (0.1, 1e4), "Relief", "Thermal"),
    F("bury", "Enterrar", "0", ["0", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"], "Relief", "Thermal"),
    F("base", "Base", "Triangle", ["Triangle", "Rectangle", "Square", "Pentagon", "Hexagon", "Octagon", "Circle"], "Relief"),
    F("border", "Largura da moldura (mm)", 5.0, (0, 1e4), "Relief", "Thermal"),
    F("border_t", "Espessura da moldura (mm)", 10.0, (0, 1e4), "Relief"),
    F("cell_shape", "Forma da célula", "Triangle", ["Triangle", "Rectangle", "Hexagon", "Circle", "Solid"], "Thermal"),
    F("cell_size", "Tamanho da célula (mm)", 10.0, (1, 1e4), "Thermal"),
    F("cell_t", "Espessura (mm)", 2.0, (0, 1e4), "Thermal"),
    # Revolving
    F("rev_shape", "Forma de revolução", "Cylinder", ["Cylinder", "Cone", "Vase", "Sphere", "Spiral"], "Revolving"),
    F("repeat", "Repetir", 2, (1, 50), "Revolving"),
    F("side", "Lado do modelo", "+X", ("radio", ["+X", "-X", "+Y", "-Y"]), "Revolving"),
    F("inner_d", "Diâmetro interior (mm)", 50.0, (1, 1e4), "Revolving"),
    F("arc", "Ângulo do arco (°)", 60.0, (5, 360), "Revolving"),
    F("model_t", "Espessura do modelo (mm)", 20.0, (0.1, 1e4), "Revolving"),
    F("cap_line", "Linha do topo (mm)", 0.0, (0, 1e4), "Revolving"), F("cap_t", "Espessura do topo (mm)", 5.0, (0, 1e4), "Revolving"),
    F("base_line", "Linha da base (mm)", 2.0, (0, 1e4), "Revolving"),
    # Slide
    F("slide_w", "Largura da fatia (mm)", 4.0, (0.5, 1e4), "Slide"), F("axle_d", "Diâmetro do eixo (mm)", 5.0, (0.5, 1e4), "Slide"),
    F("axle_y", "Eixo Y [−1, 1]", 0.0, (-1, 1), "Slide"), F("axle_z", "Eixo Z [−1, 1]", 0.0, (-1, 1), "Slide"),
    # Spinner
    F("top_d", "Diâmetro do topo (mm)", 4.0, (0.5, 1e4), "Spinner"), F("base_d", "Diâmetro da base (mm)", 8.0, (0.5, 1e4), "Spinner"),
    F("top_h", "Altura do topo (mm)", 20.0, (0, 1e4), "Spinner"), F("base_h", "Altura da base (mm)", 20.0, (0, 1e4), "Spinner"),
    F("top_off", "Desvio do topo (mm)", 5.0, (0, 1e4), "Spinner"), F("base_off", "Desvio da base (mm)", 5.0, (0, 1e4), "Spinner", "Tumbler"),
    # Spring
    F("layer_w", "Largura da camada (mm)", 10.0, (0.5, 1e4), "Spring"), F("separation", "Separação (mm)", 0.4, (0.05, 100), "Spring"),
    # Tiling
    F("tile_dir", "Direção", "Z", ("radio", ["X", "Y", "Z"]), "Tiling"),
    F("min_raise", "Elevação mínima (mm)", 0.0, (0, 100), "Tiling"), F("max_raise", "Elevação máxima (mm)", 1.0, (0, 100), "Tiling"),
    F("resolution", "Resolução (mm)", 0.5, (0.05, 100), "Tiling"),
    # Tumbler
    F("tbase", "Forma da base", "Semi-cylinder", ["Semi-cylinder", "Hemisphere"], "Tumbler"),
    # Wave
    F("osc", "Oscilar em", "Source", ["X", "Y", "Z", "Source"], "Wave"),
    F("src_x", "Fonte X [−1, 1]", -0.5, (-1, 1), "Wave"), F("src_y", "Fonte Y [−1, 1]", -0.5, (-1, 1), "Wave"), F("src_z", "Fonte Z [−1, 1]", 0.0, (-1, 1), "Wave"),
    F("amplitude", "Amplitude (mm)", 5.0, (0, 1e4), "Wave"), F("attenuation", "Atenuação", 0.0, (0, 100), "Wave"),
    F("wavelength", "Comprimento de onda (mm)", 10.0, (0.5, 1e4), "Wave"), F("phase", "Fase (°)", 0.0, (-360, 360), "Wave"),
    # Wireframe
    F("wtype", "Tipo", "Axial", ["Axial", "Edge", "Voronoi"], "Wireframe"),
    F("wdim", "Dimensão", "2D", ("radio", ["2D", "3D"]), "Wireframe"),
    # tolerância/folga (vários)
    F("clearance", "Folga (mm)", 3.0, (0, 1e4), "Contour", "Wireframe", "Relief"),
    F("clearance", "Folga (mm)", 0.0, (0, 1e4), "Cocoon"),
    F("tol", "Tolerância (mm)", 0.0, (-5, 5), "Clicker", "Grill", "Hash", "Helix", "Mold", "Pipe", "Plate", "Slide", "Star", "Tumbler"),
    # encaixe nas chapas
    F("nest", "Encaixe nas chapas", "Compact", NEST, *SHEET, group="Encaixe"),
    F("allow_rot", "Permitir rotação", "Sim", YN, *SHEET, group="Encaixe"),
]


def run(mesh, v, fast):
    m = v["method"]
    sheet = (v["sheet_w"], v["sheet_h"]); t = v["sheet_t"]; rot = v["allow_rot"] == "Sim"; nest = v["nest"]
    yes = lambda k: v[k] == "Sim"
    if m == "Block": return L3.block(mesh, v["block_size"], v["wall"], fast)
    if m == "Clicker": return L3.clicker(mesh, v["click_shape"], v["click_xy"], v["pressed"], v["base_xy"], v["base_len"], v["position"], v["tol"], fast)
    if m == "Cocoon": return L3.cocoon(mesh, v["cocoon_shape"], v["clearance"], v["ox"], v["oy"], v["oz"], fast=fast)
    if m == "Contour": return LS.contour(mesh, sheet, t, v["direction"], v["frame"], v["clearance"], v["socket"], mode=nest, rotate=rot, fast=fast)
    if m == "Folding": return LS.folding(mesh, sheet, t, v["low_poly"], v["max_fold"], v["tab"], nest, rot, fast)
    if m == "Grid": return L3.grid(mesh, yes("dual"), v["cell"], v["cell_x"], v["cell_y"], v["line_w"], v["ang_x"], v["ang_y"], v["ang_z"], fast)
    if m == "Grill": return LS.grill(mesh, sheet, t, v["direction"], v["layout"], v["base_pos"], v["base_t"], v["layers"], v["angle"], v["tol"], v["text_size"], nest, rot, fast)
    if m == "Hash": return LS.hash_(mesh, sheet, t, v["layout"], yes("surface"), v["contour_w"], v["x_layer"], v["z_layer"], v["tilt"], v["tol"], v["text_size"], nest, rot, fast)
    if m == "Helix": return L3.helix(mesh, v["cx"], v["cz"], v["arms"], v["arm_w"], v["diameter"], v["rotation"], yes("solid_base"), v["tol"], fast=fast)
    if m == "Mold": return L3.mold(mesh, v["cuts"], v["cx"], v["cz"], v["gap"], v["mold_t"], v["opening"], v["tol"], fast)
    if m == "Pipe": return L3.pipe(mesh, v["low_poly"], v["cross"], v["face"], v["outer"], v["inner"], v["connector"], v["solid"], v["contact"], v["contact_t"], v["chamfer"], v["tol"], fast)
    if m == "Plate": return LS.plate(mesh, sheet, t, v["low_poly"], v["finger"], v["tol"], v["text_size"], nest, rot, fast)
    if m == "Relief": return L3.relief(mesh, v["rdir"], v["sides"], v["polarity"], v["rthick"], v["bury"], v["base"], v["base_t"], v["border"], v["border_t"], v["clearance"], fast=fast)
    if m == "Revolving": return L3.revolving(mesh, v["rev_shape"], v["repeat"], v["side"], v["inner_d"], v["arc"], v["model_t"], v["cap_line"], v["cap_t"], v["base_line"], v["base_t"], fast)
    if m == "Slide": return L3.slide(mesh, v["slide_w"], v["axle_d"], v["axle_y"], v["axle_z"], v["tol"], fast=fast)
    if m == "Spinner": return L3.spinner(mesh, v["top_d"], v["base_d"], v["top_h"], v["base_h"], v["top_off"], v["base_off"], fast)
    if m == "Spring": return L3.spring(mesh, v["layer_w"], v["mold_t"], v["separation"], fast)
    if m == "Stack": return LS.stack(mesh, v["direction"], sheet, t, v["contour_w"], v["dowel"], v["text_size"], nest, rot, fast)
    if m == "Star": return LS.star(mesh, sheet, t, v["cx"], v["cz"], v["x_layer"], v["start"], v["tol"], v["text_size"], nest, rot, fast)
    if m == "Thermal": return L3.thermal(mesh, v["rdir"], v["rthick"], v["bury"], v["base_t"], v["border"], v["cell_shape"], v["cell_size"], v["line_w"], v["cell_t"], fast)
    if m == "Tiling": return L3.tiling(mesh, getattr(run, "image", None), v["tile_dir"], v["min_raise"], v["max_raise"], v["angle"], v["resolution"], fast)
    if m == "Tumbler": return L3.tumbler(mesh, v["tbase"], v["base_off"], v["mold_t"], v["tol"], fast)
    if m == "Wave": return L3.wave(mesh, v["osc"], [(v["src_x"], v["src_y"], v["src_z"])], v["amplitude"], v["attenuation"], v["wavelength"], v["phase"], fast)
    if m == "Wireframe": return L3.wireframe(mesh, v["layout"], v["x_layer"], v["y_layer"], v["z_layer"], v["tilt"], v["wtype"], v["wdim"], v["clearance"], v["socket"])
    raise ValueError(m)


LABELS = dict(PT)
LABELS.update({"Circle": "Círculo", "Square": "Quadrado", "Cube": "Cubo", "Cylinder": "Cilindro", "Polygon": "Polígono", "Sphere": "Esfera",
               "Native": "Forma do modelo", "Triangle": "Triângulo", "Rectangle": "Retângulo", "Pentagon": "Pentágono", "Hexagon": "Hexágono",
               "Octagon": "Octógono", "Orthogonal": "Ortogonal", "Radial": "Radial", "Male": "Macho", "Female": "Fêmea", "Solid": "Maciço",
               "Hollow": "Oco", "Quadrangle": "Quadrângulo", "Single-sided": "Uma face", "Double-sided": "Duas faces", "Positive": "Positivo",
               "Negative": "Negativo", "Cone": "Cone", "Vase": "Vaso", "Spiral": "Espiral", "Semi-cylinder": "Meio cilindro",
               "Hemisphere": "Hemisfério", "Source": "Fonte", "Axial": "Axial", "Edge": "Arestas", "Balanced": "Equilibrado",
               "Compact": "Compacto", "Ordered": "Ordenado"})
SPEC = {"title": "Converter", "methods": METHODS, "fields": FIELDS, "run": run, "labels": LABELS}
