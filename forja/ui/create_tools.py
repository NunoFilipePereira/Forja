"""Ferramentas do menu Criar: Litofania (18 formas), Photo magic (25 métodos),
Projeção (4 modos), Garrafa, Hélice de bambu, Dado, Óculos e Planador — com os mesmos campos e valores por omissão."""
from ..core import lithoshapes as LS

YN = ("radio", ["Yes", "No"])
PN = ("radio", ["Positive", "Negative"])
EI = ("radio", ["Exterior", "Interior"])
IMG = ("file", "image")
MESH = ("file", "mesh")

LABELS = {
    "Yes": "Sim", "No": "Não", "Positive": "Positivo", "Negative": "Negativo", "Exterior": "Exterior", "Interior": "Interior",
    "Upright": "Direita", "Inverted": "Invertida", "Fixed": "Fixa", "Stretch": "Esticada", "Bright": "Clara", "Dark": "Escura",
    "Solid": "Maciça", "Hollow": "Oca", "One body": "Uma peça", "Assembly": "Montagem", "Single-sided": "Uma face",
    "Double-sided": "Duas faces", "Photo": "Fotografia", "Text": "Texto", "Circle": "Círculo", "Square": "Quadrado",
    "Triangle": "Triângulo", "Rectangle": "Retângulo", "Pentagon": "Pentágono", "Hexagon": "Hexágono", "Octagon": "Octógono",
    "No frame": "Sem moldura", "Automatic": "Automática", "User input": "Escolhida", "Small": "Pequena", "Large": "Grande",
    "Inner size": "Tamanho interior", "Outer size": "Tamanho exterior", "Ignore": "Ignorar",
    "Top to surface center": "Topo ao centro da superfície", "Center to surface center": "Centro ao centro da superfície",
    "Bottom to surface center": "Base ao centro da superfície",
    # formas de litofania
    "Flat": "Plana", "Arc": "Arco", "CMYK": "CMYK (a cores)", "Dual": "Dupla (acordeão)", "Polygon": "Polígono",
    "Frustum": "Tronco de pirâmide", "Cylinder": "Cilindro", "Cone": "Cone", "Vase": "Vaso", "Sphere": "Esfera",
    "Spiral": "Espiral", "Stencil": "Recorte (molde)", "Balloon": "Balão", "Box": "Caixa", "Bronze": "Bronze (de modelo 3D)",
    "Tattoo": "Tatuagem (sobre modelo 3D)", "Tile": "Revestimento (modelo 3D)",
}


def F(key, label, default, opts, *methods, group="Saída"):
    return (key, label, default, opts, group, set(methods) if methods else None)


ALL = set(LS.SHAPES)
WRAP = {"Rectangle", "Polygon", "Frustum", "Cylinder", "Cone", "Vase"}
BORDER = ALL - {"CMYK", "Bronze", "Tattoo", "Tile"}
BORDER_T = {"Flat", "Arc", "Dual", "Rectangle", "Polygon", "Frustum", "Cylinder", "Cone", "Vase", "Sphere", "Spiral"}
PHOTO = ALL - {"Bronze"}
MODEL3D = {"Bronze", "Tattoo", "Tile"}

LITHO_FIELDS = [
    F("template", "Molde (template)", "", IMG, "Stencil", "Balloon", "Box", group="Entrada"),
    F("model", "Modelo 3D", "", MESH, *MODEL3D, group="Entrada"),
    F("photo", "Fotografia", "", IMG, *PHOTO, group="Entrada"),
    F("polarity", "Imagem", "Positive", PN, *(ALL - {"CMYK"})),
    F("orient", "Orientação", "Upright", ("radio", ["Upright", "Inverted"]), *WRAP),
    F("side", "Relevo", "Exterior", EI, "Arc", "Cylinder", "Cone", "Vase", "Sphere", "Balloon", "Box", "Bronze", "Tattoo", "Tile"),
    F("fit", "Fotografia", "Fixed", ("radio", ["Fixed", "Stretch"]), "Cylinder", "Cone", "Vase"),
    F("perspective", "Perspetiva", "Yes", YN, "Frustum"),
    F("align", "Posição", "Center to surface center",
      ("radio", ["Top to surface center", "Center to surface center", "Bottom to surface center"]), "Sphere"),
    F("bottom_off", "Desvio em baixo (°)", 0.0, (-360, 360), "Sphere"),
    F("sphere_angle", "Ângulo da esfera (°)", 180.0, (10, 360), "Sphere"),
    F("sections", "Secções da fotografia", 3, (1, 20), "Dual"),
    F("sides", "Número de lados", 3, (3, 24), "Polygon", "Frustum"),
    F("arc_angle", "Ângulo do arco (°)", 30.0, (1, 360), "Arc"),
    F("tdir", "Direção", "X", ("radio", ["X", "Y", "Z"]), "Tile"),
    F("direction", "Direção", "+Z", ["+X", "+Y", "+Z", "-X", "-Y", "-Z"], "Tattoo"),
    F("box_t", "Espessura da caixa (mm)", 5.0, (0.5, 50), "Box"),
    F("top_x", "Tamanho em cima X (mm)", 40.0, (1, 2000), "Frustum"),
    F("bot_x", "Tamanho em baixo X (mm)", 80.0, (1, 2000), "Frustum"),
    F("top_d", "Diâmetro em cima (mm)", 40.0, (1, 2000), "Cone"),
    F("bot_d", "Diâmetro em baixo (mm)", 60.0, (1, 2000), "Cone"),
    F("diam", "Diâmetro do modelo (mm)", 50.0, (5, 2000), "Cylinder", "Vase", "Sphere", "Spiral"),
    F("sx", "Tamanho do modelo X (mm)", 50.0, (1, 2000), "Flat", "CMYK", "Dual", "Rectangle", "Polygon", "Stencil", "Balloon", "Box",
      "Bronze", "Tattoo", "Tile"),
    F("sy", "Tamanho do modelo Y (mm)", 50.0, (1, 2000), "CMYK", "Rectangle", "Box", "Bronze", "Tattoo", "Tile"),
    F("sz", "Tamanho do modelo Z (mm)", 50.0, (1, 2000), *(ALL - {"CMYK", "Sphere"})),
    F("opening", "Diâmetro da abertura (mm)", 21.8, (0, 2000), "Sphere"),
    F("tmin", "Espessura mínima (mm)", 0.8, (0.1, 20), *ALL),
    F("tmax", "Espessura máxima (mm)", 3.2, (0.2, 30), *ALL),
    F("res", "Resolução (mm)", 0.1, (0.05, 5), *(ALL - {"Bronze"})),
    F("contrast", "Contraste", 0.0, (-100, 100), "Bronze"),
    F("ctrl", "Pontos de controlo", 3, (2, 3), "Vase"),
    F("cp1", "Ponto 1 (raio relativo)", 1.0, (0.2, 3), "Vase"),
    F("cp2", "Ponto 2 (raio relativo)", 1.0, (0.2, 3), "Vase"),
    F("cp3", "Ponto 3 (raio relativo)", 1.0, (0.2, 3), "Vase"),
    F("slant", "Inclinação (°)", 10.0, (-45, 45), "Flat"),
    F("tol", "Tolerância (mm)", -0.2, (-5, 5), "Box"),
    F("body", "Peças", "One body", ("radio", ["One body", "Assembly"]), "Rectangle", "Polygon"),
    F("sided", "Faces", "Single-sided", ("radio", ["Single-sided", "Double-sided"]), "Stencil", "Box"),
    F("pcx", "Centro X [−1, 1]", 0.0, (-1, 1), "Stencil", "Balloon", "Box", "Tattoo"),
    F("pcy", "Centro Y [−1, 1]", 0.0, (-1, 1), "Stencil", "Balloon", "Box", "Tattoo"),
    F("pscale", "Escala da fotografia", 1.0, (0.05, 10), "Stencil", "Balloon", "Box"),
    F("pscale_x", "Escala da fotografia X", 1.0, (0.05, 10), "Tattoo"),
    F("pscale_y", "Escala da fotografia Y", 1.0, (0.05, 10), "Tattoo"),
    F("expansion", "Expansão", 1.0, (0.1, 10), "Balloon"),
    F("extra", "Espaço extra (mm)", 0.0, (0, 100), "Balloon"),
    F("cmyk", "CMYK", "No", YN, "Stencil", "Box"),
    F("color_t", "Espessura da cor (mm)", 0.6, (0.1, 5), "CMYK", "Stencil", "Box"),
    F("clarity", "Nitidez da cor", 8.0, (0.5, 50), "CMYK", "Stencil", "Box"),
    F("angle", "Ângulo (°)", 0.0, (-360, 360), "Tile"),
    F("border", "Margem", "Dark", ("radio", ["Bright", "Dark"]), *BORDER, group="Margem e base"),
    F("border_size", "Tamanho da margem (mm)", 2.0, (0, 100), *BORDER, group="Margem e base"),
    F("border_t", "Espessura da margem (mm)", 4.0, (0.1, 50), *BORDER_T, group="Margem e base"),
    F("base_len", "Comprimento da base (mm)", 10.0, (0, 500), "Flat", "Arc", group="Margem e base"),
    F("base_t", "Espessura da base (mm)", 4.0, (0, 100), "Flat", "Arc", "Rectangle", "Polygon", "Frustum", "Cylinder", "Cone", "Vase",
      "Sphere", group="Margem e base"),
    F("base", "Base", "Solid", ("radio", ["Solid", "Hollow"]), *WRAP, group="Margem e base"),
    F("socket", "Tipo de casquilho", "O", ["O", "U"], "Flat", "Arc", group="Margem e base"),
    F("socket_d", "Diâmetro do casquilho (mm)", 20.0, (0, 200), "Flat", "Arc", "Sphere", group="Margem e base"),
    F("socket_h", "Altura do casquilho (mm)", 10.0, (0, 200), "Flat", "Arc", "Sphere", group="Margem e base"),
    F("hanger", "Argola para pendurar", "No", YN, "Flat", "Arc", "Rectangle", "Polygon", "Frustum", "Cylinder", "Vase",
      group="Margem e base"),
    F("support", "Escoras", "No", YN, "Flat", "Arc", group="Margem e base"),
]

LITHO = {"title": "Litofania", "methods": LS.SHAPES, "fields": LITHO_FIELDS, "run": LS.run, "labels": LABELS,
         "model_size": False}


def defaults(spec, method):
    """Valores por omissão de uma ferramenta para um método (útil para testes)."""
    v = {"method": method}
    for f in spec["fields"]:
        ms = f[5] if len(f) > 5 else None
        if ms is None or method in ms:
            v[f[0]] = f[2]
        else:
            v.setdefault(f[0], f[2])
    for f in spec["fields"]:  # as entradas ativas para o método têm prioridade
        ms = f[5] if len(f) > 5 else None
        if ms is None or method in ms:
            v[f[0]] = f[2]
    return v


# ==========================================================================
# Photo magic
# ==========================================================================
from ..core import photomagic as PM  # noqa: E402

PM_ALL = set(PM.METHODS)
TXT = {"Channel", "Cutter", "Extrusion", "Flexi", "Morphing", "Ridge"}
COLOR = {"Clicker", "Fabric", "Painting", "Puzzle"}
FRAME = {"Halftone", "Layering", "Pendant", "Wire"}
SX = PM_ALL - {"Box", "Clicker", "Cutter", "Teacup", "Whistle", "Pillar"}
SY = SX - {"Balloon", "Lenticule"}
SHAPES7 = ["Triangle", "Rectangle", "Square", "Pentagon", "Hexagon", "Octagon", "Circle"]
FRAMES = ["No frame"] + SHAPES7
LABELS.update({
    "Balloon": "Balão", "Block": "Blocos", "Cell": "Células", "Channel": "Canal (néon)", "Clicker": "Botão (clicker)",
    "Cutter": "Cortador de bolachas", "Extrusion": "Extrusão", "Fabric": "Tecido", "Fidget": "Fidget", "Flexi": "Articulado",
    "Halftone": "Meio-tom", "Layering": "Camadas", "Lenticule": "Lenticular", "Moire": "Moiré", "Morphing": "Morfose (3 vistas)",
    "Painting": "Pintura (camadas de cor)", "Pendant": "Pendente", "Pillar": "Pilares", "Puzzle": "Puzzle", "Ridge": "Cristas",
    "Teacup": "Chávena", "Whistle": "Apito", "Wire": "Arame", "Wormhole": "Túnel",
    "Basic": "Simples", "Twist": "Rodar (baioneta)", "Inlay": "Embutida", "Extrude": "Em relevo", "Indent": "Gravada",
    "Intrude": "Rebaixada", "Hollow lid": "Tampa vazada", "Hollow box": "Caixa vazada", "Integrate": "Integrado",
    "Separate": "Separado", "Concave Arc": "Arco côncavo", "Convex Arc": "Arco convexo", "Vertical": "Vertical",
    "Original": "Original", "Photo size": "Tamanho da foto", "Stripe": "Riscas", "Radial": "Radial", "Spiral": "Espiral",
    "Line": "Linha", "Ellipse": "Elipse", "Semiellipse": "Semi-elipse", "Sawtooth": "Dente de serra", "Sine": "Sinusoide",
    "Random": "Aleatório", "Color layer": "Camadas de cor", "Height map": "Mapa de alturas", "One layer": "Uma camada",
    "Multi-layer": "Várias camadas",
})

PM_FIELDS = [
    F("template", "Molde (template)", "", IMG, "Clicker", "Fabric", "Painting", group="Entrada"),
    F("photo", "Fotografia", "", IMG, *PM_ALL, group="Entrada"),
    F("photos", "Ordem das camadas (uma fotografia por camada)", [], ("files", "image"), "Layering", group="Entrada"),
    F("photo2", "Segunda fotografia", "", IMG, "Lenticule", "Morphing", "Moire", group="Entrada"),
    F("photo3", "Terceira fotografia (vista de cima)", "", IMG, "Morphing", group="Entrada"),
    F("source", "Origem", "Photo", ("radio", ["Photo", "Text"]), *TXT, group="Entrada"),
    F("font", "Tipo de letra", "Arial", None, *TXT, group="Entrada"),
    F("text", "Texto", "", None, "Channel", "Cutter", "Extrusion", "Flexi", "Ridge", group="Entrada"),
    F("text1", "Texto 1 (frente)", "", None, "Morphing", group="Entrada"),
    F("text2", "Texto 2 (lado)", "", None, "Morphing", group="Entrada"),
    F("text3", "Texto 3 (cima)", "", None, "Morphing", group="Entrada"),
    F("polarity", "Imagem", "Positive", PN, "Extrusion", "Morphing"),
    F("polarity", "Zonas", "Bright", ("radio", ["Bright", "Dark"]), "Cell", "Halftone", "Lenticule", "Pillar"),
    F("shape", "Forma", "Circle", ("radio", ["Circle", "Square"]), "Clicker"),
    F("size_mode", "Medidas", "Inner size", ("radio", ["Inner size", "Outer size"]), "Box"),
    F("mode", "Tipo", "Color layer", ("radio", ["Color layer", "Height map"]), "Puzzle"),
    F("layers_mode", "Camadas", "One layer", ("radio", ["One layer", "Multi-layer"]), "Painting", "Puzzle"),
    F("sided", "Faces", "Single-sided", ("radio", ["Single-sided", "Double-sided"]), "Painting"),
    F("mirror", "Espelhar o verso", "Yes", YN, "Painting"),
    F("flip", "Espelhar a fotografia", "Yes", YN, "Cutter"),
    F("pattern", "Padrão", "Stripe", ["Stripe", "Radial", "Spiral", "Stencil"], "Halftone"),
    F("pattern", "Padrão", "Line", ["Line", "Ellipse", "Semiellipse", "Sawtooth", "Sine", "Random"], "Pillar"),
    F("diam", "Diâmetro do modelo (mm)", 100.0, (10, 1000), "Teacup"),
    F("sx", "Tamanho do modelo X (mm)", 100.0, (1, 2000), *SX),
    F("sy", "Tamanho do modelo Y (mm)", 100.0, (1, 2000), *SY),
    F("sz", "Tamanho do modelo Z (mm)", 100.0, (1, 2000), "Balloon", "Lenticule", "Morphing"),
    F("sz", "Tamanho do modelo Z (mm)", 20.0, (1, 2000), "Block", "Wormhole"),
    F("sz", "Tamanho do modelo Z (mm)", 6.0, (1, 2000), "Fidget"),
    F("sz", "Tamanho do modelo Z (mm)", 40.0, (1, 2000), "Teacup"),
    F("sx", "Tamanho do modelo X (mm)", 40.0, (1, 2000), "Clicker"),
    F("sy", "Tamanho do modelo Y (mm)", 40.0, (1, 2000), "Clicker"),
    F("bx", "Tamanho da base X (mm)", 100.0, (5, 2000), "Box"),
    F("by", "Tamanho da base Y (mm)", 100.0, (5, 2000), "Box"),
    F("bz", "Tamanho da base Z (mm)", 40.0, (5, 2000), "Box"),
    F("expansion", "Expansão", 1.0, (0, 20), "Balloon", "Extrusion"),
    F("expansion", "Expansão (mm)", 0.0, (0, 20), "Layering", "Wire"),
    F("block", "Tamanho do bloco (mm)", 10.0, (0.5, 200), "Block"),
    F("gap", "Folga entre blocos (mm)", 2.0, (0, 50), "Block"),
    F("gap", "Folga (mm)", 2.0, (0.2, 50), "Wormhole"),
    F("solid_top", "Topo maciço", "No", YN, "Block"),
    F("solid_base", "Base maciça", "No", YN, "Block", "Channel", "Layering", "Wire"),
    F("thick", "Espessura (mm)", 5.0, (0.2, 100), "Box", "Fidget", "Pendant"),
    F("thick", "Espessura (mm)", 0.5, (0.1, 100), "Cell"),
    F("thick", "Espessura (mm)", 3.0, (0.2, 100), "Clicker", "Lenticule", "Wormhole"),
    F("thick", "Espessura (mm)", 5.0, (0.2, 100), "Flexi", "Pillar"),
    F("thick", "Espessura (mm)", 1.0, (0.2, 100), "Halftone"),
    F("thick", "Espessura (mm)", 4.0, (0.2, 100), "Moire"),
    F("thick", "Espessura (mm)", 2.0, (0.2, 100), "Teacup", "Wire"),
    F("thick", "Espessura (mm)", 1.5, (0.2, 100), "Whistle"),
    F("lid_type", "Tipo de tampa", "Basic", ["Basic", "Twist", "Flat", "Inlay"], "Box"),
    F("lid_z", "Altura da tampa Z (mm)", 15.0, (1, 500), "Box"),
    F("lid_indent", "Encaixe da tampa (mm)", 10.0, (0, 500), "Box"),
    F("pattern", "Padrão interior", "Ignore", ["Ignore", "Extrude", "Indent", "Intrude", "Hollow lid", "Hollow box"], "Box"),
    F("nested", "Caixa dentro de caixa", "No", YN, "Box"),
    F("chamfer", "Chanfro das arestas (mm)", 0.5, (0, 20), "Box", "Cutter"),
    F("border", "Margem (mm)", 1.0, (0, 50), "Cell"),
    F("border", "Margem (mm)", 3.0, (0, 50), "Lenticule"),
    F("border", "Margem (mm)", 2.0, (0, 50), "Painting"),
    F("cell_shape", "Forma da célula", "Square", ["Square", "Pentagon", "Hexagon", "Octagon", "Circle"], "Cell"),
    F("ch_w", "Largura do canal (mm)", 5.0, (0.5, 100), "Channel"),
    F("ch_h", "Altura do canal (mm)", 5.0, (0.5, 100), "Channel"),
    F("wall", "Espessura da parede (mm)", 2.0, (0.4, 50), "Channel"),
    F("cap_t", "Espessura da tampa (mm)", 1.0, (0.2, 50), "Channel"),
    F("base_t", "Espessura da base (mm)", 1.0, (0, 50), "Channel", "Halftone"),
    F("base_t", "Espessura da base (mm)", 5.0, (0, 50), "Extrusion"),
    F("base_t", "Espessura da base (mm)", 3.0, (0, 50), "Puzzle", "Ridge"),
    F("cap_xy", "Tamanho da haste XY (mm)", 7.0, (1, 200), "Clicker"),
    F("cap_len", "Comprimento da haste (mm)", 10.0, (1, 200), "Clicker"),
    F("pressed", "Comprimento premido (mm)", 25.0, (1, 200), "Clicker"),
    F("base_xy", "Tamanho da base XY (mm)", 10.0, (1, 200), "Clicker"),
    F("base_len", "Comprimento da base (mm)", 10.0, (1, 200), "Clicker"),
    F("inner_x", "Tamanho interior X (mm)", 40.0, (5, 1000), "Cutter"),
    F("inner_y", "Tamanho interior Y (mm)", 40.0, (5, 1000), "Cutter"),
    F("inner", "Padrão interior", "Ignore", ["Ignore", "Integrate", "Separate"], "Cutter"),
    F("inner_depth", "Profundidade interior (mm)", 7.0, (0.5, 100), "Cutter"),
    F("inner_clear", "Folga interior (mm)", 5.0, (0, 100), "Cutter"),
    F("blade_w", "Largura da lâmina (mm)", 0.5, (0.2, 10), "Cutter"),
    F("blade_h", "Altura da lâmina (mm)", 1.0, (0.2, 50), "Cutter"),
    F("mid_w", "Largura do meio (mm)", 2.0, (0.2, 20), "Cutter"),
    F("mid_h", "Altura do meio (mm)", 10.0, (0.5, 100), "Cutter"),
    F("base_w", "Largura da base (mm)", 4.0, (0.5, 50), "Cutter"),
    F("base_h", "Altura da base (mm)", 4.0, (0.5, 50), "Cutter"),
    F("base_hole", "Base vazada", "Yes", YN, "Cutter"),
    F("stopper", "Travessa", "Yes", YN, "Cutter"),
    F("extrusion", "Extrusão (mm)", 10.0, (0.5, 500), "Extrusion"),
    F("elevation", "Elevação (mm)", 10.0, (0, 500), "Extrusion"),
    F("elev_angle", "Ângulo de elevação (°)", 0.0, (-90, 90), "Extrusion"),
    F("clearance", "Folga (mm)", 1.0, (0, 50), "Extrusion"),
    F("round", "Cantos arredondados", "No", YN, "Extrusion"),
    F("support", "Suporte", "Concave Arc", ["Concave Arc", "Convex Arc", "Triangle", "Vertical"], "Extrusion"),
    F("base_shape", "Forma da base", "Triangle", SHAPES7 + ["Original", "Photo size"], "Extrusion"),
    F("keychain", "Furo de porta-chaves (mm)", 0.0, (0, 50), "Extrusion"),
    F("cell", "Tamanho da célula (mm)", 8.0, (2, 100), "Fabric"),
    F("hole", "Diâmetro do furo (mm)", 15.0, (2, 200), "Fidget"),
    F("slot", "Espessura da ranhura (mm)", 2.0, (0.4, 20), "Fidget"),
    F("section", "Secções", 5, (1, 50), "Flexi"),
    F("plane", "Ajustar plano", 1, (1, 50), "Flexi"),
    F("plane_off", "Desvio do plano (%)", 0.0, (-50, 50), "Flexi"),
    F("contrast", "Contraste", 5.0, (0.1, 50), "Halftone"),
    F("stripe_w", "Largura da risca (mm)", 5.0, (0.5, 50), "Halftone"),
    F("stripe_angle", "Ângulo da risca (°)", 0.0, (-180, 180), "Halftone"),
    F("layers", "Número de camadas", 4, (1, 30), "Layering"),
    F("layer_h", "Altura da camada (mm)", 3.0, (0.1, 50), "Layering"),
    F("layer_h", "Altura da camada (mm)", 0.4, (0.04, 5), "Painting", "Puzzle"),
    F("layer_off", "Distância entre camadas (mm)", 5.0, (0, 100), "Layering"),
    F("clipping", "Recortar", "No", YN, "Layering", "Wire"),
    F("bar", "Barra de reforço (mm)", 5.0, (0, 50), "Lenticule"),
    F("bar", "Espessura das barras (mm)", 2.0, (0.4, 50), "Morphing"),
    F("res", "Resolução (mm)", 0.1, (0.05, 5), "Balloon", "Cell", "Lenticule", "Painting", "Puzzle"),
    F("slit", "Largura da fenda (mm)", 3.0, (0.4, 50), "Moire"),
    F("grid", "Grelha", "No", YN, "Morphing"),
    F("x_sec", "Secções X", 5, (1, 100), "Morphing"),
    F("y_sec", "Secções Y", 5, (1, 100), "Morphing"),
    F("z_sec", "Secções Z", 5, (1, 100), "Morphing"),
    F("x_sec", "Secções X", 4, (1, 50), "Puzzle"),
    F("y_sec", "Secções Y", 4, (1, 50), "Puzzle"),
    F("base_color", "Camadas de base", 0, (0, 20), "Painting"),
    F("box_t", "Espessura da caixa (mm)", 3.0, (0, 50), "Painting"),
    F("box_inner_h", "Altura interior da caixa (mm)", 0.0, (0, 100), "Painting"),
    F("stopper_w", "Largura do batente (mm)", 0.0, (0, 50), "Painting"),
    F("stopper_h", "Altura do batente (mm)", 0.0, (0, 50), "Painting"),
    F("lid", "Tampa", "Large", ("radio", ["Small", "Large"]), "Painting"),
    F("lip", "Espessura do rebordo (mm)", 2.0, (0, 20), "Painting"),
    F("width_x", "Largura em X (mm)", 5.0, (0.5, 100), "Pillar"),
    F("depth_y", "Profundidade em Y (mm)", 50.0, (1, 1000), "Pillar"),
    F("height_z", "Altura em Z (mm)", 100.0, (1, 1000), "Pillar"),
    F("lod", "Nível de detalhe", 100, (10, 400), "Pillar"),
    F("lod", "Nível de detalhe", 2.0, (0.5, 10), "Wire"),
    F("cap", "Altura do topo (mm)", 10.0, (0, 200), "Pillar"),
    F("base", "Altura da base (mm)", 20.0, (0, 200), "Pillar"),
    F("text_size", "Tamanho do texto (mm)", 5.0, (1, 100), "Pillar"),
    F("ridge_scale", "Escala das cristas", 1.0, (0.1, 10), "Ridge"),
    F("curvature", "Curvatura", 0.0, (-3, 3), "Teacup"),
    F("base_h", "Altura da base (mm)", 3.0, (0.5, 50), "Teacup"),
    F("chamber_h", "Altura da câmara (mm)", 20.0, (5, 200), "Whistle"),
    F("duct_gap", "Abertura da conduta (mm)", 2.0, (0.4, 20), "Whistle"),
    F("duct_len", "Comprimento da conduta (mm)", 15.0, (2, 200), "Whistle"),
    F("opening", "Janela (mm)", 5.0, (1, 50), "Whistle"),
    F("width", "Largura (mm)", 15.0, (3, 200), "Whistle"),
    F("wire_w", "Largura do arame (mm)", 3.0, (0.4, 20), "Wire"),
    F("frame", "Moldura", "No frame", FRAMES, *FRAME, group="Moldura"),
    F("round", "Cantos arredondados", "Yes", YN, *FRAME, group="Moldura"),
    F("frame_w", "Largura da moldura (mm)", 3.0, (0.4, 50), *FRAME, group="Moldura"),
    F("color", "Cor", "Automatic", ["Automatic", "User input"], *COLOR, group="Cor"),
    F("color_layers", "Camadas de cor", 4, (1, 16), *COLOR, group="Cor"),
    F("tol", "Tolerância (mm)", 0.0, (-5, 5), "Block", "Box", "Channel", "Clicker", "Cutter", "Fidget", "Flexi", "Moire",
      "Painting", "Pendant", "Pillar", "Puzzle"),
]

PHOTO_MAGIC = {"title": "Photo magic", "methods": PM.METHODS, "fields": PM_FIELDS, "run": PM.run, "labels": LABELS,
               "model_size": False}


# ==========================================================================
# Projection e objetos
# ==========================================================================
from ..core import create3d as C3  # noqa: E402

LABELS.update({"Fence": "Vedação (sombra no chão)", "Pandora": "Pandora (caixa de luz)", "Squama": "Escamas (parede)",
               "Torch": "Lanterna (projetor)", "Left": "Esquerda", "Right": "Direita", "Top": "Cima", "Bottom": "Baixo"})

PROJECTION = {"title": "Projeção", "methods": ["Fence", "Pandora", "Squama", "Torch"], "labels": LABELS, "model_size": False,
              "run": C3.run_projection, "fields": [
                  F("shadow", "Sombra (imagem)", "", IMG, group="Entrada"),
                  F("litho", "Litofania (imagem)", "", IMG, "Pandora", group="Entrada"),
                  F("base_shape", "Forma da base (imagem)", "", IMG, "Pandora", group="Entrada"),
                  F("polarity", "Imagem", "Positive", PN, "Torch"),
                  F("shape", "Forma", "Circle", ("radio", ["Circle", "Square"]), "Torch"),
                  F("fences", "Número de vedações", 5, (1, 30), "Fence"),
                  F("fence_w", "Largura da vedação (mm)", 3.0, (0.5, 50), "Fence"),
                  F("shadow_len", "Comprimento da sombra (mm)", 100.0, (10, 5000), "Fence"),
                  F("shadow_off", "Distância da sombra (mm)", 10.0, (0, 5000), "Fence"),
                  F("light_dist", "Distância da luz (mm)", 50.0, (1, 5000), "Fence"),
                  F("light_h", "Altura da luz (mm)", 50.0, (1, 5000), "Fence", "Pandora"),
                  F("shadow_size", "Tamanho da sombra (mm)", 500.0, (10, 10000), "Pandora"),
                  F("litho_min", "Litofania: espessura mín. (mm)", 0.8, (0.1, 10), "Pandora"),
                  F("litho_max", "Litofania: espessura máx. (mm)", 3.2, (0.2, 20), "Pandora"),
                  F("litho_res", "Litofania: resolução (mm)", 0.1, (0.05, 5), "Pandora"),
                  F("sock_x", "Casquilho: tamanho X (mm)", 100.0, (10, 1000), "Pandora"),
                  F("sock_y", "Casquilho: tamanho Y (mm)", 100.0, (10, 1000), "Pandora"),
                  F("sock_d", "Diâmetro do casquilho (mm)", 10.0, (1, 200), "Pandora"),
                  F("sock_h", "Altura do casquilho (mm)", 10.0, (1, 200), "Pandora"),
                  F("stop_d", "Diâmetro do batente (mm)", 7.0, (1, 200), "Pandora"),
                  F("stop_h", "Altura do batente (mm)", 5.0, (0.5, 200), "Pandora"),
                  F("pillars", "Número de pilares", 3, (1, 12), "Pandora"),
                  F("pillar_s", "Tamanho do pilar (mm)", 5.0, (1, 50), "Pandora"),
                  F("pillar_h", "Altura do pilar (mm)", 35.0, (5, 500), "Pandora"),
                  F("sx", "Tamanho do modelo X (mm)", 300.0, (10, 3000), "Squama"),
                  F("sz", "Tamanho do modelo Z (mm)", 300.0, (10, 3000), "Squama"),
                  F("cell", "Tamanho da célula (mm)", 6.0, (1, 100), "Squama"),
                  F("light", "Fonte de luz", "Left", ["Left", "Right", "Top", "Bottom"], "Squama"),
                  F("light_angle", "Ângulo da luz (°)", 5.0, (0, 80), "Squama"),
                  F("surface_angle", "Ângulo da superfície (°)", 10.0, (0, 60), "Squama"),
                  F("film_size", "Tamanho do filme (mm)", 30.0, (5, 200), "Torch"),
                  F("film_dist", "Distância do filme (mm)", 40.0, (5, 500), "Torch"),
                  F("proj_dist", "Distância de projeção (mm)", 200.0, (10, 10000), "Torch"),
                  F("sock_size", "Tamanho do casquilho (mm)", 15.0, (3, 200), "Torch"),
                  F("sock_depth", "Profundidade do casquilho (mm)", 10.0, (1, 200), "Torch"),
                  F("thick", "Espessura (mm)", 3.0, (0.4, 50)),
                  F("tol", "Tolerância (mm)", 0.0, (-5, 5), "Pandora", "Torch"),
              ]}


def _one(fn, title, fields, labels=None):
    return {"title": title, "methods": [], "fields": fields, "labels": dict(LABELS, **(labels or {})), "model_size": False,
            "run": lambda mesh, v, fast: fn(v, fast)}


BOTTLE = _one(C3.bottle, "Garrafa", [
    F("h_section", "Secção horizontal (imagem)", "", IMG, group="Entrada"),
    F("v_section", "Secção vertical (imagem)", "", IMG, group="Entrada"),
    F("solid", "Corpo", "Hollow", ("radio", ["Solid", "Hollow"])),
    F("sx", "Tamanho do modelo X (mm)", 50.0, (5, 2000)), F("sy", "Tamanho do modelo Y (mm)", 50.0, (5, 2000)),
    F("sz", "Tamanho do modelo Z (mm)", 100.0, (5, 2000)),
    F("thick", "Espessura do modelo (mm)", 3.0, (0.4, 50)),
    F("cap", "Tampa", "Yes", YN), F("cap_h", "Altura da tampa (mm)", 15.0, (1, 200)),
    F("cap_t", "Espessura da tampa (mm)", 3.0, (0.4, 50)), F("tol", "Tolerância (mm)", 0.0, (-5, 5))])

BAMBOO = _one(C3.bamboo, "Hélice de bambu", [
    F("body", "Peças", "One body", ("radio", ["One body", "Assembly"])),
    F("blades", "Número de pás", 3, (1, 12)), F("blade_angle", "Ângulo das pás (°)", 30.0, (0, 80)),
    F("blade_len", "Comprimento das pás (mm)", 50.0, (5, 500)), F("root_w", "Largura da raiz da pá (mm)", 5.0, (1, 100)),
    F("blade_t", "Espessura da pá (mm)", 2.0, (0.4, 20)), F("shaft_d", "Diâmetro da haste (mm)", 5.0, (1, 50)),
    F("shaft_len", "Comprimento da haste (mm)", 80.0, (5, 1000)), F("tol", "Tolerância (mm)", -0.1, (-5, 5))])

DICE = _one(C3.dice, "Dado", [
    F("template", "Símbolo da face 1 (imagem)", "", IMG, group="Entrada"),
    F("size", "Tamanho (mm)", 20.0, (3, 500)), F("corner_r", "Raio dos cantos (mm)", 2.0, (0, 100)),
    F("chamfer", "Chanfro das arestas (mm)", 0.5, (0, 50)), F("extrusion", "Extrusão (mm; negativo = gravado)", -1.0, (-20, 20))])

EYEGLASSES = _one(C3.eyeglasses, "Óculos", [
    F("template", "Forma da lente (imagem)", "", IMG, group="Entrada"),
    F("body", "Peças", "One body", ("radio", ["One body", "Assembly"])),
    F("eye", "Tamanho do olho (mm)", 50.0, (20, 100)), F("aspect", "Proporção", 1.0, (0.3, 3)),
    F("frame_t", "Espessura da armação (mm)", 3.0, (1, 20)), F("frame_screw", "Parafuso na armação", "Yes", ["Yes", "No"]),
    F("bridge_x", "Ponte: tamanho X (mm)", 20.0, (5, 60)), F("bridge_z", "Ponte: tamanho Z (mm)", 3.0, (1, 20)),
    F("bridge_h", "Altura da ponte (mm)", 5.0, (0, 50)), F("bridge_curv", "Curvatura da ponte (mm)", 2.0, (-20, 20)),
    F("double_bridge", "Ponte dupla", "No", ["Yes", "No"]),
    F("nose_size", "Plaqueta: tamanho (mm)", 5.0, (1, 30)), F("nose_h", "Plaqueta: altura (mm)", 4.0, (0.5, 30)),
    F("nose_tilt", "Plaqueta: inclinação (°)", 0.0, (-45, 45)), F("corner", "Tamanho do canto (mm)", 7.0, (1, 40)),
    F("temple_len", "Comprimento da haste (mm)", 140.0, (50, 250)), F("temple_t", "Espessura da haste (mm)", 2.0, (0.5, 10)),
    F("temple_size", "Largura da haste (mm)", 5.0, (1, 30)), F("temple_h", "Altura da haste (mm)", 8.0, (1, 40)),
    F("temple_angle", "Abertura da haste (°)", 90.0, (0, 180)), F("ear_len", "Comprimento da curva da orelha (mm)", 30.0, (0, 100)),
    F("ear_angle", "Ângulo da curva da orelha (°)", 30.0, (0, 90)), F("panto", "Ângulo pantoscópico (°)", 5.0, (-20, 30)),
    F("wrap", "Ângulo de curvatura (°)", 0.0, (-30, 30)), F("screw_d", "Diâmetro do parafuso (mm)", 1.0, (0.3, 5)),
    F("tol", "Tolerância (mm)", -0.2, (-5, 5))])

GLIDER = _one(C3.glider, "Planador", [
    F("wing_logo", "Logótipo da asa (imagem)", "", IMG, group="Entrada"),
    F("tail_logo", "Logótipo da cauda (imagem)", "", IMG, group="Entrada"),
    F("fus_len", "Comprimento da fuselagem (mm)", 150.0, (30, 2000)), F("fus_w", "Largura da fuselagem (mm)", 2.0, (0.4, 50)),
    F("fus_h", "Altura da fuselagem (mm)", 3.0, (0.4, 100)), F("cockpit_h", "Altura da cabine (mm)", 10.0, (1, 200)),
    F("wing_angle", "Ângulo da asa (°)", 70.0, (10, 90)), F("wing_pos", "Posição da asa (0–1)", 0.5, (0, 1)),
    F("wing_len", "Envergadura (mm)", 80.0, (10, 2000)), F("wing_w", "Corda da asa (mm)", 30.0, (5, 500)),
    F("wing_t", "Espessura da asa (mm)", 0.8, (0.2, 10)), F("stab_len", "Estabilizador: comprimento (mm)", 30.0, (5, 500)),
    F("stab_w", "Estabilizador: largura (mm)", 20.0, (2, 500)), F("logo_t", "Espessura do logótipo (mm)", 0.4, (0.1, 5))])
