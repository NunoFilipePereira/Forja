"""Ferramenta Slicer (fatiador) no painel lateral, com as páginas e os campos do fatiador:
Model, Printer, Material, Layer, Adhesion, Support, String e Preview. Gravar = fatiar e guardar o G-code."""
import numpy as np
import trimesh
from ..core import slicer as SL

PAGES = ["Model", "Printer", "Material", "Layer", "Adhesion", "Support", "String", "Preview"]
YN = ("radio", ["Yes", "No"])
LABELS = {"Model": "Modelo", "Printer": "Impressora", "Material": "Material", "Layer": "Camada", "Adhesion": "Aderência",
          "Support": "Suporte", "String": "Fios (string)", "Preview": "Pré-visualização", "Yes": "Sim", "No": "Não",
          "Rectangle": "Retângulo", "Circle": "Círculo", "Line": "Linhas", "Cross line": "Linhas cruzadas", "Triangle": "Triângulos",
          "Square": "Quadrados", "Hexagon": "Hexágonos", "Default": "Predefinida", "Random": "Aleatória", "Corner": "Canto",
          "None": "Nenhuma", "Skirt": "Saia", "Brim": "Aba (brim)", "Raft": "Jangada (raft)", "No frame": "Sem moldura",
          "Pentagon": "Pentágono", "Octagon": "Octógono", "Line in X": "Linhas em X", "Line in Y": "Linhas em Y", "Grid": "Grelha",
          "Radial": "Radial", "One layer": "Uma camada", "All layers": "Todas as camadas", "Up to layer": "Até à camada"}


def F(k, l, d, o, page):
    return (k, l, d, o, page, {page})


def reset_model(panel):
    """Reset model: repõe o tamanho original, sem rotação nem translação."""
    e = panel.obj.mesh.extents if panel.obj is not None else (100, 100, 100)
    for k, val in (("sx", e[0]), ("sy", e[1]), ("sz", e[2]), ("rx", 0), ("ry", 0), ("rz", 0), ("tx", 0), ("ty", 0), ("tz", 0)):
        w = panel.widgets[k]; w.blockSignals(True); w.setValue(float(val)); w.blockSignals(False)
    w = panel.widgets["uniform"]._rgroup.buttons()[0]; w.setChecked(True)
    panel.changed()


FIELDS = [
    F("uniform", "Escala uniforme", "Yes", YN, "Model"),
    F("sx", "Tamanho do modelo X (mm)", 100.0, (0.1, 5000), "Model"), F("sy", "Tamanho do modelo Y (mm)", 100.0, (0.1, 5000), "Model"),
    F("sz", "Tamanho do modelo Z (mm)", 100.0, (0.1, 5000), "Model"),
    F("rx", "Rodar eixo X (°)", 0.0, (-360, 360), "Model"), F("ry", "Rodar eixo Y (°)", 0.0, (-360, 360), "Model"),
    F("rz", "Rodar eixo Z (°)", 0.0, (-360, 360), "Model"),
    F("tx", "Translação X (mm)", 0.0, (-1e4, 1e4), "Model"), F("ty", "Translação Y (mm)", 0.0, (-1e4, 1e4), "Model"),
    F("tz", "Translação Z (mm)", 0.0, (-1e4, 1e4), "Model"),
    F("_reset", "", "Repor modelo (Reset model)", ("button", reset_model), "Model"),
    F("bed", "Forma da mesa", "Rectangle", ["Rectangle", "Circle"], "Printer"),
    F("px", "Impressora X (mm)", 200.0, (10, 2000), "Printer"), F("py", "Impressora Y (mm)", 150.0, (10, 2000), "Printer"),
    F("pz", "Impressora Z (mm)", 140.0, (10, 2000), "Printer"), F("line_w", "Largura da linha (mm)", 0.4, (0.1, 2), "Printer"),
    F("start_gcode", "G-code inicial", "G28 ; origem\nG1 Z5 F3000", None, "Printer"),
    F("end_gcode", "G-code final", "G91\nG1 Z10\nG90\nG28 X0 Y0", None, "Printer"),
    F("filament_d", "Diâmetro do filamento (mm)", 1.75, (1, 3), "Material"), F("flow", "Caudal (%)", 100.0, (10, 300), "Material"),
    F("bed_temp", "Temperatura da mesa (°C)", 60.0, (0, 150), "Material"),
    F("nozzle_temp", "Temperatura do extrusor (°C)", 190.0, (0, 350), "Material"),
    F("speed", "Velocidade de impressão (mm/s)", 50.0, (1, 500), "Material"), F("travel", "Velocidade em vazio (mm/s)", 80.0, (1, 1000), "Material"),
    F("retract_speed", "Velocidade de retração (mm/s)", 25.0, (1, 200), "Material"),
    F("retract", "Distância de retração (mm)", 6.0, (0, 20), "Material"),
    F("density", "Densidade do filamento (g/cm³)", 1.24, (0.5, 5), "Material"),
    F("layer_h", "Altura da camada (mm)", 0.2, (0.04, 1.0), "Layer"),
    F("walls", "Número de paredes", 2, (1, 20), "Layer"),
    F("top_layers", "Camadas de topo", 4, (0, 50), "Layer"), F("bottom_layers", "Camadas de fundo", 4, (0, 50), "Layer"),
    F("infill", "Enchimento (%)", 10.0, (0, 100), "Layer"),
    F("pattern", "Padrão de enchimento", "Cross line", SL.PATTERNS, "Layer"),
    F("seam", "Costura Z", "Default", SL.SEAMS, "Layer"),
    F("adhesion", "Tipo", "Brim", SL.ADHESION, "Adhesion"),
    F("expansion", "Expansão (mm)", 3.0, (0, 50), "Adhesion"),
    F("support", "Suporte", "No", YN, "Support"),
    F("support_angle", "Ângulo do suporte (°)", 50.0, (0, 89), "Support"),
    F("support_density", "Densidade (%)", 10.0, (1, 100), "Support"),
    F("support_xy", "Folga XY (mm)", 0.7, (0, 10), "Support"),
    F("support_top_gap", "Folga de topo (camadas)", 1, (0, 20), "Support"),
    F("support_min_area", "Área mínima (mm²)", 2.0, (0, 1000), "Support"),
    F("string", "Fios (string)", "No", YN, "String"),
    F("frame", "Forma da moldura", "Rectangle", ["No frame", "Triangle", "Rectangle", "Square", "Pentagon", "Hexagon", "Octagon", "Circle"], "String"),
    F("round", "Cantos arredondados", "Yes", YN, "String"),
    F("frame_clear", "Folga da moldura (mm)", 5.0, (0, 100), "String"), F("frame_w", "Largura da moldura (mm)", 4.0, (0.4, 50), "String"),
    F("string_pattern", "Padrão", "Line in X", ["Line in X", "Line in Y", "Grid", "Radial"], "String"),
    F("string_dxy", "Densidade em XY (%)", 5.0, (0.5, 100), "String"), F("string_dz", "Densidade em Z (%)", 5.0, (0.5, 100), "String"),
    F("string_flow", "Caudal (%)", 200.0, (10, 500), "String"), F("string_speed", "Velocidade de impressão (mm/s)", 25.0, (1, 300), "String"),
    F("show", "Mostrar", "All layers", ["One layer", "Up to layer", "All layers"], "Preview"),
    F("layer", "Camada", 1, (1, 100000), "Preview"),
    F("downsample", "Reduzir (mostrar 1 em cada N caminhos)", 1, (1, 50), "Preview"),
]


def _prepared(mesh, v):
    m = mesh.copy()
    R = trimesh.transformations.euler_matrix(*np.radians([v["rx"], v["ry"], v["rz"]]))
    m.apply_transform(R)
    e = np.maximum(m.extents, 1e-9)
    want = np.array([v["sx"], v["sy"], v["sz"]])
    k = want / e
    if v.get("uniform", "Yes") == "Yes":
        i = int(np.argmax(np.abs(k - 1)))
        k = np.full(3, k[i])
    m.apply_scale(k)
    m.apply_translation((-m.bounds.mean(axis=0)[0] + v["px"] / 2 + v["tx"], -m.bounds.mean(axis=0)[1] + v["py"] / 2 + v["ty"],
                         -m.bounds[0][2] + v["tz"]))
    return m


def run(mesh, v, fast):
    v = dict(v, first_h=v["layer_h"] * 1.25)
    m = _prepared(mesh, v)
    e = m.extents
    warn = {}
    if v["bed"] == "Circle":
        r = np.max(np.linalg.norm(m.vertices[:, :2] - np.array([v["px"] / 2, v["py"] / 2]), axis=1))
        if r > min(v["px"], v["py"]) / 2 or e[2] > v["pz"]:
            warn["Aviso"] = "o modelo não cabe na impressora"
    elif e[0] > v["px"] or e[1] > v["py"] or e[2] > v["pz"]:
        warn["Aviso"] = "o modelo não cabe na impressora"
    layers = SL.slice_mesh(m, v, fast=fast)
    code, stats = SL.gcode(layers, v)
    n = len(layers)
    k = int(np.clip(v.get("layer", 1), 1, max(n, 1))) - 1
    sel = {"One layer": layers[k:k + 1], "Up to layer": layers[:k + 1]}.get(v.get("show", "All layers"), layers)
    ds = max(int(v.get("downsample", 1)), 1)
    lines = []
    c = 0
    for z, paths in sel:
        for kind, P in paths:
            c += 1
            if len(P) >= 2 and c % ds == 0:
                lines.append((kind, np.c_[P, np.full(len(P), z)]))
    stats.update(warn)
    stats["Tamanho"] = f"{e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm"
    if fast:
        stats["(pré-visualização)"] = "a amostrar camadas"
    return {"meshes": [], "lines": lines, "measure": stats, "gcode": code}


def save(win, result, v):
    import os
    from PySide6.QtWidgets import QFileDialog
    p, _ = QFileDialog.getSaveFileName(win, "Guardar G-code", os.path.join(win.last_dir, "modelo.gcode"), "G-code (*.gcode)")
    if p:
        open(p, "w").write(result["gcode"])
        win.log(f"G-code guardado: {p} — " + ", ".join(f"{k} {val}" for k, val in result["measure"].items()))


SPEC = {"title": "Fatiador (G-code)", "methods": PAGES, "fields": FIELDS, "run": run, "save": save, "save_label": "Fatiar e gravar G-code",
        "model_size": False, "labels": LABELS, "pages": True}
