"""Leitura e escrita do ficheiro de figura em texto (.fgs, "CLJ++ FIG token file"): definições da vista
(ângulo, fundo, eixos, barra de cores, iluminação…) e, por peça (BEGIN_BASICSHAPE … END_BASICSHAPE), nome, cores,
transparência, estilos de ponto/linha e o que se mostra. A geometria fica nos STL gravados ao lado."""
import glob
import os
import numpy as np

# índices das tabelas de cor → escalas da Forja (o 11 e o −1 são os valores por omissão)
TABLE = {-1: "Solid color", 0: "True color", 1: "Gray", 2: "Rainbow 1", 3: "Rainbow 2", 4: "Hot metal 1", 5: "Hot metal 2",
         6: "RGB", 7: "High contrast", 11: "Solid color"}
TABLE_INV = {"Solid color": -1, "True color": 0, "Gray": 1, "Rainbow 1": 2, "Rainbow 2": 3, "Hot metal 1": 4, "Hot metal 2": 5,
             "RGB": 6, "High contrast": 7}
MARK = {0: "Dot", 1: "Dot", 2: "x", 3: "+", 4: "Circle", 5: "Vertical diamond", 6: "Horizontal diamond", 7: "Upward triangle",
        8: "Downward triangle", 9: "Star"}
MARK_INV = {v: k for k, v in MARK.items() if k}
DASH = {0: "Solid line", 1: "Dotted line", 2: "Long dash", 3: "Short dash", 4: "Dot dash"}
DASH_INV = {v: k for k, v in DASH.items()}


def read_fgs(path):
    view, shapes, cur = {}, [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("BEGIN_BASICSHAPE"):
            cur = {}; continue
        if line.startswith("END_BASICSHAPE"):
            if cur is not None:
                shapes.append(cur)
            cur = None; continue
        key, _, val = line.partition(" ")
        (cur if cur is not None else view)[key] = val.strip()
    return {"view": view, "shapes": shapes}


def find_stl(folder, name):
    for pat in (f"{name}.stl", f"Shape_STL/{name}.stl", f"*/{name}.stl", f"{name}*.stl"):
        fs = glob.glob(os.path.join(folder, pat))
        if fs:
            return fs[0]
    return None


def _nums(s, n=None):
    try:
        v = [float(x) for x in str(s).split()]
    except ValueError:
        return []
    return v if n is None else v[:n]


def apply_shape_style(o, sh):
    rgb = _nums(sh.get("SurfRGB", ""), 3)
    if len(rgb) == 3:
        o.color = tuple(np.asarray(rgb) / 255.0)
    a = _nums(sh.get("Alpha", "1"), 1)
    if a:
        o.alpha = float(np.clip(a[0], 0.05, 1))
    st = {"surface": {"show": sh.get("ShowSurf", "1") != "0", "scheme": TABLE.get(int(_nums(sh.get("SurfColorTable", "-1"), 1)[0] if _nums(sh.get("SurfColorTable", "-1"), 1) else -1), "Solid color")},
          "edge": {"show": sh.get("ShowEdge", "0") != "0", "scheme": TABLE.get(int((_nums(sh.get("EdgeColorTable", "11"), 1) or [11])[0]), "Solid color"),
                   "dash": DASH.get(int((_nums(sh.get("LineStyle", "0"), 1) or [0])[0]), "Solid line"),
                   "width": float((_nums(sh.get("LineWidth", "1"), 1) or [1])[0])},
          "point": {"show": sh.get("ShowPoint", "0") != "0", "scheme": TABLE.get(int((_nums(sh.get("PointColorTable", "11"), 1) or [11])[0]), "Solid color"),
                    "marker": MARK.get(int((_nums(sh.get("PointStyle", "1"), 1) or [1])[0]), "Dot"),
                    "size": float((_nums(sh.get("PointSize", "4"), 1) or [4])[0])},
          "sampling": int((_nums(sh.get("Sample", "1"), 1) or [1])[0])}
    if st["surface"]["scheme"] == "Solid color" and not st["edge"]["show"] and not st["point"]["show"] and st["surface"]["show"]:
        o.style = None  # aspeto normal
    else:
        o.style = st


def apply_view(v3, view):
    from PySide6.QtGui import QColor
    ang = _nums(view.get("ViewingAngle", ""), 3)
    if len(ang) >= 2:
        # elevação, azimute (graus)
        v3.setCameraPosition(elevation=ang[0], azimuth=ang[1] - 90)
    bk = _nums(view.get("BKRGB", ""), 3)
    if len(bk) == 3:
        v3.set_background(QColor(int(bk[0]), int(bk[1]), int(bk[2])))
    if view.get("OrthoProj", "0") == "1" and hasattr(v3, "set_ortho"):
        v3.set_ortho(True)
    ax = _nums(view.get("Axis", ""), 1)
    if ax and hasattr(v3, "axis"):
        v3.axis.setVisible(bool(ax[0]))


def write_fgs(folder, objs, v3=None):
    """Grava Forja.fgs + um STL por objeto na pasta (o .fgd binário não é gerado)."""
    import re
    lines = ["CLJ++ FIG token file", "DATAFILE Forja.fgd", "", "Maximized 1", "WinSize 1271 843", "Viewport 0 0 1 1"]
    ortho = int(bool(getattr(v3, "ortho", False))) if v3 is not None else 0
    el = v3.opts["elevation"] if v3 is not None else 30
    az = v3.opts["azimuth"] + 90 if v3 is not None else -15
    bg = v3.bg if v3 is not None and hasattr(v3, "bg") else None
    lines += [f"OrthoProj {ortho}", "FixedAspect 1", f"ViewingAngle {el:g} {az:g} 0",
              f"BKRGB {bg.red() if bg else 0} {bg.green() if bg else 0} {bg.blue() if bg else 0}",
              "Axis 1 255 255 255", "Colorbar 0 0.5 0.5 10 200", "Legend 0 0.5 0.5 10 70 90 400 0 Arial", "Lighting 1 90 0 0.1 0.1 0.1 1",
              "XTitle X", "YTitle Z", "ZTitle Y", "XTicks ;", "YTicks ;", "ZTicks ;", "Grid 1 1 1", "AutoTick 1 1 1", "AutoLim 1 1 1", ""]
    for o in objs:
        name = re.sub(r"[^\w\-.]+", "_", o.name).strip("_") or f"Part_{o.id}"
        o.mesh.export(os.path.join(folder, name + ".stl"))
        st = getattr(o, "style", None) or {}
        sf, ed, pt = st.get("surface", {}), st.get("edge", {}), st.get("point", {})
        r, g, b = (np.asarray(o.color[:3]) * 255).astype(int)
        lines += ["BEGIN_BASICSHAPE", f"Name {name}", f"Sample {int(st.get('sampling', 1))}",
                  f"PointStyle {MARK_INV.get(pt.get('marker', 'Dot'), 1)}", f"PointSize {pt.get('size', 4):g}",
                  f"LineStyle {DASH_INV.get(ed.get('dash', 'Solid line'), 0)}", f"LineWidth {ed.get('width', 1):g}",
                  "BarWidth 1", "3DTxtSize 0.05", f"ShowPoint {int(bool(pt.get('show', False)))}",
                  f"ShowEdge {int(bool(ed.get('show', False)))}", f"ShowSurf {int(bool(sf.get('show', True)))}", "ShowTex 0",
                  f"PointColorTable {TABLE_INV.get(pt.get('scheme', 'Solid color'), 11) if pt.get('scheme') not in (None, 'Solid color') else 11}",
                  f"EdgeColorTable {TABLE_INV.get(ed.get('scheme', 'Solid color'), 11) if ed.get('scheme') not in (None, 'Solid color') else 11}",
                  f"SurfColorTable {TABLE_INV.get(sf.get('scheme', 'Solid color'), -1)}",
                  "PointRGB 255 255 255", "EdgeRGB 255 255 255", f"SurfRGB {r} {g} {b}", f"Alpha {o.alpha:g}",
                  "ClipBox 0 1 0 1 0 1", "VRMode 1", "END_BASICSHAPE", ""]
    open(os.path.join(folder, "Forja.fgs"), "w").write("\n".join(lines) + "\n")
