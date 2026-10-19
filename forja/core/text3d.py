"""Texto → polígonos (contornos das letras via Qt) → sólido 3D extrudido."""
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from PySide6.QtGui import QPainterPath, QFont


def text_polygons(text, font_family="Helvetica", size_mm=20.0, bold=False):
    """Devolve um MultiPolygon (mm) com o texto; altura das maiúsculas ≈ size_mm."""
    font = QFont(font_family)
    font.setPixelSize(100)
    font.setBold(bold)
    path = QPainterPath()
    y = 0.0
    lines = text.split("\n")
    for ln in lines:
        path.addText(0, y, font, ln)
        y += 120
    polys = []
    for qpoly in path.toFillPolygons():
        pts = [(p.x(), -p.y()) for p in qpoly]
        if len(pts) >= 3:
            polys.append(pts)
    # contornos exteriores e furos vêm misturados: regra par-ímpar (XOR)
    result = None
    for pts in polys:
        poly = Polygon(pts).buffer(0)
        if poly.area <= 1e-6:
            continue
        result = poly if result is None else result.symmetric_difference(poly)
    if result is None:
        return MultiPolygon()
    minx, miny, maxx, maxy = result.bounds
    k = size_mm / 72.0  # 100 px ≈ 72 px de altura de maiúscula
    from shapely import affinity
    result = affinity.translate(result, -minx, -miny)
    result = affinity.scale(result, k, k, origin=(0, 0))
    return result


def extrude(geom, height):
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    meshes = [trimesh.creation.extrude_polygon(g, height) for g in geoms if g.area > 0]
    if not meshes:
        raise ValueError("Sem geometria para extrudir.")
    return trimesh.util.concatenate(meshes)


def text_mesh(text, size_mm=20, depth=5, font="Helvetica", bold=True):
    return extrude(text_polygons(text, font, size_mm, bold), depth)
