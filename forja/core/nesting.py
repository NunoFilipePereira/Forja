"""Nesting 2D: encaixar peças em chapas (heurística bottom-left com rotações)."""
import numpy as np
from shapely import affinity
from shapely.prepared import prep
from shapely.geometry import box
from shapely.strtree import STRtree


def _normalize(g):
    x0, y0, _, _ = g.bounds
    return affinity.translate(g, -x0, -y0)


def nest(items, plate_w, plate_h, spacing=2.0, margin=2.0, rotations=(0, 90, 180, 270),
         max_plates=20, progress=None):
    """items: lista de (id, geometria shapely fechada).
    Devolve lista de chapas: [{"placements": [(id, geom_colocada, angulo)], "usage": %}] e não colocados."""
    order = sorted(items, key=lambda it: -it[1].area)
    plates = []
    unplaced = []
    usable = box(margin, margin, plate_w - margin, plate_h - margin)
    for n, (iid, geom) in enumerate(order):
        if progress:
            progress(n, len(order))
        variants = []
        for a in rotations:
            g = _normalize(affinity.rotate(geom, a, origin="centroid"))
            variants.append((a, g))
        placed = False
        for plate in plates + [None]:
            if plate is None:
                if len(plates) >= max_plates:
                    break
                plate = {"placements": [], "geoms": []}
                new = True
            else:
                new = False
            pos = _find_position(plate, variants, plate_w, plate_h, spacing, margin, usable)
            if pos is not None:
                a, g = pos
                plate["placements"].append((iid, g, a))
                plate["geoms"].append(g.buffer(spacing / 2, join_style="mitre"))
                if new:
                    plates.append(plate)
                placed = True
                break
            if new:
                break
        if not placed:
            unplaced.append(iid)
    for p in plates:
        area = sum(g.area for _, g, _ in p["placements"])
        p["usage"] = 100.0 * area / (plate_w * plate_h)
        del p["geoms"]
    return plates, unplaced


def _find_position(plate, variants, W, H, spacing, margin, usable):
    geoms = plate["geoms"]
    tree = STRtree(geoms) if geoms else None
    xs = {margin} | {g.bounds[2] + spacing / 2 for g in geoms}
    ys = {margin} | {g.bounds[3] + spacing / 2 for g in geoms}
    # também posições alinhadas pelos lados esquerdo/inferior das peças existentes
    xs |= {g.bounds[0] + spacing / 2 for g in geoms}
    ys |= {g.bounds[1] + spacing / 2 for g in geoms}
    cands = sorted((y, x) for x in xs for y in ys)
    best = None
    for a, g in variants:
        gw, gh = g.bounds[2], g.bounds[3]
        if gw > W - 2 * margin + 1e-9 or gh > H - 2 * margin + 1e-9:
            continue
        for y, x in cands:
            if x + gw > W - margin + 1e-9 or y + gh > H - margin + 1e-9:
                continue
            if best is not None and (y, x) >= best[0]:
                break
            cand = affinity.translate(g, x, y)
            cb = cand.buffer(spacing / 2 - 1e-6, join_style="mitre") if spacing > 0 else cand
            if tree is not None:
                hits = tree.query(cb, predicate="intersects")
                if len(hits):
                    continue
            # tenta deslizar para baixo/esquerda (compactação)
            cand = _slide(cand, tree, geoms, spacing, margin)
            key = (cand.bounds[1], cand.bounds[0])
            if best is None or key < best[0]:
                best = (key, a, cand)
            break
    if best is None:
        return None
    return best[1], best[2]


def _slide(g, tree, geoms, spacing, margin, step=1.0):
    for dx, dy in ((0, -1), (-1, 0), (0, -1), (-1, 0)):
        moved = g
        s = step * 8
        while s >= 0.25:
            trial = affinity.translate(moved, dx * s, dy * s)
            ok = trial.bounds[0] >= margin - 1e-9 and trial.bounds[1] >= margin - 1e-9
            if ok and tree is not None:
                cb = trial.buffer(spacing / 2 - 1e-6, join_style="mitre") if spacing > 0 else trial
                ok = not len(tree.query(cb, predicate="intersects"))
            if ok:
                moved = trial
            else:
                s /= 2
        g = moved
    return g


def report(plates, unplaced, W, H):
    lines = []
    for i, p in enumerate(plates, 1):
        ids = ", ".join(f"{iid} ({g.area:.4g})" for iid, g, _ in p["placements"])
        lines.append(f"Chapa {i}: {W:g}×{H:g},  Peças: {len(p['placements'])},  "
                     f"Aproveitamento: {p['usage']:.4g}%.  IDs (área): {ids}")
    if unplaced:
        lines.append(f"Não colocadas (maiores que a chapa ou sem espaço): {', '.join(map(str, unplaced))}")
    return "\n".join(lines)
