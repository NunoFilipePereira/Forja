"""Fatiador FDM com as opções habituais: paredes, camadas de topo e de fundo, enchimento (Line, Cross line,
Triangle, Square, Hexagon), costura Z (Default, Random, Corner), aderência (None, Skirt, Brim, Raft + expansão),
suportes (ângulo, densidade, folga XY, folga de topo em camadas, área mínima), modo "String" (fios dentro de uma
moldura) e G-code com extrusão E."""
import math
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, box as sbox
from shapely.ops import unary_union
from shapely import affinity

PATTERNS = ["Line", "Cross line", "Triangle", "Square", "Hexagon"]
SEAMS = ["Default", "Random", "Corner"]
ADHESION = ["None", "Skirt", "Brim", "Raft"]


def _sections(mesh, layer_h, first_h):
    z0, z1 = mesh.bounds[0][2], mesh.bounds[1][2]
    zs = [z0 + first_h / 2]
    while zs[-1] + layer_h < z1:
        zs.append(zs[-1] + layer_h)
    out = []
    for z in zs:
        s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if s is None:
            out.append((z, Polygon()))
            continue
        g = Polygon()
        for ent in s.discrete:
            P = np.asarray(ent)
            if len(P) >= 3:
                g = g.symmetric_difference(Polygon(P[:, :2]).buffer(0))
        out.append((z, g.buffer(0)))
    return out


def _lines_in(region, spacing, angle, offset=0.0):
    if region.is_empty or spacing <= 0:
        return []
    c = region.centroid
    r = affinity.rotate(region, -angle, origin=c)
    x0, y0, x1, y1 = r.bounds
    out = []
    flip = False
    y = y0 + (offset % spacing) + 1e-3
    while y < y1:
        seg = r.intersection(LineString([(x0 - 1, y), (x1 + 1, y)]))
        segs = list(seg.geoms) if hasattr(seg, "geoms") else ([seg] if not seg.is_empty else [])
        segs = [s for s in segs if s.geom_type == "LineString"]
        segs.sort(key=lambda s: s.bounds[0], reverse=flip)
        for s in segs:
            pts = np.array(s.coords)
            if flip:
                pts = pts[::-1]
            out.append(np.array(affinity.rotate(LineString(pts), angle, origin=c).coords))
        flip = not flip
        y += spacing
    return out


def _hex_lines(region, spacing, layer):
    """Favo de mel aproximado: zig-zags desfasados alternadamente."""
    if region.is_empty:
        return []
    x0, y0, x1, y1 = region.bounds
    a = spacing
    h = a * math.sqrt(3) / 2
    out = []
    y = y0
    k = 0
    while y < y1 + h:
        xs = np.arange(x0 - a, x1 + 2 * a, a / 2)
        ys = y + np.where((np.arange(len(xs)) + k + layer) % 2 == 0, 0, h * 0.5)
        line = LineString(np.c_[xs, ys]).intersection(region)
        for s in getattr(line, "geoms", [line]):
            if s.geom_type == "LineString" and not s.is_empty:
                out.append(np.array(s.coords))
        y += h
        k += 1
    return out


def _rings(g, seam="Default", rng=None):
    gs = g.geoms if hasattr(g, "geoms") else [g]
    out = []
    for p in gs:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        for ring in [p.exterior] + list(p.interiors):
            R = np.array(ring.coords)[:-1]
            if len(R) < 3:
                continue
            if seam == "Random" and rng is not None:
                k = int(rng.integers(len(R)))
            elif seam == "Corner":
                # canto mais vivo (maior mudança de direção)
                d1 = R - np.roll(R, 1, 0); d2 = np.roll(R, -1, 0) - R
                n1 = np.linalg.norm(d1, axis=1) + 1e-9; n2 = np.linalg.norm(d2, axis=1) + 1e-9
                k = int(np.argmin(np.sum(d1 * d2, 1) / n1 / n2))
            else:
                k = int(np.argmin(R[:, 0] + R[:, 1]))
            R = np.roll(R, -k, 0)
            out.append(np.vstack([R, R[:1]]))
    return out


def _infill(region, s, spacing, i, ang):
    pat = s.get("pattern", "Cross line")
    if pat == "Line":
        return _lines_in(region, spacing, ang)
    if pat == "Cross line":
        return _lines_in(region, spacing * 2, 45) + _lines_in(region, spacing * 2, -45)
    if pat == "Square":
        return _lines_in(region, spacing * 2, 0) + _lines_in(region, spacing * 2, 90)
    if pat == "Triangle":
        out = []
        for a in (0, 60, 120):
            out += _lines_in(region, spacing * 3, a)
        return out
    if pat == "Hexagon":
        return _hex_lines(region, spacing * 2.5, i)
    return _lines_in(region, spacing, ang)


def _frame_geom(shape, inner, clear, width, round_corner):
    from .cimg import regular
    x0, y0, x1, y1 = inner.bounds
    w, h = x1 - x0 + 2 * clear, y1 - y0 + 2 * clear
    if shape == "No frame":
        return Polygon(), inner.buffer(clear)
    if shape == "Square":
        w = h = max(w, h)
    k = {"Triangle": 2.1, "Circle": 1.45, "Pentagon": 1.3, "Hexagon": 1.2, "Octagon": 1.12}.get(shape, 1.0)
    inn = affinity.translate(regular(shape, w * k, h * k), (x0 + x1) / 2, (y0 + y1) / 2)
    if round_corner and shape not in ("Circle",):
        inn = inn.buffer(-2).buffer(2)
    return inn.buffer(width, join_style="round" if round_corner else "mitre").difference(inn), inn


def slice_mesh(mesh, s, fast=False):
    """s: definições. Devolve a lista de camadas: (z, [(tipo, polilinha XY)])."""
    lw = s["line_w"]; lh = s["layer_h"]
    secs = _sections(mesh, lh, s.get("first_h", lh))
    if fast:
        secs = secs[::max(1, len(secs) // 40)]
    n = len(secs)
    top_n = max(int(s.get("top_layers", s.get("top_bottom", 4))), 0)
    bot_n = max(int(s.get("bottom_layers", s.get("top_bottom", 4))), 0)
    rng = np.random.default_rng(3)
    infill_spacing = lw / max(s["infill"] / 100.0, 0.01) if s["infill"] > 0 else 0
    seam = s.get("seam", "Default")
    # suportes: saliências a partir do ângulo
    sup_need = [None] * n
    if s.get("support", "No") == "Yes":
        ang = math.radians(np.clip(s.get("support_angle", 50.0), 0, 89))
        allow = lh * math.tan(ang)
        over = [Polygon()] * n
        for i in range(1, n):
            if not secs[i][1].is_empty:
                below = secs[i - 1][1]
                over[i] = secs[i][1].difference(below.buffer(allow)).buffer(0) if not below.is_empty else secs[i][1]
        tg = max(int(s.get("support_top_gap", 1)), 0)
        acc = Polygon()
        for i in range(n - 1, -1, -1):
            if i + tg + 1 < n and not over[i + tg + 1].is_empty:
                acc = unary_union([acc, over[i + tg + 1]])
            if not acc.is_empty:
                need = acc.difference(secs[i][1].buffer(s.get("support_xy", 0.7) + lw / 2)).buffer(0)
                ma = s.get("support_min_area", 2.0)
                parts = [p for p in getattr(need, "geoms", [need]) if p.area >= ma]
                sup_need[i] = unary_union(parts) if parts else None
            acc = acc.difference(secs[i][1]).buffer(0) if not acc.is_empty else acc
    # modo "String": moldura à volta e fios a atravessar
    string = s.get("string", "No") == "Yes"
    if string:
        allg = unary_union([g for _, g in secs if not g.is_empty])
        frame, inner_f = _frame_geom(s.get("frame", "Rectangle"), allg, s.get("frame_clear", 5.0), s.get("frame_w", 4.0),
                                     s.get("round", "Yes") == "Yes")
    layers = []
    for i, (z, g) in enumerate(secs):
        paths = []
        if g.is_empty and not string:
            layers.append((z, paths)); continue
        if not g.is_empty:
            cur = g.buffer(-lw / 2)
            for w in range(int(s["walls"])):
                if cur.is_empty:
                    break
                for r in _rings(cur, seam, rng):
                    paths.append(("parede exterior" if w == 0 else "parede", r))
                cur = cur.buffer(-lw)
            inner = cur.buffer(lw / 2) if not cur.is_empty else cur
            solid = Polygon()
            if not inner.is_empty and (top_n or bot_n):
                if i < bot_n or i >= n - top_n:
                    solid = inner
                else:
                    cov_b = secs[i - bot_n][1] if bot_n else inner
                    for j in range(max(0, i - bot_n + 1), i):
                        cov_b = cov_b.intersection(secs[j][1])
                    cov_a = secs[i + 1][1] if top_n else inner
                    for j in range(i + 2, min(n, i + top_n + 1)):
                        cov_a = cov_a.intersection(secs[j][1])
                    solid = inner.difference(cov_b.intersection(cov_a)).buffer(0)
            ang = 45 if i % 2 == 0 else -45
            if not solid.is_empty:
                paths += [("sólido", p) for p in _lines_in(solid, lw, ang)]
            sparse = inner.difference(solid).buffer(0) if not inner.is_empty else inner
            if infill_spacing and not sparse.is_empty:
                paths += [("enchimento", p) for p in _infill(sparse, s, infill_spacing, i, ang)]
        if sup_need[i] is not None and not sup_need[i].is_empty:
            sp = lw / max(s.get("support_density", 10.0) / 100, 0.02)
            paths += [("suporte", p) for p in _lines_in(sup_need[i], sp, 0 if i % 2 == 0 else 90)]
        if string:
            for r in _rings(frame.buffer(-lw / 2)):
                paths.append(("moldura", r))
            dz = max(int(round(100 / max(s.get("string_dz", 5.0), 0.5))), 1)
            if i % dz == 0:
                sp = lw / max(s.get("string_dxy", 5.0) / 100, 0.01)
                region = inner_f.buffer(lw)
                pat = s.get("string_pattern", "Line in X")
                if pat == "Line in X":
                    sl = _lines_in(region, sp, 0)
                elif pat == "Line in Y":
                    sl = _lines_in(region, sp, 90)
                elif pat == "Grid":
                    sl = _lines_in(region, sp, 0 if (i // dz) % 2 == 0 else 90)
                else:
                    c = region.centroid
                    R = max(region.bounds[2] - region.bounds[0], region.bounds[3] - region.bounds[1])
                    k = max(int(2 * math.pi * R / 2 / sp), 6)
                    sl = []
                    for a in np.linspace(0, math.pi, k, endpoint=False):
                        ln = LineString([(c.x - R * math.cos(a), c.y - R * math.sin(a)), (c.x + R * math.cos(a), c.y + R * math.sin(a))]).intersection(region)
                        sl += [np.array(q.coords) for q in getattr(ln, "geoms", [ln]) if q.geom_type == "LineString" and not q.is_empty]
                paths += [("fio", p) for p in sl]
        if i == 0 and not g.is_empty:
            ad = s.get("adhesion", "Brim")
            ex = s.get("expansion", 3.0)
            if ad == "Skirt":
                for k in range(2):
                    paths = [("saia", r) for r in _rings(g.buffer(ex + k * lw))] + paths
            elif ad == "Brim":
                for k in range(max(int(ex / lw), 1)):
                    paths = [("aba", r) for r in _rings(g.buffer(lw * (k + 1)))] + paths
            elif ad == "Raft":
                paths = [("jangada", p) for p in _lines_in(g.buffer(ex), lw * 2, 0)] + paths
        layers.append((z, paths))
    return layers


def gcode(layers, s):
    fil_a = math.pi * (s["filament_d"] / 2) ** 2
    flow = s["flow"] / 100.0
    lw, lh = s["line_w"], s["layer_h"]
    F = s["speed"] * 60; T = s["travel"] * 60; R = s["retract_speed"] * 60
    L = [";Gerado pela Forja (fatiador)", f";Camadas: {len(layers)}", "G21", "G90", "M82",
         f"M140 S{s['bed_temp']:.0f}", f"M104 S{s['nozzle_temp']:.0f}", f"M190 S{s['bed_temp']:.0f}", f"M109 S{s['nozzle_temp']:.0f}",
         s.get("start_gcode", "G28"), "G92 E0", "M106 S255"]
    E = 0.0
    length = 0.0
    for li, (z, paths) in enumerate(layers):
        L.append(f";LAYER:{li}")
        L.append(f"G0 Z{z + lh / 2:.3f} F{T:.0f}")
        for kind, P in paths:
            if len(P) < 2:
                continue
            if s["retract"] > 0:
                E -= s["retract"]; L.append(f"G1 E{E:.5f} F{R:.0f}")
            L.append(f"G0 X{P[0][0]:.3f} Y{P[0][1]:.3f} F{T:.0f}")
            if s["retract"] > 0:
                E += s["retract"]; L.append(f"G1 E{E:.5f} F{R:.0f}")
            if kind == "fio":
                sp = s.get("string_speed", 25.0) * 60; fl = flow * s.get("string_flow", 200.0) / 100
            else:
                sp = F * (0.5 if li == 0 or kind == "parede exterior" else 1.0); fl = flow
            for a, b in zip(P[:-1], P[1:]):
                d = math.hypot(b[0] - a[0], b[1] - a[1])
                length += d
                E += d * lw * lh / fil_a * fl
                L.append(f"G1 X{b[0]:.3f} Y{b[1]:.3f} E{E:.5f} F{sp:.0f}")
    L += ["M107", "M104 S0", "M140 S0", s.get("end_gcode", "G28 X0 Y0"), "M84"]
    used_mm = E
    grams = used_mm * fil_a / 1000 * s.get("density", 1.24)
    t = length / max(s["speed"], 1) * 1.25
    stats = {"Camadas": len(layers), "Filamento (m)": f"{used_mm / 1000:.2f}", "Peso (g)": f"{grams:.1f}",
             "Tempo estimado": f"{t / 3600:.1f} h" if t > 3600 else f"{t / 60:.0f} min"}
    return "\n".join(L) + "\n", stats
