"""Tela 2D: desenho de formas, seleção, régua e transferidor, sobreposições (nesting/percursos)."""
import math
import numpy as np
from shapely.geometry import Point, box as sbox
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainterPath, QPen, QBrush, QColor, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPathItem, QInputDialog
from ..core import shapes2d as s2

TOOLS = {
    "select": "Selecionar", "pt": "Ponto", "ln": "Linha", "rc": "Retângulo", "rcr": "Retângulo rodado",
    "cir": "Círculo", "ell": "Elipse", "pol": "Polígono", "pl": "Polilinha", "spl": "Spline",
    "arr": "Seta", "txt": "Texto", "ruler": "Régua", "protractor": "Transferidor",
    "arc": "Arco (3 pontos)", "ff": "Mão livre",
}
DRAG_TOOLS = {"ln", "rc", "rcr", "cir", "ell", "arr", "ruler"}
CLICK_TOOLS = {"pol", "pl", "spl", "protractor", "arc"}
DASHES = {"contínua": Qt.SolidLine, "tracejada": Qt.DashLine, "pontilhada": Qt.DotLine, "traço-ponto": Qt.DashDotLine,
          "traço longo": Qt.CustomDashLine}


def geom_to_path(g):
    path = QPainterPath()
    path.setFillRule(Qt.OddEvenFill)
    if g.geom_type == "Point":
        path.addEllipse(QPointF(g.x, g.y), 0.8, 0.8)
        return path
    for pts in s2.iter_paths(g):
        if len(pts) < 2:
            continue
        path.moveTo(*pts[0][:2])
        for x, y in pts[1:, :2]:
            path.lineTo(x, y)
    return path


class Canvas2D(QGraphicsView):
    changed = Signal()
    message = Signal(str)
    selection_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.scene().setSceneRect(-5000, -5000, 10000, 10000)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.scale(3, -3)
        self.centerOn(100, 60)
        self.shapes: list[s2.Shape] = []
        self.selected: set[int] = set()
        self.items = {}
        self.overlay_items = []
        self.tool = "select"
        self.snap = True
        self.grid_step = 10.0
        self.undo_stack = []
        self._start = None
        self._pts = []
        self._preview = None
        self._moving = False
        self._pan = None
        self.bg = QColor(250, 250, 248)
        self.unit, self.unit_k = "mm", 1.0

    # --- modelo -------------------------------------------------------
    def push_undo(self):
        self.undo_stack.append([s.copy() for s in self.shapes])
        self.undo_stack = self.undo_stack[-50:]

    def undo(self):
        if self.undo_stack:
            self.shapes = self.undo_stack.pop()
            self.selected.clear()
            self.redraw()

    def add_shapes(self, shapes, select=True):
        self.push_undo()
        self.shapes += shapes
        if select:
            self.selected = {s.id for s in shapes}
        self.redraw()

    def replace_selected(self, new_shapes):
        self.push_undo()
        keep = [s for s in self.shapes if s.id not in self.selected]
        self.shapes = keep + new_shapes
        self.selected = {s.id for s in new_shapes}
        self.redraw()

    def selected_shapes(self):
        return [s for s in self.shapes if s.id in self.selected]

    def delete_selected(self):
        if self.selected:
            self.push_undo()
            self.shapes = [s for s in self.shapes if s.id not in self.selected]
            self.selected.clear()
            self.redraw()

    def select_all(self):
        self.selected = {s.id for s in self.shapes}
        self.redraw()

    # --- desenho ------------------------------------------------------
    def redraw(self):
        sc = self.scene()
        for it in self.items.values():
            sc.removeItem(it)
        self.items = {}
        for z, s in enumerate(self.shapes):
            if getattr(s, "hidden", False):
                continue
            sel = s.id in self.selected
            st = getattr(s, "style", {}) or {}
            path = geom_to_path(s.geom)
            if (st.get("arrow_start") or st.get("arrow_end")) and s.geom.geom_type == "LineString":
                path.addPath(self._arrow_heads(np.array(s.geom.coords), st))
            it = QGraphicsPathItem(path)
            col = QColor(st["stroke"]) if st.get("stroke") else QColor(30, 60, 120)
            pen = QPen(QColor(230, 120, 20) if sel else col)
            pen.setCosmetic(True); pen.setWidthF((st.get("width") or 1.3) + (0.9 if sel else 0))
            dash = st.get("dash", "contínua")
            pen.setStyle(DASHES.get(dash, Qt.SolidLine))
            if dash == "traço longo":
                pen.setDashPattern([10, 4])
            it.setPen(pen)
            if s.closed or s.geom.geom_type == "Point":
                if st.get("fill"):
                    fc = QColor(st["fill"]); fc.setAlpha(160)
                    it.setBrush(QBrush(fc))
                else:
                    it.setBrush(QBrush(QColor(230, 120, 20, 50) if sel else QColor(60, 110, 200, 35)))
            it.setZValue(1 + z * 1e-4 + (1 if sel else 0))
            sc.addItem(it)
            self.items[s.id] = it
        self.changed.emit()
        self.selection_changed.emit()

    def _arrow_heads(self, pts, st):
        path = QPainterPath()
        L = max(2.0, 12 / max(self.transform().m11(), 1e-6))
        ends = []
        if st.get("arrow_end") and len(pts) > 1:
            ends.append((pts[-1], pts[-2]))
        if st.get("arrow_start") and len(pts) > 1:
            ends.append((pts[0], pts[1]))
        for tip, prev in ends:
            d = tip[:2] - prev[:2]; n = np.linalg.norm(d)
            if n == 0:
                continue
            d /= n; nrm = np.array([-d[1], d[0]])
            a = tip[:2] - d * L + nrm * L * 0.4; b = tip[:2] - d * L - nrm * L * 0.4
            path.moveTo(*tip[:2]); path.lineTo(*a); path.lineTo(*b); path.closeSubpath()
        return path

    def drawForeground(self, p, rect):
        """Réguas graduadas nas margens superior e esquerda (em mm)."""
        if not getattr(self, "show_rulers", True):
            return
        p.save()
        p.resetTransform()
        vw, vh = self.viewport().width(), self.viewport().height()
        p.fillRect(0, 0, vw, 18, QColor(240, 240, 236, 230)); p.fillRect(0, 0, 18, vh, QColor(240, 240, 236, 230))
        scale = self.transform().m11()
        step = next(m * 10 ** k for k in range(-3, 7) for m in (1, 2, 5) if m * 10 ** k * scale >= 40)
        tl = self.mapToScene(0, 0); br = self.mapToScene(vw, vh)
        pen = QPen(QColor(90, 90, 90)); p.setPen(pen)
        f = p.font(); f.setPointSizeF(8); p.setFont(f)
        x = math.floor(tl.x() / step) * step
        while x <= br.x():
            px = self.mapFromScene(QPointF(x, 0)).x()
            p.drawLine(px, 12, px, 18); p.drawText(px + 2, 10, f"{x:g}")
            x += step
        y0, y1 = sorted((tl.y(), br.y()))
        y = math.floor(y0 / step) * step
        while y <= y1:
            py = self.mapFromScene(QPointF(0, y)).y()
            p.drawLine(12, py, 18, py)
            p.save(); p.translate(10, py - 2); p.rotate(-90); p.drawText(0, 0, f"{y:g}"); p.restore()
            y += step
        p.restore()

    def clear_overlay(self):
        for it in self.overlay_items:
            self.scene().removeItem(it)
        self.overlay_items = []

    def add_overlay_paths(self, paths, color=QColor(220, 30, 30), width=1.0):
        path = QPainterPath()
        for p in paths:
            if len(p) < 2:
                continue
            path.moveTo(*p[0][:2])
            for x, y in p[1:, :2]:
                path.lineTo(x, y)
        it = QGraphicsPathItem(path)
        pen = QPen(color); pen.setCosmetic(True); pen.setWidthF(width)
        it.setPen(pen); it.setZValue(5)
        self.scene().addItem(it)
        self.overlay_items.append(it)

    def add_overlay_rect(self, x, y, w, h, color=QColor(120, 120, 120)):
        it = QGraphicsPathItem()
        p = QPainterPath(); p.addRect(QRectF(x, y, w, h)); it.setPath(p)
        pen = QPen(color); pen.setCosmetic(True); pen.setStyle(Qt.DashLine)
        it.setPen(pen); it.setBrush(QBrush(QColor(200, 200, 200, 40))); it.setZValue(0)
        self.scene().addItem(it)
        self.overlay_items.append(it)

    def drawBackground(self, p, rect):
        p.fillRect(rect, self.bg)
        scale = self.transform().m11()
        step = self.grid_step
        while step * scale < 8:
            step *= 5
        x0 = math.floor(rect.left() / step) * step
        y0 = math.floor(rect.top() / step) * step
        pen = QPen(QColor(225, 225, 222)); pen.setCosmetic(True)
        p.setPen(pen)
        x = x0
        while x < rect.right():
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom())); x += step
        y = y0
        while y < rect.bottom():
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y)); y += step
        pen = QPen(QColor(200, 80, 80)); pen.setCosmetic(True); p.setPen(pen)
        p.drawLine(QPointF(0, rect.top()), QPointF(0, rect.bottom()))
        p.drawLine(QPointF(rect.left(), 0), QPointF(rect.right(), 0))

    def fit(self):
        if not self.shapes:
            return
        b = np.array([s.geom.bounds for s in self.shapes])
        r = QRectF(b[:, 0].min(), b[:, 1].min(), b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min())
        self.fitInView(r.adjusted(-10, -10, 10, 10), Qt.KeepAspectRatio)

    # --- interação ----------------------------------------------------
    def set_tool(self, t):
        self.tool = t
        self._cancel()
        self.setCursor(Qt.ArrowCursor if t == "select" else Qt.CrossCursor)
        self.message.emit(f"Ferramenta: {TOOLS[t]}" + ("  (clique para pontos; duplo clique/Enter termina; Esc cancela)"
                                                      if t in CLICK_TOOLS else ""))

    def _p(self, e):
        p = self.mapToScene(e.position().toPoint())
        x, y = p.x(), p.y()
        if self.snap and not (e.modifiers() & Qt.AltModifier):
            g = 1.0 if self.transform().m11() > 2 else 5.0
            x, y = round(x / g) * g, round(y / g) * g
        return x, y

    def _p_raw(self, e):
        p = self.mapToScene(e.position().toPoint())
        return (p.x(), p.y())

    def _cancel(self):
        self._start = None
        self._pts = []
        if self._preview is not None:
            self.scene().removeItem(self._preview)
            self._preview = None

    def _set_preview(self, geom_or_path):
        if self._preview is None:
            self._preview = QGraphicsPathItem()
            pen = QPen(QColor(20, 160, 90)); pen.setCosmetic(True); pen.setStyle(Qt.DashLine)
            self._preview.setPen(pen); self._preview.setZValue(10)
            self.scene().addItem(self._preview)
        path = geom_or_path if isinstance(geom_or_path, QPainterPath) else geom_to_path(geom_or_path)
        self._preview.setPath(path)

    def _shape_from_drag(self, a, b, final=False):
        (x1, y1), (x2, y2) = a, b
        t = self.tool
        if t == "ln" or t == "ruler":
            return s2.Shape("ln", dict(x1=x1, y1=y1, x2=x2, y2=y2))
        if t == "arr":
            if (x1, y1) == (x2, y2):
                return None
            return s2.Shape("arr", dict(x1=x1, y1=y1, x2=x2, y2=y2))
        if t in ("rc", "rcr"):
            if x1 == x2 or y1 == y2:
                return None
            return s2.Shape("rc", dict(left=min(x1, x2), bottom=min(y1, y2), right=max(x1, x2), top=max(y1, y2)))
        if t == "cir":
            r = math.hypot(x2 - x1, y2 - y1)
            return s2.Shape("cir", dict(cx=x1, cy=y1, r=r)) if r > 0 else None
        if t == "ell":
            rx, ry = abs(x2 - x1) / 2, abs(y2 - y1) / 2
            if rx == 0 or ry == 0:
                return None
            return s2.Shape("ell", dict(cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, rx=rx, ry=ry, angle=0))
        return None

    def _pick(self, x, y):
        tol = 6 / self.transform().m11()
        pt = Point(x, y)
        best = None
        for s in reversed(self.shapes):
            d = s.geom.distance(pt) if not s.closed else (0 if s.geom.contains(pt) else s.geom.distance(pt))
            if d <= tol and (best is None or d < best[0]):
                best = (d, s)
        return best[1] if best else None

    def mousePressEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton) or (e.button() == Qt.LeftButton and e.modifiers() & Qt.MetaModifier):
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() != Qt.LeftButton:
            return
        x, y = self._p(e)
        t = self.tool
        if t == "select":
            s = self._pick(x, y)
            add = e.modifiers() & Qt.ShiftModifier
            if s is not None:
                if add:
                    self.selected ^= {s.id}
                elif s.id not in self.selected:
                    self.selected = {s.id}
                self._moving = True
                self.push_undo()
                self._start = (x, y)
                self.redraw()
            else:
                if not add:
                    self.selected.clear()
                    self.redraw()
                self._start = (x, y)
                self._moving = False
        elif t in DRAG_TOOLS:
            self._start = (x, y)
        elif t == "ff":
            self._start = (x, y); self._pts = [(x, y)]
        elif t == "pt":
            self.add_shapes([s2.Shape("pt", dict(x=x, y=y))])
        elif t == "txt":
            text, ok = QInputDialog.getText(self, "Texto", "Texto:")
            if ok and text:
                size, ok2 = QInputDialog.getDouble(self, "Texto", "Altura (mm):", 10, 0.1, 1000, 2)
                if ok2:
                    self.add_shapes([s2.Shape("txt", dict(x=x, y=y, size=size, text=text))])
        elif t in CLICK_TOOLS:
            self._pts.append((x, y))
            if t == "arc" and len(self._pts) == 3:
                (x1, y1), (x2, y2), (x3, y3) = self._pts
                self._cancel()
                self.add_shapes([s2.Shape("arc", dict(x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3))])
                return
            if t == "protractor" and len(self._pts) == 3:
                a, v, b = map(np.array, self._pts)
                u1, u2 = a - v, b - v
                ang = math.degrees(math.acos(np.clip(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2) + 1e-12), -1, 1)))
                self.message.emit(f"Ângulo: {ang:.3f}°  (vértice em {v[0]:.2f}, {v[1]:.2f})")
                self._cancel()

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        x, y = self._p(e)
        t = self.tool
        txt = f"x={x:.2f}  y={y:.2f}"
        if t == "select" and self._start is not None:
            dx, dy = x - self._start[0], y - self._start[1]
            if self._moving:
                for sid in self.selected:
                    self.items[sid].setPos(dx, dy)
                txt += f"   Δ=({dx:.2f}, {dy:.2f})"
            else:
                r = QPainterPath(); r.addRect(QRectF(QPointF(*self._start), QPointF(x, y)).normalized())
                self._set_preview(r)
        elif t in DRAG_TOOLS and self._start is not None:
            s = self._shape_from_drag(self._start, (x, y))
            if s is not None:
                self._set_preview(s.geom)
            dx, dy = x - self._start[0], y - self._start[1]
            txt += f"   Δ=({dx:.2f}, {dy:.2f})  dist={math.hypot(dx, dy):.3f}"
        elif t == "ff" and self._start is not None:
            self._pts.append(self._p_raw(e))
            path = QPainterPath(); path.moveTo(*self._pts[0])
            for q in self._pts[1:]:
                path.lineTo(*q)
            self._set_preview(path)
        elif t in CLICK_TOOLS and self._pts:
            pts = self._pts + [(x, y)]
            path = QPainterPath(); path.moveTo(*pts[0])
            drawn = s2.catmull_rom(pts) if t == "spl" and len(pts) > 2 else pts
            for q in drawn[1:]:
                path.lineTo(float(q[0]), float(q[1]))
            self._set_preview(path)
        self.message.emit(txt)

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.setCursor(Qt.ArrowCursor if self.tool == "select" else Qt.CrossCursor)
            return
        if e.button() != Qt.LeftButton or self._start is None:
            return
        x, y = self._p(e)
        t = self.tool
        if t == "select":
            if self._moving:
                dx, dy = x - self._start[0], y - self._start[1]
                if dx or dy:
                    new, sel = [], set()
                    for sh in self.shapes:
                        if sh.id in self.selected:
                            sh = s2.move(sh, dx, dy); sel.add(sh.id)
                        new.append(sh)
                    self.shapes, self.selected = new, sel
                else:
                    self.undo_stack.pop()
                self.redraw()
            else:
                r = sbox(min(self._start[0], x), min(self._start[1], y), max(self._start[0], x), max(self._start[1], y))
                if r.area > 0:
                    self.selected |= {s.id for s in self.shapes if r.intersects(s.geom)}
                self.redraw()
            self._moving = False
            self._cancel()
            return
        if t == "ff":
            pts = self._pts
            self._cancel()
            if len(pts) > 2:
                closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 5 / self.transform().m11()
                if closed:
                    self.add_shapes([s2.Shape("pol", {"pts": [tuple(q) for q in np.array(s2.LineString(pts).simplify(0.2).coords)]})])
                else:
                    self.add_shapes([s2.Shape("ff", {"pts": pts})])
            return
        if t in DRAG_TOOLS:
            s = self._shape_from_drag(self._start, (x, y), True)
            if t == "ruler":
                dx, dy = x - self._start[0], y - self._start[1]
                k, u = self.unit_k, self.unit
                self.message.emit(f"Distância: {math.hypot(dx, dy) * k:.4f} {u}   Δx={dx * k:.4f}  Δy={dy * k:.4f}  "
                                  f"ângulo={math.degrees(math.atan2(dy, dx)):.2f}°")
            elif s is not None:
                if t == "rcr":
                    ang, ok = QInputDialog.getDouble(self, "Retângulo rodado", "Ângulo (graus):", 30, -360, 360, 2)
                    p = s.params
                    s = s2.Shape("rcr", dict(cx=(p["left"] + p["right"]) / 2, cy=(p["bottom"] + p["top"]) / 2,
                                             w=p["right"] - p["left"], h=p["top"] - p["bottom"], angle=ang if ok else 0))
                self.add_shapes([s])
            self._cancel()

    def mouseDoubleClickEvent(self, e):
        if self.tool in ("pol", "pl", "spl"):
            self._finish_click_shape()
        elif self.tool == "select":
            s = self._pick(*self._p(e))
            if s is not None:
                info = s2.shape_info(s)
                self.message.emit("   ".join(f"{k}: {v}" for k, v in info.items()))

    def _finish_click_shape(self):
        pts = list(dict.fromkeys(self._pts))
        t = self.tool
        self._cancel()
        if t == "pol" and len(pts) >= 3:
            self.add_shapes([s2.Shape("pol", {"pts": pts})])
        elif t == "pl" and len(pts) >= 2:
            self.add_shapes([s2.Shape("pl", {"pts": pts})])
        elif t == "spl" and len(pts) >= 3:
            closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 2
            self.add_shapes([s2.Shape("spl", {"pts": pts[:-1] if closed else pts, "closed": closed})])

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish_click_shape()
        elif e.key() == Qt.Key_Escape:
            self._cancel()
        elif e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(e)

    def wheelEvent(self, e):
        k = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.scale(k, k)
