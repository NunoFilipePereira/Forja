"""Visualizador 3D (OpenGL via pyqtgraph)."""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import trimesh
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QVector3D, QColor
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader
from .viewimage import ColorBar

# Iluminação: luz principal + luz de frente (acompanha a câmara) + ambiente, nas duas faces.
LIT = ShaderProgram("forja_lit", [
    VertexShader("""
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform float u_light[6];   // ambiente, difusa, especular, direção x y z (espaço da câmara)
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            vec3 n = normalize(v_normal);
            vec3 key = normalize(vec3(u_light[3], u_light[4], u_light[5]));
            float d = max(dot(n, key), 0.0) * u_light[1] + abs(n.z) * 0.35 * u_light[1] / 0.55;
            float spec = pow(max(dot(reflect(-key, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0) * u_light[2];
            vec3 rgb = v_color.rgb * (u_light[0] + d) + vec3(spec);
            gl_FragColor = vec4(min(rgb, vec3(1.0)), v_color.a);
        }
    """),
], uniforms={"u_light": [0.30, 0.55, 0.18, 0.45, 0.55, 0.70]})


class LabelOverlay:
    pass


def _make_overlay(view):
    from PySide6.QtWidgets import QWidget
    from PySide6.QtGui import QPainter, QMatrix4x4
    from PySide6.QtCore import QRectF

    class _Overlay(QWidget):
        """Etiquetas de texto (números e títulos dos eixos) desenhadas por cima da vista 3D."""
        def __init__(self, v):
            super().__init__(v)
            self.v = v
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(Qt.WA_NoSystemBackground, True)

        def paintEvent(self, e):
            v = self.v
            labels = list(getattr(v, "_labels", [])) + list(getattr(v, "_extra_labels", []))
            if not labels:
                return
            w, h = self.width(), self.height()
            vp = QMatrix4x4(); vp.viewport(0, h, w, -h)
            m = vp * v.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * v.viewMatrix()
            cam = v.cameraPosition()
            p = QPainter(self); p.setRenderHint(QPainter.TextAntialiasing)
            for pos, text, color, font in labels:
                q = QVector3D(float(pos[0]), float(pos[1]), float(pos[2]))
                pt = m.map(q)
                if not (-50 < pt.x() < w + 50 and -50 < pt.y() < h + 50):
                    continue
                p.setFont(font); p.setPen(QColor(*color))
                fm = p.fontMetrics(); tw = fm.horizontalAdvance(text)
                p.drawText(int(pt.x() - tw / 2), int(pt.y() + fm.ascent() / 2), text)
            p.end()
    return _Overlay(view)


class View3D(gl.GLViewWidget):
    picked = Signal(int, bool)  # id do objeto (0 = vazio), acrescentar à seleção
    measured = Signal(str)
    MODES = ["Sólido", "Sólido + arestas", "Arame", "Pontos"]

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.mesh_items = {}
        self.selected_ids = set()
        self.mode = "Sólido"
        self.show_overhang = False
        self.bg = QColor(38, 41, 48)
        self.setBackgroundColor(self.bg)
        self.grid = gl.GLGridItem()
        self.grid.setSize(200, 200); self.grid.setSpacing(10, 10)
        self.grid.setColor((255, 255, 255, 50))
        self.addItem(self.grid)
        self.axis = gl.GLAxisItem(); self.axis.setSize(30, 30, 30)
        self.addItem(self.axis)
        # eixos em caixa (paredes do fundo com grelha tracejada, números e títulos)
        self.axes_box = True
        self._box_items, self._box_key = [], None
        self.grid.setVisible(False); self.axis.setVisible(False)
        self._labels = []
        self._overlay = _make_overlay(self)
        self.setCameraPosition(distance=250, elevation=28, azimuth=-60)
        self.ortho = False
        self.clip = None  # (origem, normal): mostra só o lado do plano para onde a normal aponta ao contrário
        self.point_size = 2.0
        self.measure_mode = False
        self.measure_pts = []
        self.measure_items = []
        self.legend = ColorBar(); self.legend.setParent(self); self.legend.setVisible(False)
        self.legend.setStyleSheet("background: transparent")
        doc.listeners.append(self.refresh)

    # --- aparência ------------------------------------------------------
    def set_lighting(self, ambient=0.30, diffuse=0.55, specular=0.18, azimuth=40.0, elevation=45.0):
        a, e = np.radians(azimuth), np.radians(elevation)
        LIT["u_light"] = [ambient, diffuse, specular, np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)]
        self.update()

    def set_ortho(self, on):
        if on == self.ortho:
            return
        f0 = self.opts["fov"]
        f1 = 1.0 if on else 60.0
        self.opts["distance"] *= np.tan(np.radians(f0) / 2) / np.tan(np.radians(f1) / 2)
        self.opts["fov"] = f1
        self.ortho = on
        self.update()

    def show_legend(self, cmap, lo, hi):
        self.legend.set(cmap, lo, hi)
        self._place_legend()

    def hide_legend(self):
        self.legend.set(None, 0, 0)

    def _place_legend(self):
        lw, lh = getattr(self, "legend_size", None) or (0, 0)
        if lw and lh and lh > lw:  # barra vertical à direita (largura × altura em píxeis)
            self.legend.setGeometry(self.width() - lw - 70, max(10, (self.height() - lh) // 2), lw + 60, lh)
        else:
            self.legend.setGeometry(20, self.height() - 40, min(lw or 500, self.width() - 40), max(lh or 26, 10))
        self.legend.raise_()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._place_legend()

    # --- pré-visualização do resultado de uma ferramenta ------------------
    def clear_preview_meshes(self):
        for it in getattr(self, "prev_items", []):
            self.removeItem(it)
        self.prev_items = []
        for oid in getattr(self, "prev_hidden", []):
            if oid in self.mesh_items:
                self.mesh_items[oid].setVisible(True)
        self.prev_hidden = []
        self.update()

    def show_preview_lines(self, lines):
        cols = {"parede exterior": (1.0, 0.55, 0.2, 1), "parede": (1.0, 0.8, 0.3, 1), "enchimento": (0.4, 0.75, 1.0, 0.8),
                "sólido": (0.9, 0.35, 0.35, 0.9), "suporte": (0.6, 0.9, 0.5, 0.7), "saia": (0.8, 0.8, 0.8, 1), "aba": (0.8, 0.8, 0.8, 1),
                "jangada": (0.7, 0.7, 0.7, 1)}
        groups = {}
        for kind, P in lines:
            groups.setdefault(kind, []).append(P)
        for kind, Ps in groups.items():
            segs = []
            for P in Ps:
                segs.append(np.repeat(P, 2, axis=0)[1:-1])
            pos = np.vstack(segs)
            it = gl.GLLinePlotItem(pos=pos, color=cols.get(kind, (1, 1, 1, 1)), width=1, mode="lines", antialias=False)
            self.addItem(it); self.prev_items.append(it)
        self.update()

    def show_preview_meshes(self, meshes, hide=(), colors=None):
        self.clear_preview_meshes()
        cols = [(0.95, 0.75, 0.3), (0.45, 0.75, 0.95), (0.6, 0.85, 0.45), (0.9, 0.5, 0.55), (0.75, 0.6, 0.9)]
        for k, m in enumerate(meshes):
            if not len(getattr(m, "faces", [])):
                continue
            c = tuple(colors[k][:3]) if colors and k < len(colors) and colors[k] is not None else cols[k % len(cols)]
            md = gl.MeshData(vertexes=np.asarray(m.vertices, np.float32), faces=np.asarray(m.faces, np.uint32))
            it = gl.GLMeshItem(meshdata=md, color=c + (1.0,), smooth=False, shader=LIT, glOptions="opaque")
            self.addItem(it); self.prev_items.append(it)
        self.prev_hidden = list(hide)
        for oid in self.prev_hidden:
            if oid in self.mesh_items:
                self.mesh_items[oid].setVisible(False)
        self.update()

    # --- pré-visualização do corte ----------------------------------------
    def clear_cut_preview(self):
        for it in getattr(self, "cut_items", []):
            self.removeItem(it)
        self.cut_items = []
        self.update()

    def show_cut_preview(self, mesh, planes, active=-1, cone=None):
        self.clear_cut_preview()
        lo, hi = mesh.bounds
        c = (lo + hi) / 2
        R = float(np.linalg.norm(hi - lo)) * 0.65
        for k, (o, n) in enumerate(planes):
            n = np.asarray(n, float); n /= np.linalg.norm(n)
            p0 = c - n * ((c - o) @ n)
            a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(n, a); u /= np.linalg.norm(u); v = np.cross(n, u)
            # o plano só um pouco maior do que o modelo (projeção da caixa envolvente no plano)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - p0
            ru = float(np.abs(corners @ u).max()) * 1.12; rv = float(np.abs(corners @ v).max()) * 1.12
            V = np.array([p0 + (su * ru * u + sv * rv * v) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            col = (0.78, 0.0, 0.78, 0.80) if k == active else (0.45, 0.25, 0.95, 0.45)
            it = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=V, faces=np.array([[0, 1, 2], [0, 2, 3]])), color=col, smooth=False,
                               drawEdges=True, edgeColor=col[:3] + (0.9,), glOptions="translucent")
            it.setDepthValue(100)  # desenhado depois das peças: não fica escondido quando as peças são recriadas
            self.addItem(it); self.cut_items.append(it)
        if cone is not None:
            h = (hi - lo)
            cen = lo + np.array([cone["cx"], cone["cy"], cone["cz"]]) * h
            d = np.array([cone["dx"], cone["dy"], cone["dz"]], float)
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([0, 0, 1.0])
            a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(d, a); u /= np.linalg.norm(u); v = np.cross(d, u)
            t = np.linspace(0, 2 * np.pi, int(cone["sides"]) + 1)
            b0 = cen - d * cone["cone_h"] / 2; b1 = cen + d * cone["cone_h"] / 2
            r0, r1 = cone["d1"] / 2, cone["d2"] / 2
            ring0 = np.array([b0 + r0 * (np.cos(x) * u + np.sin(x) * v) for x in t]); ring1 = np.array([b1 + r1 * (np.cos(x) * u + np.sin(x) * v) for x in t])
            for pts in (ring0, ring1):
                it = gl.GLLinePlotItem(pos=pts, color=(1, 0.85, 0.1, 1), width=2, antialias=True); it.setDepthValue(100); self.addItem(it); self.cut_items.append(it)
            for k in range(0, len(t), max(1, len(t) // 8)):
                it = gl.GLLinePlotItem(pos=np.array([ring0[k], ring1[k]]), color=(1, 0.85, 0.1, 1), width=1); it.setDepthValue(100); self.addItem(it); self.cut_items.append(it)
        self.update()

    # --- régua 3D ---------------------------------------------------------
    def clear_measure(self):
        for it in self.measure_items:
            self.removeItem(it)
        self.measure_items = []
        self.measure_pts = []

    def _measure_click(self, x, y):
        hit = self.pick_point(x, y)
        if hit is None:
            return
        p = hit[1]
        if len(self.measure_pts) >= 2:
            self.clear_measure()
        self.measure_pts.append(p)
        dot = gl.GLScatterPlotItem(pos=np.array([p]), size=10, color=(1, 0.9, 0.1, 1), pxMode=True)
        dot.setGLOptions("additive"); self.addItem(dot); self.measure_items.append(dot)
        if len(self.measure_pts) == 2:
            a, b = self.measure_pts
            ln = gl.GLLinePlotItem(pos=np.array([a, b]), color=(1, 0.9, 0.1, 1), width=2, antialias=True)
            ln.setGLOptions("additive"); self.addItem(ln); self.measure_items.append(ln)
            d = b - a
            self.measured.emit(f"Distância 3D: {np.linalg.norm(d):.3f} mm   Δ = ({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})   "
                               f"de ({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}) a ({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f})")
        else:
            self.measured.emit(f"Ponto 1: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) — clique no segundo ponto")

    # ------------------------------------------------------------------
    def set_background(self, c: QColor):
        self.bg = c
        self.setBackgroundColor(c)

    def set_grid(self, size, visible=True):
        self.grid.setSize(size, size)
        self.grid.setVisible(visible)

    def _key(self, o):
        """Tudo o que muda o aspeto de um objeto; se não mudou, o item OpenGL é reaproveitado."""
        m = o.mesh
        vc = None if o.vcolors is None else (o.vcolors[0], len(o.vcolors[1]))
        clip = None if self.clip is None else (tuple(np.round(self.clip[0], 6)), tuple(np.round(self.clip[1], 6)))
        return (hash(m.vertices), hash(m.faces), tuple(o.color), o.alpha, o.id in self.selected_ids, self.mode,
                self.show_overhang, vc, clip, self.point_size, repr(getattr(o, "style", None)))

    def refresh(self):
        cache = getattr(self, "_cache", {})
        new_items, new_cache = {}, {}
        alive = set()
        for o in self.doc.objects:
            if not o.visible or not len(o.mesh.vertices):
                continue
            k = self._key(o)
            if o.id in self.mesh_items and cache.get(o.id) == k:
                new_items[o.id] = self.mesh_items[o.id]
            else:
                if o.id in self.mesh_items:
                    self._remove_item(self.mesh_items[o.id])
                new_items[o.id] = self._make_item(o)
                self.addItem(new_items[o.id])
                for c in getattr(new_items[o.id], "_pending", []):
                    c.setParentItem(new_items[o.id])
            new_cache[o.id] = k
            alive.add(o.id)
        for oid, it in self.mesh_items.items():
            if oid not in alive:
                self._remove_item(it)
        self.mesh_items, self._cache = new_items, new_cache
        self._box_key = None
        self.update()

    # ------------------------------------------------------------------ eixos em caixa
    @staticmethod
    def _nice_ticks(a, b, n=5):
        span = max(b - a, 1e-9)
        raw = span / n
        mag = 10 ** np.floor(np.log10(raw))
        step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
        t0, t1 = np.floor(a / step) * step, np.ceil(b / step) * step
        if t1 - t0 < 1e-12:
            t1 = t0 + step
        return np.arange(t0, t1 + step * 0.5, step), step

    def set_axes_box(self, on):
        self.axes_box = bool(on)
        self.grid.setVisible(not on)
        self._box_key = None
        if not on:
            self._clear_box()
        self.update()

    def _clear_box(self):
        self._labels = []
        for it in self._box_items:
            if it in self.items:
                self.removeItem(it)
        self._box_items = []

    def _camera_dir(self):
        el, az = np.radians(self.opts["elevation"]), np.radians(self.opts["azimuth"])
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def _update_box(self):
        from PySide6.QtGui import QFont
        from ..core.colormaps import dashed_segments
        b = self.doc.bounds()
        cam = self._camera_dir()
        key = (tuple(np.round(b.ravel(), 4)), tuple(np.sign(np.round(cam, 3))), getattr(self, "text3d_size", 11),
               self.bg.lightness() if hasattr(self, "bg") else 0)
        if key == self._box_key:
            return
        self._box_key = key
        self._clear_box()
        ticks = [self._nice_ticks(b[0][i], b[1][i]) for i in range(3)]
        lo = np.array([t[0][0] for t in ticks]); hi = np.array([t[0][-1] for t in ticks])
        # paredes do fundo: do lado oposto à câmara
        back = np.where(cam > 0, lo, hi)
        front = np.where(cam > 0, hi, lo)
        dark = not (hasattr(self, "bg") and self.bg.lightness() > 140)
        cg = (0.85, 0.85, 0.85, 0.35) if dark else (0.25, 0.25, 0.25, 0.35)
        ce = (0.9, 0.9, 0.9, 0.8) if dark else (0.15, 0.15, 0.15, 0.8)
        tc = (230, 230, 230, 255) if dark else (30, 30, 30, 255)
        ref = float(np.max(hi - lo)) / 120
        A, B = [], []
        # grelha nas 3 paredes do fundo (x = back[0], y = back[1], z = back[2])
        for wall in range(3):
            o = [i for i in range(3) if i != wall]
            for k in o:
                other = [i for i in o if i != k][0]
                for t in ticks[k][0]:
                    p0 = np.zeros(3); p1 = np.zeros(3)
                    p0[wall] = p1[wall] = back[wall]
                    p0[k] = p1[k] = t
                    p0[other], p1[other] = lo[other], hi[other]
                    A.append(p0); B.append(p1)
        G = dashed_segments(np.asarray(A), np.asarray(B), "Dotted line", ref)
        it = gl.GLLinePlotItem(pos=G, color=cg, width=1, mode="lines", antialias=True)
        self.addItem(it); self._box_items.append(it)
        # arestas da caixa nas paredes do fundo
        E = []
        for wall in range(3):
            o = [i for i in range(3) if i != wall]
            c = []
            for u in (lo[o[0]], hi[o[0]]):
                for v in (lo[o[1]], hi[o[1]]):
                    q = np.zeros(3); q[wall] = back[wall]; q[o[0]] = u; q[o[1]] = v; c.append(q)
            E += [c[0], c[1], c[1], c[3], c[3], c[2], c[2], c[0]]
        it = gl.GLLinePlotItem(pos=np.asarray(E), color=ce, width=1, mode="lines", antialias=True)
        self.addItem(it); self._box_items.append(it)
        # números e títulos nas arestas da frente, junto ao chão (X e Y) e na aresta vertical (Z)
        f = QFont("Helvetica", int(getattr(self, "text3d_size", 11)))
        ft = QFont("Helvetica", int(getattr(self, "text3d_size", 11) * 1.3)); ft.setBold(True)
        off = float(np.max(hi - lo)) * 0.04
        zf = back[2]
        for axis, edge_other, title in ((0, 1, "X"), (1, 0, "Y")):
            for t in ticks[axis][0]:
                p = np.zeros(3); p[axis] = t; p[edge_other] = front[edge_other] + np.sign(front[edge_other] - back[edge_other] or 1) * off; p[2] = zf
                self._labels.append((p.copy(), f"{t:g}", tc, f))
            p = np.zeros(3); p[axis] = (lo[axis] + hi[axis]) / 2
            p[edge_other] = front[edge_other] + np.sign(front[edge_other] - back[edge_other] or 1) * off * 3; p[2] = zf
            self._labels.append((p.copy(), title, tc, ft))
        for t in ticks[2][0]:
            p = np.array([back[0], front[1], t], float)
            p[1] += np.sign(front[1] - back[1] or 1) * off
            self._labels.append((p.copy(), f"{t:g}", tc, f))
        p = np.array([back[0], front[1] + np.sign(front[1] - back[1] or 1) * off * 3, (lo[2] + hi[2]) / 2])
        self._labels.append((p.copy(), "Z", tc, ft))

    def _box_needs_update(self):
        b = self.doc.bounds()
        cam = self._camera_dir()
        key = (tuple(np.round(b.ravel(), 4)), tuple(np.sign(np.round(cam, 3))), getattr(self, "text3d_size", 11),
               self.bg.lightness() if hasattr(self, "bg") else 0)
        return key != self._box_key

    def _rebuild_box_later(self):
        if getattr(self, "_box_pending", False):
            return
        self._box_pending = True
        from PySide6.QtCore import QTimer

        def go():
            self._box_pending = False
            try:
                self._update_box()
            except Exception:
                pass
            self.update()
        QTimer.singleShot(0, go)

    def paintGL(self, *a, **k):
        # a caixa de eixos nunca é criada durante o desenho OpenGL (dá erro): marca-se para depois
        if getattr(self, "axes_box", False):
            try:
                if self._box_needs_update():
                    self._rebuild_box_later()
            except Exception:
                pass
        r = super().paintGL(*a, **k)
        ov = getattr(self, "_overlay", None)
        if ov is not None:
            if ov.size() != self.size():
                ov.resize(self.size())
            ov.raise_(); ov.update()
        return r

    def _remove_item(self, it):
        for c in list(it.childItems()):
            if c in self.items:
                self.removeItem(c)
        if it in self.items:
            self.removeItem(it)

    def _make_item(self, o):
        m = o.mesh
        if self.clip is not None and len(m.faces):
            org, nrm = self.clip
            try:
                cm = trimesh.intersections.slice_mesh_plane(m, -np.asarray(nrm, float), org, cap=len(m.faces) < 300000)
            except Exception:
                cm = trimesh.intersections.slice_mesh_plane(m, -np.asarray(nrm, float), org, cap=False)
            if not len(cm.faces):
                return gl.GLScatterPlotItem(pos=np.zeros((0, 3)))
            m = cm
        if getattr(o, "style", None):
            return self._make_styled(o, m)
        sel = o.id in self.selected_ids
        r, g, b = o.color
        if sel:
            r, g, b = min(1, r * 1.25 + 0.15), min(1, g * 1.25 + 0.15), min(1, b * 1.25 + 0.15)
        a = o.alpha
        if self.mode == "Pontos" or not len(m.faces):
            return gl.GLScatterPlotItem(pos=np.asarray(m.vertices), size=self.point_size, color=(r, g, b, a), pxMode=True)
        md = gl.MeshData(vertexes=np.asarray(m.vertices, np.float32), faces=np.asarray(m.faces, np.uint32))
        if o.vcolors is not None and o.vcolors[0] == id(m) and len(o.vcolors[1]) == len(m.vertices):
            md.setVertexColors(o.vcolors[1])
        elif self.show_overhang:
            from ..core.meshops import overhang_faces
            fc = np.tile([r, g, b, a], (len(m.faces), 1)).astype(np.float32)
            fc[overhang_faces(m)] = [0.9, 0.15, 0.15, a]
            md.setFaceColors(fc)
        wire = self.mode == "Arame"
        item = gl.GLMeshItem(meshdata=md, color=(r, g, b, a), smooth=False,
                             shader=None if wire else LIT,
                             drawFaces=not wire,
                             drawEdges=wire or self.mode == "Sólido + arestas" or sel and len(m.faces) < 20000,
                             edgeColor=(1, 1, 0.4, 0.6) if sel else (0.1, 0.1, 0.1, 0.5),
                             glOptions="translucent" if a < 1 else "opaque")
        return item

    def _make_styled(self, o, m):
        """Superfície, arestas e pontos com escalas de cor e estilos."""
        from ..core import colormaps as CM
        st = o.style
        sel = o.id in self.selected_ids
        a = o.alpha
        V = np.asarray(m.vertices, float)
        z = V[:, 2]
        t = (z - z.min()) / max(np.ptp(z), 1e-9)
        root = gl.GLGraphicsItem.GLGraphicsItem()
        root._pending = []

        def cols(part, idx=None):
            sch = st.get(part, {}).get("scheme", "True color")
            if sch == "True color" and o.vcolors is not None and o.vcolors[0] == id(o.mesh) and len(o.vcolors[1]) == len(V):
                c = np.asarray(o.vcolors[1], np.float32).copy(); c[:, 3] = a
            else:
                c = CM.colors(t, sch, o.color, a)
            return c if idx is None else c[idx]
        sf = st.get("surface", {})
        if sf.get("show", True) and len(m.faces) and self.mode != "Pontos":
            md = gl.MeshData(vertexes=V.astype(np.float32), faces=np.asarray(m.faces, np.uint32))
            md.setVertexColors(cols("surface"))
            it = gl.GLMeshItem(meshdata=md, smooth=False, shader=LIT, drawEdges=False,
                               glOptions="translucent" if a < 1 else "opaque")
            root._pending.append(it)
        ed = st.get("edge", {})
        if (ed.get("show", False) or sel) and len(m.faces):
            E = m.edges_unique
            if len(E) > 300000:
                E = E[:: int(np.ceil(len(E) / 300000))]
            ref = max(float(np.max(m.extents)) / 60, 0.2)
            dash = ed.get("dash", "Solid line")
            if dash == "Solid line":
                P = V[E].reshape(-1, 3); C = cols("edge", E.reshape(-1))
            else:
                P = CM.dashed_segments(V[E[:, 0]], V[E[:, 1]], dash, ref)
                tz = (P[:, 2] - z.min()) / max(np.ptp(z), 1e-9)
                C = CM.colors(tz, ed.get("scheme", "True color") if ed.get("scheme") != "True color" else "Solid color",
                              (0.1, 0.1, 0.1) if ed.get("scheme", "True color") == "True color" else o.color, a)
            if sel and not ed.get("show", False):
                C = np.tile([1, 1, 0.4, 0.6], (len(P), 1)).astype(np.float32)
            li = gl.GLLinePlotItem(pos=P, color=C, width=float(ed.get("width", 1.0)), mode="lines", antialias=True)
            root._pending.append(li)
        pt = st.get("point", {})
        if pt.get("show", False) or self.mode == "Pontos" or not len(m.faces):
            step = max(int(st.get("sampling", 1)), 1)
            idx = np.arange(0, len(V), step)
            mk = pt.get("marker", "Dot")
            size = float(pt.get("size", self.point_size))
            if mk == "Dot":
                sp = gl.GLScatterPlotItem(pos=V[idx], size=size, color=cols("point", idx), pxMode=True)
                root._pending.append(sp)
            else:
                segs = CM.marker_segments(V[idx], mk, size * max(float(np.max(m.extents)) / 400, 0.05))
                k = len(segs) // max(len(idx), 1)
                C = np.repeat(cols("point", idx), k, axis=0)
                li = gl.GLLinePlotItem(pos=segs, color=C, width=1.0, mode="lines", antialias=True)
                root._pending.append(li)
        return root

    # --- seleção por clique -----------------------------------------------
    # modos do rato: arrastar objetos, deslocar, rodar, mover a luz
    drag_mode = "Rotate"
    object_dragged = Signal(int, object)   # id do objeto, deslocamento (dx, dy, dz) em mm
    light_moved = Signal(float, float)     # azimute, elevação

    def mousePressEvent(self, ev):
        self._press = ev.position()
        self._last = ev.position()
        self._float_id = None
        if self.drag_mode == "Float" and ev.button() == Qt.LeftButton:
            self._float_id = self.pick(self._press.x(), self._press.y())
        super().mousePressEvent(ev) if self.drag_mode == "Rotate" or ev.button() != Qt.LeftButton else None

    def mouseMoveEvent(self, ev):
        if self.drag_mode == "Rotate" or not (ev.buttons() & Qt.LeftButton):
            return super().mouseMoveEvent(ev)
        p = ev.position(); d = p - self._last; self._last = p
        if self.drag_mode == "Shift":
            # desloca a vista (como arrastar com o botão do meio)
            self.pan(d.x(), d.y(), 0, relative="view")
        elif self.drag_mode == "Float" and self._float_id:
            # arrasta o objeto no plano da vista
            dist = self.opts["distance"]
            k = dist * np.tan(np.radians(self.opts["fov"]) / 2) * 2 / max(self.height(), 1)
            az, el = np.radians(self.opts["azimuth"]), np.radians(self.opts["elevation"])
            right = np.array([-np.sin(az), np.cos(az), 0.0])
            up = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
            self.object_dragged.emit(int(self._float_id), right * d.x() * k - up * d.y() * k)
        elif self.drag_mode == "Move light":
            from .view3d import LIT as _L
            u = list(_L.uniformData.get("u_light", [0.3, 0.55, 0.18, 0.4, 0.7, 0.6]))
            az = np.degrees(np.arctan2(u[3], u[5])) + d.x() * 0.5
            el = np.clip(np.degrees(np.arcsin(np.clip(u[4], -1, 1))) - d.y() * 0.5, -89, 89)
            self.set_lighting(u[0], u[1], u[2], az, el)
            self.light_moved.emit(float(az), float(el))

    def zoom(self, k):
        self.opts["distance"] *= k
        self.update()

    def viewport_shift(self, side, sign):
        """Viewport: alarga/encolhe a margem de um lado — desloca o conteúdo da vista."""
        step = 0.05 * self.width() * sign
        dx = {"Left": step, "Right": -step}.get(side, 0)
        dy = {"Top": step, "Bottom": -step}.get(side, 0)
        self.pan(dx, dy, 0, relative="view")

    def mouseReleaseEvent(self, ev):
        p = ev.position()
        if ev.button() == Qt.LeftButton and hasattr(self, "_press") and (p - self._press).manhattanLength() < 4 \
                and self.measure_mode:
            self._measure_click(p.x(), p.y())
        elif ev.button() == Qt.LeftButton and hasattr(self, "_press") and (p - self._press).manhattanLength() < 4:
            oid = self.pick(p.x(), p.y())
            self.picked.emit(oid or 0, bool(ev.modifiers() & (Qt.ShiftModifier | Qt.MetaModifier)))
        super().mouseReleaseEvent(ev)

    def ray(self, x, y):
        w, h = self.width(), self.height()
        m = self.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * self.viewMatrix()
        inv, ok = m.inverted()
        nx, ny = 2 * x / w - 1, 1 - 2 * y / h
        a = inv.map(QVector3D(nx, ny, -1)); b = inv.map(QVector3D(nx, ny, 1))
        o = np.array([a.x(), a.y(), a.z()]); d = np.array([b.x(), b.y(), b.z()]) - o
        return o, d / np.linalg.norm(d)

    def pick(self, x, y):
        hit = self.pick_point(x, y)
        return hit[0] if hit else None

    def pick_point(self, x, y):
        """(id, ponto) do objeto visível mais próximo sob o cursor (interseção raio–malha)."""
        o, d = self.ray(x, y)
        best = None
        for obj in self.doc.objects:
            if not obj.visible or not len(obj.mesh.faces):
                continue
            lo, hi = obj.mesh.bounds
            # teste rápido com a caixa envolvente
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tmin = np.nanmax(np.minimum(t1, t2)); tmax = np.nanmin(np.maximum(t1, t2))
            if tmax < max(tmin, 0):
                continue
            locs, _, _ = obj.mesh.ray.intersects_location([o], [d], multiple_hits=False)
            if len(locs):
                k = np.argmin(np.linalg.norm(locs - o, axis=1))
                dist = np.linalg.norm(locs[k] - o)
                if best is None or dist < best[0]:
                    best = (dist, obj.id, locs[k])
        return (best[1], best[2]) if best else None

    def set_selection(self, ids):
        self.selected_ids = set(ids)
        self.refresh()

    # --- câmara -------------------------------------------------------
    def fit(self):
        b = self.doc.bounds()
        c = b.mean(axis=0)
        size = float(np.linalg.norm(b[1] - b[0])) or 100
        self.opts["center"] = QVector3D(*c)
        self.setCameraPosition(distance=size * 1.6 * (np.tan(np.radians(30)) / np.tan(np.radians(self.opts["fov"]) / 2)))
        gsize = max(100, np.ceil(max(b[1][:2] - b[0][:2]) * 1.4 / 50) * 50)
        self.grid.setSize(gsize, gsize)
        self.update()

    def view(self, name):
        presets = {"Cima": (90, -90), "Baixo": (-90, -90), "Frente": (0, -90), "Trás": (0, 90),
                   "Esquerda": (0, 180), "Direita": (0, 0), "Isométrica": (30, -45)}
        el, az = presets[name]
        self.setCameraPosition(elevation=el, azimuth=az)
        self.fit()

    def snapshot(self, path):
        img = self.grabFramebuffer()
        img.save(path)
