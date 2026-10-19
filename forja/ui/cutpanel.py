"""Painel Cortar: Input / Cutting Plane / Connector, com pré-visualização dos planos."""
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel, QScrollArea, QDoubleSpinBox, QSpinBox,
                               QCheckBox, QComboBox, QPushButton, QHBoxLayout, QSlider, QRadioButton, QButtonGroup)
from ..core import cutting as C
from ..i18n import tr as _tr

PT = {"Planar cut": "Corte planar", "Conic cut": "Corte cónico", "Flexi cut": "Corte flexível", "Grid cut": "Corte em grelha",
      "Radial cut": "Corte radial", "Modular cut": "Corte modular", "Plug": "Macho (plug)", "Dowel": "Cavilha", "None": "Nenhum",
      "Prism": "Prisma", "Frustum": "Tronco", "Triangle": "Triângulo", "Square": "Quadrado", "Pentagon": "Pentágono",
      "Hexagon": "Hexágono", "Octagon": "Octógono", "Circle": "Círculo", "D-shape": "Forma D", "Lips": "Lábios", "Native": "Nativa (secção)",
      "Any": "Automático"}


def tr(t):
    return _tr(PT.get(t, t))


def _dspin(v, lo, hi, step=0.1, dec=3):
    w = QDoubleSpinBox(); w.setDecimals(dec); w.setRange(lo, hi); w.setSingleStep(step); w.setValue(v); w.setKeyboardTracking(False)
    return w


def _radio(options, current):
    box = QWidget(); lay = QHBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0)
    grp = QButtonGroup(box)
    for k, o in enumerate(options):
        b = QRadioButton(tr(o)); b.setProperty("val", o); grp.addButton(b, k); lay.addWidget(b)
        b.setChecked(o == current)
    box.group = grp
    return box


def _radio_val(box):
    b = box.group.checkedButton()
    return b.property("val") if b else None


class SliderSpin(QWidget):
    """Cursor + caixa numérica ligados."""
    def __init__(self, lo, hi, v, cb):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.spin = _dspin(v, lo, hi, 0.05, 3); self.sl = QSlider(Qt.Horizontal); self.sl.setRange(0, 1000)
        self.lo, self.hi = lo, hi
        lay.addWidget(self.spin); lay.addWidget(self.sl, 1)
        self._set_slider(v)
        self.sl.valueChanged.connect(lambda k: self.spin.setValue(lo + (hi - lo) * k / 1000))
        self.spin.valueChanged.connect(lambda x: (self._set_slider(x), cb()))

    def _set_slider(self, v):
        self.sl.blockSignals(True); self.sl.setValue(int((v - self.lo) / (self.hi - self.lo) * 1000)); self.sl.blockSignals(False)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        self.spin.setValue(v)


def _layout_extras(parts, extras, spacing=5.0):
    """Cavilhas e outras peças soltas: em fila à frente das peças, de pé, afastadas entre si e das peças
    (saem todas do corte no mesmo sítio e ficariam umas dentro das outras)."""
    if not extras:
        return
    lo = np.min([p.bounds[0] for p in parts], axis=0)
    hi = np.max([p.bounds[1] for p in parts], axis=0)
    dmax = max(float(max(e.extents[0], e.extents[1])) for e in extras)
    x = lo[0]
    y = lo[1] - spacing * 2 - dmax
    row_w = max(hi[0] - lo[0], dmax * 4)
    for e in extras:
        b = e.bounds
        w = b[1][0] - b[0][0]
        if x + w > lo[0] + row_w and x > lo[0]:
            x = lo[0]; y -= dmax + spacing
        e.apply_translation((x - b[0][0], y - b[0][1], -b[0][2]))
        x += w + spacing


class _Source:
    """O modelo original de um corte (fica guardado nas peças para se poder voltar a ajustar o corte)."""
    def __init__(self, mesh, name):
        self.mesh, self.name, self.id = mesh, name, None


_GROUPS = iter(range(1, 10 ** 9))


class CutPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.obj = None
        self.planes = [[0.0, 0.0, 1.0, 0.0]]  # A, B, C, D por plano
        self._loading = False
        outer = QVBoxLayout(self); outer.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel(tr("Cortar")); f = self.title.font(); f.setBold(True); self.title.setFont(f); outer.addWidget(self.title)
        sc = QScrollArea(); sc.setWidgetResizable(True); outer.addWidget(sc, 1)
        body = QWidget(); self.body = QVBoxLayout(body); sc.setWidget(body)

        # ---- Input ----
        g = QGroupBox(tr("Entrada")); fl = QFormLayout(g); self.body.addWidget(g)
        self.method = QComboBox()
        for m in C.METHODS:
            self.method.addItem(tr(m), m)
        self.unit = QComboBox()
        for u in C.UNITS:
            self.unit.addItem(u, u)
        self.unit.setCurrentText("mm")
        self.sx, self.sy, self.sz = (_dspin(100, 0.001, 1e6, 1.0) for _ in range(3))
        fl.addRow(tr("Método"), self.method); fl.addRow(tr("Unidade"), self.unit)
        fl.addRow(tr("Tamanho do modelo X"), self.sx); fl.addRow(tr("Tamanho do modelo Y"), self.sy); fl.addRow(tr("Tamanho do modelo Z"), self.sz)

        # ---- Planar / Flexi ----
        self.g_planar = QGroupBox(tr("Plano de corte")); fl = QFormLayout(self.g_planar); self.body.addWidget(self.g_planar)
        fl.addRow(QLabel(tr("Plano: Ax + By + Cz = D (coordenadas do modelo de −1 a 1)")))
        self.n_planes = QSpinBox(); self.n_planes.setRange(1, 20); self.adj = QSpinBox(); self.adj.setRange(1, 1)
        fl.addRow(tr("Número de planos"), self.n_planes); fl.addRow(tr("Ajustar plano"), self.adj)
        self.A = SliderSpin(-1, 1, 0, self.plane_changed); self.B = SliderSpin(-1, 1, 0, self.plane_changed)
        self.Cc = SliderSpin(-1, 1, 1, self.plane_changed); self.D = SliderSpin(-1, 1, 0, self.plane_changed)
        for lab, w in (("A", self.A), ("B", self.B), ("C", self.Cc), ("D", self.D)):
            fl.addRow(lab, w)
        quick = QHBoxLayout()
        for lab, abc in (("⟂ X", (1, 0, 0)), ("⟂ Y", (0, 1, 0)), ("⟂ Z", (0, 0, 1))):
            b = QPushButton(lab); b.clicked.connect(lambda _=False, abc=abc: self.set_normal(abc)); quick.addWidget(b)
        fl.addRow(tr("Orientação rápida"), quick)
        self.regional = _radio(["Sim", "Não"], "Não"); fl.addRow(tr("Corte regional"), self.regional)
        self.region_size = _dspin(30, 1, 1e5, 1.0); fl.addRow(tr("Tamanho da região (mm)"), self.region_size)

        # ---- Conic ----
        self.g_conic = QGroupBox(tr("Cone (C: centro, D: direção)")); fl = QFormLayout(self.g_conic); self.body.addWidget(self.g_conic)
        self.cx = SliderSpin(0, 1, 0.5, self.changed); self.cy = SliderSpin(0, 1, 0.5, self.changed); self.cz = SliderSpin(0, 1, 0.95, self.changed)
        self.dx = SliderSpin(-1, 1, 0, self.changed); self.dy = SliderSpin(-1, 1, 0, self.changed); self.dz = SliderSpin(-1, 1, 1, self.changed)
        for lab, w in (("CX", self.cx), ("CY", self.cy), ("CZ", self.cz), ("DX", self.dx), ("DY", self.dy), ("DZ", self.dz)):
            fl.addRow(lab, w)
        self.sides = QSpinBox(); self.sides.setRange(3, 360); self.sides.setValue(36)
        self.vrot = QSpinBox(); self.vrot.setRange(0, 360); self.vrot.setValue(9)
        self.d1, self.d2, self.cone_h = _dspin(5, 0.1, 1e5, 1.0), _dspin(10, 0.1, 1e5, 1.0), _dspin(10, 0.1, 1e5, 1.0)
        fl.addRow(tr("Número de lados"), self.sides); fl.addRow(tr("Rotação dos vértices"), self.vrot)
        fl.addRow(tr("Diâmetro 1 (mm)"), self.d1); fl.addRow(tr("Diâmetro 2 (mm)"), self.d2); fl.addRow(tr("Altura (mm)"), self.cone_h)

        # ---- Grid ----
        self.g_grid = QGroupBox(tr("Plano de corte (grelha)")); fl = QFormLayout(self.g_grid); self.body.addWidget(self.g_grid)
        row = QHBoxLayout(); self.gx, self.gy, self.gz = (QSpinBox() for _ in range(3))
        for w, v in ((self.gx, 2), (self.gy, 2), (self.gz, 1)):
            w.setRange(1, 50); w.setValue(v); row.addWidget(w)
        fl.addRow(tr("Secções em X, Y, Z"), row)
        self.staggered = QCheckBox(tr("Padrão de corte desfasado")); fl.addRow(self.staggered)
        self.adj_axis = QComboBox(); self.adj_axis.addItems(["X", "Y", "Z"])
        self.adj_index = QSpinBox(); self.adj_index.setRange(0, 49); self.adj_off = _dspin(0, -1e5, 1e5, 1.0)
        fl.addRow(tr("Ajustar plano"), self.adj_axis); fl.addRow(tr("N.º do plano"), self.adj_index); fl.addRow(tr("Deslocamento (mm)"), self.adj_off)

        # ---- Radial ----
        self.g_radial = QGroupBox(tr("Plano de corte (radial)")); fl = QFormLayout(self.g_radial); self.body.addWidget(self.g_radial)
        self.r_dir = _radio(["X", "Y", "Z"], "Z"); fl.addRow(tr("Direção"), self.r_dir)
        self.r_cuts = QSpinBox(); self.r_cuts.setRange(1, 36); self.r_cuts.setValue(2)
        self.r_start = _dspin(0, -360, 360, 5.0); self.r_cx = SliderSpin(-1, 1, 0, self.changed); self.r_cy = SliderSpin(-1, 1, 0, self.changed)
        fl.addRow(tr("Número de cortes"), self.r_cuts); fl.addRow(tr("Ângulo inicial (°)"), self.r_start)
        fl.addRow(tr("Centro X [−1, 1]"), self.r_cx); fl.addRow(tr("Centro Y [−1, 1]"), self.r_cy)

        # ---- Modular ----
        self.g_mod = QGroupBox(tr("Impressora 3D")); fl = QFormLayout(self.g_mod); self.body.addWidget(self.g_mod)
        self.multi = _radio(["Uma impressora", "Várias impressoras"], "Uma impressora"); fl.addRow(self.multi)
        self.bed = QComboBox(); self.bed.addItem(tr("Retângulo"), "Rectangle"); self.bed.addItem(tr("Círculo"), "Circle")
        x, y, z = win.printer() if hasattr(win, "printer") else (200, 150, 140)
        self.px, self.py, self.pz = _dspin(x, 1, 1e5, 1.0), _dspin(y, 1, 1e5, 1.0), _dspin(z, 1, 1e5, 1.0)
        self.border = _dspin(0, 0, 1e4, 0.5); self.axial = _radio(["Sim", "Não"], "Não")
        fl.addRow(tr("Forma da mesa"), self.bed); fl.addRow(tr("Impressora X (mm)"), self.px); fl.addRow(tr("Impressora Y (mm)"), self.py)
        fl.addRow(tr("Impressora Z (mm)"), self.pz); fl.addRow(tr("Folga da borda (mm)"), self.border); fl.addRow(tr("Corte axial"), self.axial)

        # ---- Comum ----
        self.g_common = QGroupBox(tr("Acabamento")); fl = QFormLayout(self.g_common); self.body.addWidget(self.g_common)
        self.chamfer = _dspin(0, 0, 100, 0.2); self.close_cut = _radio(["Sim", "Não"], "Sim"); self.part_num = _radio(["Sim", "Não"], "Não")
        self.gap = _dspin(10, 0, 1e4, 1.0)
        fl.addRow(tr("Chanfro do corte (mm)"), self.chamfer); fl.addRow(tr("Fechar o corte"), self.close_cut)
        fl.addRow(tr("Numerar peças"), self.part_num); fl.addRow(tr("Afastar peças depois (mm)"), self.gap)

        # ---- Connector ----
        self.g_con = QGroupBox(tr("Conector")); fl = QFormLayout(self.g_con); self.body.addWidget(self.g_con)
        self.con = _radio(["Plug", "Dowel", "None"], "Plug"); self.prof = _radio(["Prism", "Frustum"], "Prism")
        self.shape = QComboBox()
        for s in C.SHAPES:
            self.shape.addItem(tr(s), s)
        self.shape.setCurrentIndex(C.SHAPES.index("Octagon"))
        self.number = QComboBox()
        for n in C.NUMBERS:
            self.number.addItem(tr(n), n)
        self.depth_ratio, self.size, self.tol = _dspin(1.5, 0.1, 20, 0.1), _dspin(5, 0.5, 500, 0.5), _dspin(-0.2, -5, 5, 0.05)
        self.lap = _radio(["Sim", "Não"], "Não"); self.lap_depth = _dspin(5, 0.1, 1e4, 0.5)
        fl.addRow(tr("Tipo"), self.con); fl.addRow(tr("Perfil"), self.prof); fl.addRow(tr("Forma"), self.shape); fl.addRow(tr("Número"), self.number)
        fl.addRow(tr("Razão de profundidade"), self.depth_ratio); fl.addRow(tr("Tamanho (mm)"), self.size); fl.addRow(tr("Tolerância (mm)"), self.tol)
        fl.addRow(tr("Sobreposição se possível"), self.lap); fl.addRow(tr("Profundidade da sobreposição (mm)"), self.lap_depth)
        self.body.addStretch(1)

        self.info = QLabel(" "); self.info.setWordWrap(True); outer.addWidget(self.info)
        row = QHBoxLayout(); outer.addLayout(row)
        self.b_cut = QPushButton(tr("Cortar")); self.b_cut.setDefault(True); self.b_close = QPushButton(tr("Fechar"))
        row.addWidget(self.b_cut); row.addWidget(self.b_close)
        self.b_cut.clicked.connect(self.do_cut); self.b_close.clicked.connect(self.close_tool)

        # ligações
        for w in (self.method, self.bed, self.shape, self.number, self.adj_axis):
            w.currentIndexChanged.connect(self.changed)
        self.method.currentIndexChanged.connect(self.update_visibility)
        for w in (self.n_planes,):
            w.valueChanged.connect(self.n_planes_changed)
        self.adj.valueChanged.connect(self.load_plane)
        for w in (self.gx, self.gy, self.gz, self.adj_index, self.r_cuts, self.sides, self.vrot):
            w.valueChanged.connect(self.changed)
        for w in (self.region_size, self.d1, self.d2, self.cone_h, self.adj_off, self.r_start, self.px, self.py, self.pz, self.border,
                  self.chamfer, self.depth_ratio, self.size, self.tol, self.lap_depth):
            w.valueChanged.connect(self.changed)
        for w in (self.sx, self.sy, self.sz):
            w.valueChanged.connect(self.size_changed)
        self.unit.currentIndexChanged.connect(self.unit_changed)
        for b in (self.regional, self.r_dir, self.multi, self.axial, self.close_cut, self.part_num, self.con, self.prof, self.lap):
            b.group.buttonToggled.connect(lambda *_: self.changed())
        self.staggered.toggled.connect(self.changed)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.preview)
        self.edit_group = None
        self.update_visibility()

    # ------------------------------------------------------------------
    def start(self, obj, edit=None):
        """Abre o painel para um objeto; com 'edit' (dados guardados numa peça já cortada) volta ao modelo original
        e às definições usadas, para ajustar e cortar de novo (as peças anteriores são substituídas)."""
        self.edit_group = None
        if edit is not None:
            obj = _Source(edit["source_mesh"], edit["source_name"])
            self.edit_group = edit["group"]
            self.set_values(edit["values"], edit.get("gap"))
        self.obj = obj
        self._loading = True
        k = C.UNITS[self.unit.currentData()]
        e = obj.mesh.extents / k
        self.sx.setValue(e[0]); self.sy.setValue(e[1]); self.sz.setValue(e[2])
        self._loading = False
        self.title.setText(f"{tr('Cortar')} — {obj.name}")
        self.preview()

    def update_visibility(self, *_):
        m = self.method.currentData()
        self.g_planar.setVisible(m in ("Planar cut", "Flexi cut"))
        self.g_conic.setVisible(m == "Conic cut")
        self.g_grid.setVisible(m == "Grid cut")
        self.g_radial.setVisible(m == "Radial cut")
        self.g_mod.setVisible(m == "Modular cut")
        self.g_common.setVisible(m != "Conic cut")
        self.g_con.setVisible(m not in ("Conic cut",))

    def set_normal(self, abc):
        self.A.setValue(abc[0]); self.B.setValue(abc[1]); self.Cc.setValue(abc[2])

    def n_planes_changed(self, n):
        while len(self.planes) < n:
            last = self.planes[-1]
            self.planes.append([last[0], last[1], last[2], max(-0.95, min(0.95, last[3] + 0.3))])
        self.adj.setRange(1, n)
        self.changed()

    def plane_changed(self):
        if self._loading:
            return
        i = self.adj.value() - 1
        while len(self.planes) <= i:
            self.planes.append([0, 0, 1, 0])
        self.planes[i] = [self.A.value(), self.B.value(), self.Cc.value(), self.D.value()]
        self.changed()

    def load_plane(self, k):
        self._loading = True
        a, b, c, d = self.planes[k - 1]
        self.A.setValue(a); self.B.setValue(b); self.Cc.setValue(c); self.D.setValue(d)
        self._loading = False
        self.changed()

    def unit_changed(self, *_):
        if self.obj:
            self.start(self.obj)

    def size_changed(self, *_):
        """Mudar o tamanho do modelo reescala-o."""
        if self._loading or not self.obj:
            return
        k = C.UNITS[self.unit.currentData()]
        e = self.obj.mesh.extents
        want = np.array([self.sx.value(), self.sy.value(), self.sz.value()]) * k
        from ..core import meshops as mo
        self.win.doc.snapshot()
        self.obj.mesh = mo.scale(self.obj.mesh.copy(), *(want / np.maximum(e, 1e-9)))
        self.win.doc.changed()
        self.changed()

    def values(self):
        return {"method": self.method.currentData(), "n_planes": self.n_planes.value(), "planes": [list(p) for p in self.planes],
                "regional": _radio_val(self.regional) == "Sim", "region_size": self.region_size.value(),
                "chamfer": self.chamfer.value(), "close": _radio_val(self.close_cut) == "Sim", "part_number": _radio_val(self.part_num) == "Sim",
                "connector": _radio_val(self.con), "profile": _radio_val(self.prof), "shape": self.shape.currentData(),
                "number": self.number.currentData(), "depth_ratio": self.depth_ratio.value(), "size": self.size.value(),
                "tolerance": self.tol.value(), "lap": _radio_val(self.lap) == "Sim", "lap_depth": self.lap_depth.value(),
                "cx": self.cx.value(), "cy": self.cy.value(), "cz": self.cz.value(), "dx": self.dx.value(), "dy": self.dy.value(),
                "dz": self.dz.value(), "sides": self.sides.value(), "vrot": self.vrot.value(), "d1": self.d1.value(), "d2": self.d2.value(),
                "cone_h": self.cone_h.value(), "gx": self.gx.value(), "gy": self.gy.value(), "gz": self.gz.value(),
                "staggered": self.staggered.isChecked(), "adj_axis": self.adj_axis.currentText(), "adj_index": self.adj_index.value(),
                "adj_offset": self.adj_off.value(), "r_dir": _radio_val(self.r_dir), "r_cuts": self.r_cuts.value(), "r_start": self.r_start.value(),
                "r_cx": self.r_cx.value(), "r_cy": self.r_cy.value(), "px": self.px.value(), "py": self.py.value(), "pz": self.pz.value(),
                "border": self.border.value(), "bed": self.bed.currentData(), "axial": _radio_val(self.axial) == "Sim"}

    def changed(self, *_):
        self.timer.start(120)

    def set_values(self, v, gap=None):
        """Repõe no painel as definições de um corte anterior."""
        def combo(w, val, data=True):
            i = w.findData(val) if data else w.findText(str(val))
            if i >= 0:
                w.setCurrentIndex(i)

        def radio(box, val):
            for b in box.group.buttons():
                b.setChecked(b.property("val") == val)
        yn = lambda x: "Sim" if x else "Não"
        self._loading = True
        try:
            combo(self.method, v.get("method"))
            self.planes = [list(p) for p in v.get("planes", [[0, 0, 1, 0]])] or [[0, 0, 1, 0]]
            self.n_planes.setValue(int(v.get("n_planes", len(self.planes))))
            radio(self.regional, yn(v.get("regional"))); self.region_size.setValue(v.get("region_size", self.region_size.value()))
            self.chamfer.setValue(v.get("chamfer", 0)); radio(self.close_cut, yn(v.get("close", True)))
            radio(self.part_num, yn(v.get("part_number"))); radio(self.con, v.get("connector")); radio(self.prof, v.get("profile"))
            combo(self.shape, v.get("shape")); combo(self.number, v.get("number"))
            for k, w in (("depth_ratio", self.depth_ratio), ("size", self.size), ("tolerance", self.tol), ("lap_depth", self.lap_depth),
                         ("cx", self.cx), ("cy", self.cy), ("cz", self.cz), ("dx", self.dx), ("dy", self.dy), ("dz", self.dz),
                         ("sides", self.sides), ("vrot", self.vrot), ("d1", self.d1), ("d2", self.d2), ("cone_h", self.cone_h),
                         ("gx", self.gx), ("gy", self.gy), ("gz", self.gz), ("adj_index", self.adj_index), ("adj_offset", self.adj_off),
                         ("r_cuts", self.r_cuts), ("r_start", self.r_start), ("r_cx", self.r_cx), ("r_cy", self.r_cy),
                         ("px", self.px), ("py", self.py), ("pz", self.pz), ("border", self.border)):
                if k in v:
                    w.setValue(type(w.value())(v[k]))
            radio(self.lap, yn(v.get("lap"))); self.staggered.setChecked(bool(v.get("staggered")))
            combo(self.adj_axis, v.get("adj_axis", "X"), data=False); radio(self.r_dir, v.get("r_dir")); combo(self.bed, v.get("bed"))
            radio(self.axial, yn(v.get("axial")))
            if gap is not None:
                self.gap.setValue(gap)
            self.adj.setRange(1, max(self.n_planes.value(), 1)); self.adj.setValue(1)
            a, b, c, d = self.planes[0]
            self.A.setValue(a); self.B.setValue(b); self.Cc.setValue(c); self.D.setValue(d)
        finally:
            self._loading = False
        self.update_visibility()

    def preview(self):
        if not self.obj:
            return
        v = self.values()
        try:
            planes = C.preview_planes(self.obj.mesh, v)
        except Exception as e:
            self.info.setText(f"⚠ {e}"); return
        self.win.view3d.show_cut_preview(self.obj.mesh, planes, self.adj.value() - 1 if v["method"] in ("Planar cut", "Flexi cut") else -1,
                                         cone=v if v["method"] == "Conic cut" else None)
        n = len(planes)
        if v["method"] == "Modular cut":
            nx, ny, nz = C.modular_counts(self.obj.mesh, (v["px"], v["py"], v["pz"]), v["border"], v["bed"])
            self.info.setText(f"{nx}×{ny}×{nz} = {nx * ny * nz} {tr('módulos')}")
        else:
            self.info.setText(f"{n} {tr('plano(s) de corte')}" if n else " ")

    def do_cut(self):
        """Corta (ou volta a cortar). O painel fica aberto: pode ajustar e cortar outra vez — as peças anteriores deste
        corte são substituídas, porque o corte é sempre feito sobre o modelo original guardado."""
        if not self.obj:
            return
        v = self.values(); o = self.obj; mesh = o.mesh.copy()
        gap = self.gap.value()

        def done(r):
            parts, extras, info = r
            if len(parts) < 2:
                msg = {"Conic cut": "O cone não atravessa o modelo: mude o centro (CX, CY, CZ — de 0 a 1 em cada eixo) ou aumente os diâmetros/altura.",
                       "Modular cut": "O modelo já cabe na impressora indicada: não é preciso cortar. Para dividir, reduza o tamanho da impressora X/Y/Z.",
                       "Grid cut": "A grelha não divide o modelo: aumente o número de divisões (X, Y ou Z).",
                       "Radial cut": "Os cortes radiais não passam pelo modelo: verifique o centro e a direção.",
                       }.get(v["method"], "O corte não dividiu o objeto: o plano não passa pelo modelo (ajuste D ou a orientação A, B, C).")
                self.win.statusBar().showMessage(msg, 10000)
                self.win.log("Cortar: " + msg)
                self.info.setText("⚠ " + msg)
                return
            C.explode_parts(parts, gap)
            _layout_extras(parts, extras)
            doc = self.win.doc
            doc.snapshot()
            group = self.edit_group or next(_GROUPS)
            src_mesh = o.mesh.copy()
            src_name = o.name
            if self.edit_group is not None:
                old = [x for x in doc.objects if (getattr(x, "tool", None) or {}).get("group") == group]
                for x in old:
                    doc.objects.remove(x)
            elif o in doc.objects:
                doc.objects.remove(o)
            tool = {"kind": "cut", "values": v, "gap": gap, "source_mesh": src_mesh, "source_name": src_name, "group": group}
            new = [doc.add(m, f"{src_name} parte {i + 1}") for i, m in enumerate(parts)]
            new += [doc.add(e, f"{src_name} cavilha {i + 1}") for i, e in enumerate(extras)]
            for x in new:
                x.tool = tool
            doc.changed()
            self.edit_group = group
            self.obj = _Source(src_mesh, src_name)
            self.win._select_objs(new[:len(parts)])
            self.title.setText(f"{tr('Cortar')} — {src_name} ({tr('a editar')})")
            self.win.log(f"Corte ({v['method']}): {len(parts)} peças" + (f", {len(extras)} cavilhas" if extras else "") +
                         (f" — {info}" if info else "") + " · pode ajustar e cortar de novo")
            self.preview()
        self.win.task(tr("A cortar…"), lambda: C.cut(mesh, v), done)

    def close_tool(self):
        self.win.view3d.clear_cut_preview()
        self.obj = None
        self.edit_group = None
        self.win.show_param_page()
