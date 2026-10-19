"""Painel genérico de ferramentas: Entrada (método, unidade, tamanho do modelo), Saída (parâmetros do
método escolhido), Medidas (resultado), pré-visualização ao vivo e botões Gravar modelo / Fechar.

Especificação de uma ferramenta:
  {"title", "methods": [...], "fields": [(chave, rótulo, omissão, opções, secção, métodos_onde_aparece)], "run": fn(malha, valores, rápido)}
  opções: lista → escolha; ("radio", [...]) → botões de opção; tuplo numérico → intervalo; None → automático.
  run devolve {"meshes": [...], "shapes": [...] (opcional, formas 2D), "measure": {rótulo: valor}}."""
import time
import numpy as np
from PySide6.QtCore import Qt, QTimer, QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel, QScrollArea, QDoubleSpinBox, QSpinBox,
                               QCheckBox, QLineEdit, QComboBox, QPushButton, QHBoxLayout, QRadioButton, QButtonGroup, QFileDialog)
from ..i18n import tr

UNITS = {"meter": 1000.0, "dm": 100.0, "cm": 10.0, "mm": 1.0, "inch": 25.4}


class _Sig(QObject):
    done = Signal(object, object)


class _Job(QRunnable):
    def __init__(self, token, fn):
        super().__init__(); self.token, self.fn = token, fn; self.sig = _Sig()

    def run(self):
        try:
            r = self.fn()
        except Exception as e:
            r = e
        self.sig.done.emit(self.token, r)


class ToolPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.spec = None
        self.obj = None
        self.widgets = {}
        self.rows = {}
        self._token = 0
        self._jobs = set()
        self.result = None
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel(""); f = self.title.font(); f.setBold(True); self.title.setFont(f); lay.addWidget(self.title)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); lay.addWidget(self.scroll, 1)
        self.status = QLabel(" "); self.status.setWordWrap(True); lay.addWidget(self.status)
        row = QHBoxLayout(); lay.addLayout(row)
        self.b_save = QPushButton(tr("Gravar modelo")); self.b_close = QPushButton(tr("Fechar"))
        self.b_save.setDefault(True)
        row.addWidget(self.b_save); row.addWidget(self.b_close)
        self.b_save.clicked.connect(self.save); self.b_close.clicked.connect(self.close_tool)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(lambda: self.run(True))
        self.final_timer = QTimer(self); self.final_timer.setSingleShot(True); self.final_timer.timeout.connect(lambda: self.run(False))

    # ------------------------------------------------------------------
    def lab(self, x):
        return tr((self.spec or {}).get("labels", {}).get(x, x)) if isinstance(x, str) else str(x)

    def start(self, spec, obj=None, key=None, edit=None):
        """Abre a ferramenta. 'key' identifica-a (para as peças guardarem com que ferramenta foram feitas);
        'edit' = dados guardados numa peça (valores usados, grupo) para voltar a ajustar e substituir o resultado."""
        self.key = key
        self.edit_group = edit["group"] if edit else None
        self.spec, self.obj = spec, obj
        self.result = None
        self.title.setText(tr(spec["title"]) + (f" — {obj.name}" if obj else ""))
        self.b_save.setText(tr(spec.get("save_label", "Gravar modelo")))
        if hasattr(self, "_started"):
            del self._started
        body = QWidget(); outer = QVBoxLayout(body)
        self.widgets, self.rows, groups = {}, {}, {}
        # Entrada
        gi = QGroupBox(tr("Entrada")); fi = QFormLayout(gi); outer.addWidget(gi)
        if spec.get("methods"):
            m = QComboBox()
            for x in spec["methods"]:
                m.addItem(self.lab(x), x)
            m.currentIndexChanged.connect(self._method_changed)
            fi.addRow(tr("Página" if spec.get("pages") else "Método"), m); self.widgets["method"] = m
            if spec.get("pages"):
                nav = QWidget(); nl = QHBoxLayout(nav); nl.setContentsMargins(0, 0, 0, 0)
                bp, bn = QPushButton(tr("◀ Anterior")), QPushButton(tr("Seguinte ▶"))
                bp.clicked.connect(lambda: m.setCurrentIndex(max(m.currentIndex() - 1, 0)))
                bn.clicked.connect(lambda: m.setCurrentIndex(min(m.currentIndex() + 1, m.count() - 1)))
                nl.addWidget(bp); nl.addWidget(bn); fi.addRow("", nav)
        if obj is not None and spec.get("model_size", True):
            u = QComboBox()
            for x in UNITS:
                u.addItem(x, x)
            u.setCurrentText("mm"); u.currentIndexChanged.connect(self._fill_size)
            fi.addRow(tr("Unidade"), u); self.widgets["_unit"] = u
            for ax in "XYZ":
                w = QDoubleSpinBox(); w.setDecimals(3); w.setRange(0.001, 1e7); w.setKeyboardTracking(False)
                w.valueChanged.connect(self._size_changed)
                fi.addRow(tr(f"Tamanho do modelo {ax}"), w); self.widgets[f"_size{ax}"] = w
        groups["Entrada"] = (gi, fi)  # os campos da secção "Entrada" juntam-se ao método
        for fdef in spec["fields"]:
            key, label, default, opts, group = fdef[:5]
            methods = fdef[5] if len(fdef) > 5 else None
            if group not in groups:
                gb = QGroupBox(tr(group or "Saída")); fl = QFormLayout(gb); groups[group] = (gb, fl); outer.addWidget(gb)
            w = self._widget(default, opts)
            lab = QLabel(tr(label))
            groups[group][1].addRow(lab, w)
            uid = key
            n = 1
            while uid in self.widgets:  # a mesma chave pode repetir-se com valores por omissão diferentes por método
                n += 1; uid = f"{key}#{n}"
            self.widgets[uid] = w
            self.rows[uid] = (lab, w, methods, group)
        self.measure_box = QGroupBox(tr("Medidas")); self.measure = QFormLayout(self.measure_box); outer.addWidget(self.measure_box)
        outer.addStretch(1)
        self.scroll.setWidget(body)
        self._groups = groups
        self._fill_size()
        if edit:
            self.set_values(edit["values"])
        self._method_changed()

    FILTERS = {"image": "Imagens (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif)",
               "mesh": "Modelos 3D (*.stl *.obj *.ply *.off *.3mf *.glb)"}

    def _widget(self, default, opts):
        if isinstance(opts, tuple) and len(opts) == 2 and opts[0] == "button":
            b = QPushButton(tr(str(default)))
            b.clicked.connect(lambda: opts[1](self))
            b._button = True
            return b
        if isinstance(opts, tuple) and len(opts) == 2 and opts[0] == "files":
            # lista de ficheiros com Adicionar / Remover / Subir / Descer
            from PySide6.QtWidgets import QListWidget, QGridLayout
            box = QWidget(); g = QGridLayout(box); g.setContentsMargins(0, 0, 0, 0)
            lst = QListWidget(); lst.setMaximumHeight(110)
            g.addWidget(lst, 0, 0, 1, 4)
            bad, brm, bup, bdn = (QPushButton(tr(t)) for t in ("Adicionar…", "Remover", "Subir", "Descer"))
            for i, b in enumerate((bad, brm, bup, bdn)):
                g.addWidget(b, 1, i)

            def add():
                import os
                ps, _ = QFileDialog.getOpenFileNames(self, tr("Escolher ficheiros"), getattr(self.win, "last_dir", os.path.expanduser("~")),
                                                     self.FILTERS.get(opts[1], "*"))
                for p_ in ps:
                    lst.addItem(p_)
                if ps:
                    self.changed()

            def rm():
                for it in lst.selectedItems():
                    lst.takeItem(lst.row(it))
                self.changed()

            def move(d):
                r = lst.currentRow()
                if r < 0 or not (0 <= r + d < lst.count()):
                    return
                it = lst.takeItem(r); lst.insertItem(r + d, it); lst.setCurrentRow(r + d); self.changed()
            bad.clicked.connect(add); brm.clicked.connect(rm)
            bup.clicked.connect(lambda: move(-1)); bdn.clicked.connect(lambda: move(1))
            box._files = lst
            for p_ in (default or []):
                lst.addItem(p_)
            return box
        if isinstance(opts, tuple) and len(opts) == 2 and opts[0] == "file":
            # ficheiro (botão Procurar); vazio = imagem atual da Forja ou exemplo
            box = QWidget(); l = QHBoxLayout(box); l.setContentsMargins(0, 0, 0, 0)
            e = QLineEdit(str(default or "")); e.setPlaceholderText(tr("(imagem atual / exemplo)") if opts[1] == "image" else tr("(objeto selecionado)"))
            b = QPushButton(tr("Procurar…"))
            l.addWidget(e, 1); l.addWidget(b)

            def browse():
                import os
                p, _ = QFileDialog.getOpenFileName(self, tr("Escolher ficheiro"), getattr(self.win, "last_dir", os.path.expanduser("~")),
                                                   self.FILTERS.get(opts[1], "*"))
                if p:
                    e.setText(p); self.changed()
            b.clicked.connect(browse); e.editingFinished.connect(self.changed)
            box._file = e
            return box
        if isinstance(opts, tuple) and len(opts) == 2 and opts[0] == "radio":
            box = QWidget(); l = QHBoxLayout(box); l.setContentsMargins(0, 0, 0, 0)
            grp = QButtonGroup(box)
            for k, o in enumerate(opts[1]):
                b = QRadioButton(self.lab(o)); b.setProperty("val", o); grp.addButton(b, k); l.addWidget(b); b.setChecked(o == default)
            grp.buttonToggled.connect(lambda *_: self.changed())
            box._rgroup = grp
            return box
        if isinstance(opts, list):
            w = QComboBox()
            for o in opts:
                w.addItem(self.lab(o), o)
            if default in opts:
                w.setCurrentIndex(opts.index(default))
            w.currentIndexChanged.connect(self.changed)
            return w
        if isinstance(default, bool):
            w = QCheckBox(); w.setChecked(default); w.toggled.connect(self.changed); return w
        if isinstance(default, int):
            w = QSpinBox(); w.setRange(*(opts or (-10 ** 6, 10 ** 6))); w.setValue(default); w.setKeyboardTracking(False)
            w.valueChanged.connect(self.changed); return w
        if isinstance(default, float):
            w = QDoubleSpinBox(); w.setDecimals(3); w.setRange(*(opts or (-1e7, 1e7))); w.setValue(default)
            w.setSingleStep(max(abs(default) / 10, 0.05) if default else 0.1); w.setKeyboardTracking(False)
            w.valueChanged.connect(self.changed); return w
        w = QLineEdit(str(default)); w.editingFinished.connect(self.changed); return w

    @staticmethod
    def _value(w):
        if hasattr(w, "_files"):
            return [w._files.item(i).text() for i in range(w._files.count())]
        if hasattr(w, "_file"):
            return w._file.text().strip()
        if hasattr(w, "_rgroup"):
            b = w._rgroup.checkedButton(); return b.property("val") if b else None
        if isinstance(w, QComboBox): return w.currentData()
        if isinstance(w, QCheckBox): return w.isChecked()
        if isinstance(w, (QSpinBox, QDoubleSpinBox)): return w.value()
        return w.text()

    def values(self):
        v = {}
        m = self.widgets["method"].currentData() if "method" in self.widgets else None
        if "method" in self.widgets:
            v["method"] = m
        active = set()
        for uid, w in self.widgets.items():
            if uid.startswith("_") or uid == "method" or hasattr(w, "_button"):
                continue
            key = uid.split("#")[0]
            methods = self.rows[uid][2] if uid in self.rows else None
            on = methods is None or m in methods
            if on or (key not in v and key not in active):
                v[key] = self._value(w)
                if on:
                    active.add(key)
        return v

    def set_values(self, vals):
        """Repõe no painel os valores usados numa aplicação anterior da ferramenta."""
        m = vals.get("method")
        if m is not None and "method" in self.widgets:
            w = self.widgets["method"]; i = w.findData(m)
            if i >= 0:
                w.blockSignals(True); w.setCurrentIndex(i); w.blockSignals(False)
        for uid, w in self.widgets.items():
            if uid.startswith("_") or uid == "method" or hasattr(w, "_button"):
                continue
            key = uid.split("#")[0]
            methods = self.rows[uid][2] if uid in self.rows else None
            if key not in vals or not (methods is None or m in methods):
                continue
            val = vals[key]
            w.blockSignals(True)
            try:
                if hasattr(w, "_files"):
                    w._files.clear(); [w._files.addItem(x) for x in (val or [])]
                elif hasattr(w, "_file"):
                    w._file.setText(str(val or ""))
                elif hasattr(w, "_rgroup"):
                    for b in w._rgroup.buttons():
                        b.setChecked(b.property("val") == val)
                elif isinstance(w, QComboBox):
                    i = w.findData(val)
                    if i >= 0:
                        w.setCurrentIndex(i)
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(val))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
                elif isinstance(w, QLineEdit):
                    w.setText(str(val))
            except Exception:
                pass
            finally:
                w.blockSignals(False)

    def _method_changed(self, *_):
        m = self.widgets["method"].currentData() if "method" in self.widgets else None
        shown = {}
        for key, (lab, w, methods, group) in self.rows.items():
            vis = methods is None or m in methods
            lab.setVisible(vis); w.setVisible(vis)
            shown[group] = shown.get(group, False) or vis
        for group, (gb, fl) in self._groups.items():
            gb.setVisible(shown.get(group, False) or (group == "Entrada" and fl.rowCount() > 0 and
                                                       any(k.startswith("_") or k == "method" for k in self.widgets)))
        if not (self.spec or {}).get("pages") or not hasattr(self, "_started"):
            self._started = True
            self.changed()

    def _fill_size(self, *_):
        if self.obj is None or "_unit" not in self.widgets:
            return
        k = UNITS[self.widgets["_unit"].currentData()]
        e = self.obj.mesh.extents / k
        for ax, val in zip("XYZ", e):
            w = self.widgets[f"_size{ax}"]; w.blockSignals(True); w.setValue(val); w.blockSignals(False)

    def _size_changed(self, *_):
        if self.obj is None:
            return
        from ..core import meshops as mo
        k = UNITS[self.widgets["_unit"].currentData()]
        want = np.array([self.widgets[f"_size{a}"].value() for a in "XYZ"]) * k
        e = np.maximum(self.obj.mesh.extents, 1e-9)
        self.win.doc.snapshot()
        self.obj.mesh = mo.scale(self.obj.mesh.copy(), *(want / e))
        self.win.doc.changed()
        self.changed()

    # ------------------------------------------------------------------
    def changed(self, *_):
        if not self.spec:
            return
        self.timer.start(150)
        if self.spec.get("auto_final", True) and not self.spec.get("pages"):
            self.final_timer.start(1200)

    def run(self, fast):
        if not self.spec:
            return
        self._token += 1
        token = (self._token, fast)
        v = self.values(); spec = self.spec
        mesh = self.obj.mesh.copy() if self.obj is not None else None
        job = _Job(token, lambda: spec["run"](mesh, v, fast))
        self._jobs.add(job)
        job.sig.done.connect(lambda t, r, job=job: self._done(t, r, job))
        self.status.setText(tr("A calcular…"))
        self._t0 = time.time()
        QThreadPool.globalInstance().start(job)

    def _done(self, token, r, job):
        if job is not None:
            self._jobs.discard(job)
        if token[0] != self._token:
            return
        if isinstance(r, Exception):
            self.status.setText(f"⚠ {r}"); return
        self.result = r
        hide = [self.obj.id] if self.obj is not None and r.get("meshes") else []
        g = getattr(self, "edit_group", None)
        if g is not None and r.get("meshes"):
            hide += [x.id for x in self.win.doc.objects if (getattr(x, "tool", None) or {}).get("group") == g]
        self.win.view3d.show_preview_meshes(r.get("meshes", []), colors=r.get("colors"), hide=hide)
        if r.get("lines"):
            self.win.view3d.show_preview_lines(r["lines"])
        while self.measure.rowCount():
            self.measure.removeRow(0)
        for k, val in (r.get("measure") or {}).items():
            self.measure.addRow(tr(k), QLabel(str(val)))
        self.status.setText((tr("Pré-visualização") if token[1] else tr("Pronto")) + f" ({time.time() - self._t0:.1f} s)")

    def save(self):
        if not self.spec:
            return
        v = self.values(); spec = self.spec
        mesh = self.obj.mesh.copy() if self.obj is not None else None
        name = (self.obj.name + " — " if self.obj else "") + self.lab(v.get("method") or spec["title"])

        if spec.get("save"):
            self.win.task(tr("A calcular…"), lambda: spec["run"](mesh, v, False), lambda r: (spec["save"](self.win, r, v), self._done((self._token, False), r, None)))
            return

        key, group = getattr(self, "key", None), getattr(self, "edit_group", None)

        def done(r):
            meshes = r.get("meshes", [])
            names = r.get("names") or []
            cols = r.get("colors") or []
            doc = self.win.doc
            # voltar a gravar substitui o resultado anterior desta ferramenta (o mesmo grupo)
            nonlocal group
            old = [x for x in doc.objects if group is not None and (getattr(x, "tool", None) or {}).get("group") == group]
            old_pos = None
            if old:
                import numpy as _np
                old_pos = _np.min([x.mesh.bounds[0] for x in old], axis=0)
                doc.snapshot()
                for x in old:
                    doc.objects.remove(x)
            if group is None:
                from .cutpanel import _GROUPS
                group = next(_GROUPS)
            tool = {"kind": "tool", "key": key, "values": dict(v), "group": group,
                    "source_id": self.obj.id if self.obj is not None else None}
            shift = None
            if old_pos is not None and meshes:
                import numpy as _np
                lo = _np.min([m.bounds[0] for m in meshes], axis=0)
                shift = old_pos - lo
                for m in meshes:
                    m.apply_translation(shift)
            for k, m in enumerate(meshes):
                nm = name if len(meshes) == 1 else f"{name} — {names[k]}" if k < len(names) else f"{name} {k + 1}"
                if self.obj is None and old_pos is None:
                    # a 1.ª peça é colocada ao lado do que já existe; as outras acompanham-na (conjunto montado)
                    if shift is None:
                        b0 = m.bounds[0].copy(); self.win._place_new(m); shift = m.bounds[0] - b0
                    else:
                        m.apply_translation(shift)
                o = self.win.add_mesh(m, nm, select=False, place=False)
                if key:
                    o.tool = tool
                if k < len(cols) and cols[k] is not None:
                    rgba = np.array(list(cols[k])[:3] + [1.0], np.float32)
                    o.vcolors = (id(o.mesh), np.tile(rgba, (len(o.mesh.vertices), 1)))
            if r.get("shapes"):
                self.win.canvas.add_shapes(r["shapes"])
            self.win.log(f"{name}: {len(meshes)} objeto(s)" + (f", {len(r['shapes'])} formas 2D" if r.get("shapes") else ""))
            if self.obj is not None and meshes:
                self.obj.visible = False
                self.win.doc.changed()
            if not key:
                self.close_tool(); return
            # o painel fica aberto: pode ajustar e gravar de novo (substitui este resultado)
            self.edit_group = group
            self.win.view3d.clear_preview_meshes()
            self.status.setText(tr("Gravado. Pode ajustar e gravar de novo — o resultado anterior é substituído."))
            self.b_save.setText(tr("Gravar de novo (substituir)"))
        self.win.task(tr("A gerar…"), lambda: spec["run"](mesh, v, False), done)

    def close_tool(self):
        self.spec = None
        self.win.view3d.clear_preview_meshes()
        self.win.show_param_page()
