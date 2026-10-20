"""Painel de parâmetros: ajusta em tempo real qualquer objeto criado por um gerador paramétrico."""
import time
import traceback
import uuid
import numpy as np
from PySide6.QtCore import Qt, QTimer, QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel, QScrollArea, QDoubleSpinBox, QSpinBox,
                               QCheckBox, QLineEdit, QComboBox, QPushButton, QHBoxLayout)
from ..core import params as P
from ..i18n import tr

PREVIEW_SCALE = {"px_mm": 2.5, "res": 0.4, "layers": 0.5, "seg": 0.5, "sub": None}


def preview_values(v):
    """Versão leve dos valores para pré-visualizar enquanto se ajusta."""
    out = dict(v)
    for k, f in PREVIEW_SCALE.items():
        if k in out and f:
            if isinstance(out[k], int):
                out[k] = max(8, int(out[k] * f))
            else:
                out[k] = out[k] * f
    return out


class _Sig(QObject):
    done = Signal(object, object)   # (token, resultado ou Exception)


class _Job(QRunnable):
    def __init__(self, token, gid, values, src):
        super().__init__()
        self.token, self.gid, self.values, self.src = token, gid, values, src
        self.sig = _Sig()

    def run(self):
        try:
            r = P.build(self.gid, self.values, self.src)
        except Exception as e:
            r = e
        self.sig.done.emit(self.token, r)


class ParamPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.obj = None
        self.widgets = {}
        self._jobs = set()
        self._token = 0
        self._last_build = 0.0
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel(tr("Selecione um objeto criado na Forja 3D para ver os seus ajustes."))
        self.title.setWordWrap(True)
        f = self.title.font(); f.setBold(True); self.title.setFont(f)
        lay.addWidget(self.title)
        self.status = QLabel(" "); lay.addWidget(self.status)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        lay.addWidget(self.scroll, 1)
        row = QHBoxLayout(); lay.addLayout(row)
        self.b_reset = QPushButton(tr("Repor valores")); self.b_reset.clicked.connect(self.reset)
        self.b_new = QPushButton(tr("Criar cópia")); self.b_new.clicked.connect(self.copy_obj)
        row.addWidget(self.b_reset); row.addWidget(self.b_new)
        self.preview_timer = QTimer(self); self.preview_timer.setSingleShot(True); self.preview_timer.timeout.connect(lambda: self.rebuild(True))
        self.final_timer = QTimer(self); self.final_timer.setSingleShot(True); self.final_timer.timeout.connect(lambda: self.rebuild(False))
        self.show_obj(None)

    # ------------------------------------------------------------------
    def show_obj(self, obj):
        gen = getattr(obj, "gen", None) if obj else None
        self.obj = obj if gen else None
        self.widgets = {}
        box = QWidget(); outer = QVBoxLayout(box); outer.setContentsMargins(0, 0, 0, 0)
        self.b_reset.setEnabled(bool(gen)); self.b_new.setEnabled(bool(gen))
        if not gen:
            self.title.setText(tr("Selecione um objeto criado na Forja 3D para ver os seus ajustes."))
            self.status.setText(" ")
            outer.addStretch(1); self.scroll.setWidget(box)
            return
        spec = P.REGISTRY[gen["id"]]
        self.title.setText(f"{tr(spec['name'])} — {obj.name}")
        groups = {}
        for p in spec["params"]:
            key, label, default, opts, group = (list(p) + [None, None])[:5]
            if group not in groups:
                gb = QGroupBox(tr(group or "Parâmetros")); fl = QFormLayout(gb)
                groups[group] = fl; outer.addWidget(gb)
            val = gen["values"].get(key, default)
            w = self._widget(val if val is not None else default, opts, default)
            groups[group].addRow(tr(label), w)
            self.widgets[key] = w
        outer.addStretch(1)
        self.scroll.setWidget(box)
        self.status.setText(tr("Mude um valor: o objeto atualiza-se sozinho."))

    def _widget(self, val, opts, default):
        if isinstance(opts, list):
            w = QComboBox()
            for o in opts:
                w.addItem(tr(str(o)), str(o))
            if str(val) in [str(o) for o in opts]:
                w.setCurrentIndex([str(o) for o in opts].index(str(val)))
            w.currentIndexChanged.connect(self.changed)
        elif isinstance(default, bool):
            w = QCheckBox(); w.setChecked(bool(val)); w.toggled.connect(self.changed)
        elif isinstance(default, int):
            w = QSpinBox(); w.setRange(*(opts or (-10 ** 6, 10 ** 6))); w.setValue(int(val)); w.setKeyboardTracking(False)
            w.valueChanged.connect(self.changed)
        elif isinstance(default, float):
            w = QDoubleSpinBox(); w.setDecimals(3); w.setRange(*(opts or (-1e7, 1e7))); w.setValue(float(val))
            w.setSingleStep(max(abs(default) / 10, 0.1) if default else 0.1); w.setKeyboardTracking(False)
            w.valueChanged.connect(self.changed)
        else:
            w = QLineEdit(str(val)); w.editingFinished.connect(self.changed)
        return w

    def values(self):
        out = {}
        for k, w in self.widgets.items():
            if isinstance(w, QComboBox): out[k] = w.currentData()
            elif isinstance(w, QCheckBox): out[k] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)): out[k] = w.value()
            else: out[k] = w.text()
        return out

    # ------------------------------------------------------------------
    def changed(self, *_):
        if not self.obj:
            return
        self.obj.gen["values"] = self.values()
        # obra pesada: pré-visualização leve primeiro, versão final quando se para de mexer
        if self._last_build > 0.8:
            self.preview_timer.start(150)
            self.final_timer.start(1500)
        else:
            self.final_timer.start(120)

    def rebuild(self, preview):
        if not self.obj:
            return
        gen = self.obj.gen
        self._token += 1
        token = (self._token, preview, gen["group"])
        vals = preview_values(gen["values"]) if preview else dict(gen["values"])
        job = _Job(token, gen["id"], vals, gen.get("src") or {})
        self._jobs.add(job)
        job.sig.done.connect(lambda t, r, job=job: self._done(t, r, job))
        self.status.setText(tr("A atualizar (pré-visualização)…") if preview else tr("A atualizar…"))
        self._t0 = time.time()
        QThreadPool.globalInstance().start(job)

    def _done(self, token, result, job):
        self._jobs.discard(job)
        n, preview, group = token
        if n != self._token:
            return  # já há um pedido mais recente
        if isinstance(result, Exception):
            self.status.setText(f"⚠ {result}")
            return
        if not preview:
            self._last_build = time.time() - self._t0
        self.win.replace_group(group, result if isinstance(result, list) else [result], final=not preview)
        self.status.setText(tr("Pré-visualização (a calcular a versão final…)") if preview else
                            f"{tr('Atualizado')} ({self._last_build:.1f} s)")

    def reset(self):
        if self.obj:
            self.obj.gen["values"] = P.defaults(self.obj.gen["id"])
            self.show_obj(self.obj)
            self.rebuild(False)

    def copy_obj(self):
        if self.obj:
            g = self.obj.gen
            self.win.create_param(g["id"], dict(g["values"]), g.get("src"), name=self.obj.name + " cópia", ask=False)


def new_gen(gid, values, src):
    return {"id": gid, "values": dict(values), "src": src or {}, "group": uuid.uuid4().hex}
