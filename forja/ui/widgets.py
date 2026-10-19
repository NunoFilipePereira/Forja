"""Componentes reutilizáveis: diálogo de parâmetros, tarefas em segundo plano, registo."""
import traceback
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from ..i18n import tr
from PySide6.QtWidgets import (QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QSpinBox,
                               QCheckBox, QLineEdit, QComboBox, QPlainTextEdit, QVBoxLayout, QLabel,
                               QProgressDialog, QApplication)


class ParamDialog(QDialog):
    """fields: lista de (chave, rótulo, valor_por_omissão[, opções]).
    Tipo inferido do valor: float, int, bool, str; lista de opções → caixa de escolha."""

    def __init__(self, parent, title, fields, note=None):
        super().__init__(parent)
        self.setWindowTitle(tr(title))
        lay = QVBoxLayout(self)
        if note:
            lb = QLabel(tr(note)); lb.setWordWrap(True); lay.addWidget(lb)
        form = QFormLayout()
        lay.addLayout(form)
        self.widgets = {}
        for f in fields:
            key, label, val = f[0], f[1], f[2]
            opts = f[3] if len(f) > 3 else None
            if isinstance(opts, list):
                w = QComboBox()
                for o in opts:
                    w.addItem(tr(str(o)), str(o))
                if str(val) in [str(o) for o in opts]:
                    w.setCurrentIndex([str(o) for o in opts].index(str(val)))
            elif isinstance(val, bool):
                w = QCheckBox(); w.setChecked(val)
            elif isinstance(val, int):
                w = QSpinBox(); w.setRange(*(opts or (-1_000_000, 1_000_000))); w.setValue(val)
            elif isinstance(val, float):
                w = QDoubleSpinBox(); w.setDecimals(3); w.setRange(*(opts or (-1e7, 1e7)))
                w.setValue(val); w.setSingleStep(max(abs(val) / 10, 0.1))
            elif isinstance(val, str) and "\n" in val:
                w = QPlainTextEdit(val); w.setMinimumHeight(120)
            else:
                w = QLineEdit(str(val))
            form.addRow(tr(label), w)
            self.widgets[key] = w
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self):
        out = {}
        for k, w in self.widgets.items():
            if isinstance(w, QComboBox): out[k] = w.currentData() if w.currentData() is not None else w.currentText()
            elif isinstance(w, QCheckBox): out[k] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)): out[k] = w.value()
            elif isinstance(w, QPlainTextEdit): out[k] = w.toPlainText()
            else: out[k] = w.text()
        return out

    @staticmethod
    def ask(parent, title, fields, note=None):
        d = ParamDialog(parent, title, fields, note)
        return d.values() if d.exec() == QDialog.Accepted else None


class TextDialog(QDialog):
    def __init__(self, parent, title, text, editable=False):
        super().__init__(parent)
        self.setWindowTitle(tr(title)); self.resize(720, 480)
        lay = QVBoxLayout(self)
        self.ed = QPlainTextEdit(text); self.ed.setReadOnly(not editable)
        f = self.ed.font(); f.setFamily("Menlo"); self.ed.setFont(f)
        lay.addWidget(self.ed)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | (QDialogButtonBox.Cancel if editable else QDialogButtonBox.NoButton))
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def show_text(parent, title, text):
        TextDialog(parent, title, text).exec()

    @staticmethod
    def edit(parent, title, text):
        d = TextDialog(parent, title, text, True)
        return d.ed.toPlainText() if d.exec() == QDialog.Accepted else None


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.sig = _Signals()

    def run(self):
        try:
            self.sig.done.emit(self.fn())
        except Exception as e:
            self.sig.failed.emit(f"{e}\n\n{traceback.format_exc(limit=3)}")


_live = set()


def run_task(parent, label, fn, on_done, on_error):
    """Executa fn() num fio separado com um indicador de progresso; chama on_done(resultado)."""
    prog = QProgressDialog(label, None, 0, 0, parent)
    prog.setWindowTitle("A processar"); prog.setWindowModality(Qt.WindowModal)
    prog.setMinimumDuration(300); prog.setValue(0)
    task = _Task(fn)
    _live.add(task)

    def fin(res):
        prog.close(); _live.discard(task); on_done(res)

    def err(msg):
        prog.close(); _live.discard(task); on_error(msg)

    task.sig.done.connect(fin)
    task.sig.failed.connect(err)
    QThreadPool.globalInstance().start(task)
