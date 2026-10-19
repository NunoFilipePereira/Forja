"""Painel Lote: carregar ficheiros ou uma pasta, folhear (primeiro, último, página acima/abaixo,
anterior, seguinte), listar nomes, renomear, apagar (para o Lixo), exportar todos e aplicar operações em lote."""
import os
import glob
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                               QInputDialog, QApplication)
from ..i18n import tr

FORMATS = {
    "Todos os formatos suportados": ["stl", "obj", "ply", "off", "3mf", "glb", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif",
                                     "dxf", "svg", "forja"],
    "Malhas 3D (STL, OBJ, PLY, OFF, 3MF, GLB)": ["stl", "obj", "ply", "off", "3mf", "glb"],
    "Imagens (PNG, JPEG, BMP, TIFF, GIF)": ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif"],
    "Desenhos 2D (DXF, SVG)": ["dxf", "svg"],
    "Projetos Forja (*.forja)": ["forja"],
}
PAGE = 10


class BatchPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.files = []
        self.idx = -1
        self.shown = []  # ids dos objetos 3D / nomes das imagens mostrados pelo lote
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout(); lay.addLayout(row)
        b1, b2 = QPushButton(tr("Carregar ficheiros…")), QPushButton(tr("Carregar pasta…"))
        b1.clicked.connect(self.load_files); b2.clicked.connect(self.load_folder)
        row.addWidget(b1); row.addWidget(b2)
        self.lst = QListWidget(); lay.addWidget(self.lst, 1)
        self.lst.currentRowChanged.connect(self._row)
        self.pos = QLabel("—"); lay.addWidget(self.pos)
        nav = QHBoxLayout(); lay.addLayout(nav)
        for txt, fn, tip in (("⏮", self.first, "Primeiro (Home)"), ("⏫", self.page_up, "Página acima (PgUp)"),
                             ("◀", self.prev, "Anterior (↑ / ←)"), ("▶", self.next, "Seguinte (↓ / →)"),
                             ("⏬", self.page_down, "Página abaixo (PgDn)"), ("⏭", self.last, "Último (End)")):
            b = QPushButton(txt); b.setToolTip(tr(tip)); b.clicked.connect(fn); b.setMaximumWidth(44); nav.addWidget(b)
        ops = QHBoxLayout(); lay.addLayout(ops)
        for txt, fn in (("Listar nomes", self.list_names), ("Renomear…", self.rename), ("Apagar", self.delete)):
            b = QPushButton(tr(txt)); b.clicked.connect(fn); ops.addWidget(b)
        ops2 = QHBoxLayout(); lay.addLayout(ops2)
        for txt, fn in (("Exportar todos…", self.export_all), ("Operação em lote…", self.batch_op)):
            b = QPushButton(tr(txt)); b.clicked.connect(fn); ops2.addWidget(b)
        for key, fn in (("Home", self.first), ("End", self.last), ("PgUp", self.page_up), ("PgDown", self.page_down)):
            sc = QShortcut(QKeySequence(key), self.lst); sc.setContext(Qt.WidgetWithChildrenShortcut); sc.activated.connect(fn)

    # ------------------------------------------------------------------ carregar
    def load_files(self):
        exts = " ".join(f"*.{e}" for e in FORMATS["Todos os formatos suportados"])
        ps, _ = QFileDialog.getOpenFileNames(self, tr("Carregar lote"), self.win.last_dir, f"{tr('Suportados')} ({exts})")
        if ps:
            self.set_files(ps)

    def load_folder(self, folder=None, fmt=None):
        if not folder:
            fmt_, ok = QInputDialog.getItem(self, tr("Carregar pasta"), tr("Escolha o formato"), list(FORMATS), 0, False)
            if not ok:
                return
            fmt = fmt_
            folder = QFileDialog.getExistingDirectory(self, tr("Carregar pasta"), self.win.last_dir)
            if not folder:
                return
        exts = FORMATS.get(fmt or "Todos os formatos suportados")
        fs = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().rsplit(".", 1)[-1] in exts)
        if not fs:
            return self.win.error(tr("A pasta não tem ficheiros nesse formato."))
        self.set_files(fs)

    def set_files(self, files):
        self.files = list(files)
        self.lst.blockSignals(True); self.lst.clear()
        for f in self.files:
            self.lst.addItem(os.path.basename(f))
        self.lst.blockSignals(False)
        self.win.log(f"Lote: {len(self.files)} ficheiro(s)")
        self.go(0)

    # ------------------------------------------------------------------ navegar
    def _row(self, r):
        if r != self.idx and 0 <= r < len(self.files):
            self.go(r)

    def go(self, i):
        if not self.files:
            return
        i = max(0, min(len(self.files) - 1, i))
        self.idx = i
        self.lst.blockSignals(True); self.lst.setCurrentRow(i); self.lst.blockSignals(False)
        self.pos.setText(f"{i + 1} / {len(self.files)} — {os.path.basename(self.files[i])}")
        self._show(self.files[i])

    def _show(self, path):
        w = self.win
        # retira o que o lote mostrou antes (não mexe nos outros objetos)
        old = [o for o in w.doc.objects if o.id in self.shown]
        if old:
            w.doc.remove(old)
        n_obj = {o.id for o in w.doc.objects}
        n_img = len(w.images)
        ext = path.lower().rsplit(".", 1)[-1]
        if ext in ("stl", "obj", "ply", "off", "3mf", "glb"):
            import trimesh
            try:
                m = trimesh.load(path, force="mesh")
                w.add_mesh(m, os.path.basename(path), select=False, place=False)
            except Exception as e:
                w.error(f"{os.path.basename(path)}: {e}")
        else:
            w.open_path(path)
        self.shown = [o.id for o in w.doc.objects if o.id not in n_obj]
        if self.shown:
            w._select_objs([o for o in w.doc.objects if o.id in self.shown])
            w.tabs.setCurrentWidget(w.view3d)
            try:
                w.view3d.fit()
            except Exception:
                pass
        elif len(w.images) > n_img:
            w.tabs.setCurrentWidget(w.imgview.parentWidget() if hasattr(w.imgview, "parentWidget") else w.imgview)

    def first(self): self.go(0)
    def last(self): self.go(len(self.files) - 1)
    def prev(self): self.go(self.idx - 1)
    def next(self): self.go(self.idx + 1)
    def page_up(self): self.go(self.idx - PAGE)
    def page_down(self): self.go(self.idx + PAGE)

    # ------------------------------------------------------------------ ficheiros
    def list_names(self):
        if not self.files:
            return
        txt = "\n".join(self.files)
        QApplication.clipboard().setText(txt)
        p, _ = QFileDialog.getSaveFileName(self, tr("Guardar a lista de nomes"), os.path.join(self.win.last_dir, "lista.txt"), "Texto (*.txt)")
        if p:
            open(p, "w").write(txt + "\n")
        self.win.log(f"Lote: {len(self.files)} nomes copiados" + (f" e guardados em {p}" if p else ""))

    def rename(self):
        """Renomeia todos os ficheiros do lote: modelo com {n} (número) e {nome} (nome atual)."""
        if not self.files:
            return
        pat, ok = QInputDialog.getText(self, tr("Renomear"), tr("Novo nome ({n} = número, {nome} = nome atual):"), text="peca_{n:03d}")
        if not ok or not pat:
            return
        new = []
        for k, f in enumerate(self.files, 1):
            d, base = os.path.split(f)
            stem, ext = os.path.splitext(base)
            try:
                nn = pat.format(n=k, nome=stem) + ext
            except Exception:
                return self.win.error(tr("Modelo de nome inválido. Exemplo: peca_{n:03d}"))
            dst = os.path.join(d, nn)
            if os.path.exists(dst) and dst != f:
                return self.win.error(f"{nn}: {tr('já existe um ficheiro com esse nome.')}")
            new.append(dst)
        for f, dst in zip(self.files, new):
            if f != dst:
                os.rename(f, dst)
        self.win.log(f"Lote: {len(new)} ficheiro(s) renomeados")
        self.set_files(new)

    def delete(self):
        """Apaga o ficheiro atual — vai para o Lixo (pode ser recuperado)."""
        if not self.files or self.idx < 0:
            return
        f = self.files[self.idx]
        if QMessageBox.question(self, tr("Apagar"), f"{tr('Mover para o Lixo')}:\n{f}?") != QMessageBox.Yes:
            return
        from PySide6.QtCore import QFile
        if not QFile.moveToTrash(f):
            return self.win.error(tr("Não foi possível mover o ficheiro para o Lixo."))
        del self.files[self.idx]
        self.lst.takeItem(self.idx)
        self.win.log(f"Lote: {os.path.basename(f)} movido para o Lixo")
        if self.files:
            self.go(min(self.idx, len(self.files) - 1))

    # ------------------------------------------------------------------ exportar / operações
    def export_all(self):
        if not self.files:
            return
        import trimesh
        meshes = [f for f in self.files if f.lower().rsplit(".", 1)[-1] in FORMATS["Malhas 3D (STL, OBJ, PLY, OFF, 3MF, GLB)"]]
        imgs = [f for f in self.files if f.lower().rsplit(".", 1)[-1] in FORMATS["Imagens (PNG, JPEG, BMP, TIFF, GIF)"]]
        fmts = (["stl", "obj", "ply", "off", "3mf"] if meshes else []) + (["png", "jpg", "bmp", "tif"] if imgs else [])
        if not fmts:
            return self.win.error(tr("O lote não tem malhas nem imagens para exportar."))
        fmt, ok = QInputDialog.getItem(self, tr("Exportar todos"), tr("Formato"), fmts, 0, False)
        if not ok:
            return
        d = QFileDialog.getExistingDirectory(self, tr("Pasta de destino"), self.win.last_dir)
        if not d:
            return
        n = 0
        for f in self.files:
            ext = f.lower().rsplit(".", 1)[-1]
            stem = os.path.splitext(os.path.basename(f))[0]
            if f in meshes and fmt in ("stl", "obj", "ply", "off", "3mf"):
                trimesh.load(f, force="mesh").export(os.path.join(d, f"{stem}.{fmt}")); n += 1
            elif f in imgs and fmt in ("png", "jpg", "bmp", "tif"):
                import cv2
                a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if a is not None:
                    cv2.imwrite(os.path.join(d, f"{stem}.{fmt}"), a); n += 1
        self.win.log(f"Lote: {n} ficheiro(s) exportados para {d}")

    def batch_op(self):
        meshes = [f for f in self.files if f.lower().rsplit(".", 1)[-1] in FORMATS["Malhas 3D (STL, OBJ, PLY, OFF, 3MF, GLB)"]]
        if not meshes:
            return self.win.error(tr("O lote não tem modelos 3D."))
        self.win.run_batch(meshes)
