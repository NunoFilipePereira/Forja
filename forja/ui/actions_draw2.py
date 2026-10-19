"""Desenho 2D avançado: formas, curvas, estilos, ordem, área de transferência, lista de formas, padrões, vetorização."""
import os
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QFileDialog, QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox
from PySide6.QtCore import Qt
from ..core import shapes2d as s2, draw2 as d2, generators as gn, imageops as iop


class Draw2Actions:
    def menus_draw2(self):
        return [("&Desenho 2D", [
            None,
            ("Inserir", [
                ("Arco (3 cliques)", lambda: self.set_tool("arc")),
                ("Mão livre", lambda: self.set_tool("ff")),
                ("Leque (setor de coroa)…", self.d2_fan),
                ("Forma básica…", self.d2_basic),
                ("Curva (espiral, onda, dente de serra…)…", self.d2_curve),
                ("Texto com tipo de letra…", self.d2_text),
                ("Grelha de régua…", self.d2_grid_ruler),
                ("Imagem atual (para decalcar)…", self.d2_insert_image),
            ]),
            ("Espaçamento da grelha…", self.d2_grid_spacing),
            ("Estilo da seleção", [
                ("Cor do traço…", lambda: self.d2_color("stroke")),
                ("Cor de preenchimento…", lambda: self.d2_color("fill")),
                ("Sem preenchimento", lambda: self.d2_set_style(fill=None)),
                ("Espessura da linha…", self.d2_width),
                ("Tipo de linha…", self.d2_dash),
                ("Setas nas pontas…", self.d2_arrows),
            ]),
            ("Ordem", [("Trazer para a frente", lambda: self.d2_order(True)), ("Enviar para trás", lambda: self.d2_order(False))]),
            ("Copiar", self.d2_copy, "Ctrl+C"), ("Colar", self.d2_paste, "Ctrl+V"),
            ("Lista de formas (nomes, mostrar/ocultar)…", self.d2_list),
            ("Mostrar/ocultar réguas", self.d2_rulers),
            None,
            ("Padrão de recorte dentro da forma…", self.d2_pattern),
            ("Imagem → meio-tom vetorial (laser)…", self.d2_halftone),
            ("Fotografia → vetor por níveis…", self.d2_photo_vector),
            ("Exportar EPS…", self.d2_export_eps),
        ])]

    def _sel2d(self):
        sel = self.canvas.selected_shapes()
        if not sel:
            self.error("Selecione formas no separador 2D.")
        return sel

    # --- inserir -----------------------------------------------------------
    def d2_grid_ruler(self):
        """Grelha de régua: quadrícula com marcas a cada passo."""
        from shapely.geometry import LineString, MultiLineString
        from ..core.shapes2d import Shape
        v = self.ask("Grelha de régua", [("w", "Largura (mm)", 100.0), ("h", "Altura (mm)", 100.0), ("s", "Passo (mm)", 10.0, (0.1, 1000)),
                                         ("x", "Canto X (mm)", 0.0), ("y", "Canto Y (mm)", 0.0)])
        if not v:
            return
        lines = []
        for k in range(int(v["w"] / v["s"]) + 1):
            x = v["x"] + k * v["s"]; lines.append(LineString([(x, v["y"]), (x, v["y"] + v["h"])]))
        for k in range(int(v["h"] / v["s"]) + 1):
            y = v["y"] + k * v["s"]; lines.append(LineString([(v["x"], y), (v["x"] + v["w"], y)]))
        sh = Shape("geo", {}, MultiLineString(lines), "grelha de régua")
        sh.style.update(stroke="#3070c0", width=0.3)
        self.canvas.add_shapes([sh])
        self.tabs.setCurrentWidget(self.canvas)

    def d2_grid_spacing(self):
        """Espaçamento da grelha e ajuste à grelha."""
        c = self.canvas
        v = self.ask("Espaçamento da grelha", [("s", "Espaçamento (mm)", float(c.grid_step), (0.01, 10000)), ("snap", "Ajustar à grelha", bool(c.snap))])
        if v:
            c.grid_step = v["s"]; c.snap = v["snap"]; c.viewport().update()

    def d2_insert_image(self):
        """Imagem no desenho 2D: a imagem atual como fundo, à escala, para decalcar."""
        from PySide6.QtGui import QImage, QPixmap
        import numpy as np
        it = self.imgview.item
        if it is None:
            return self.error("Abra ou selecione primeiro uma imagem no separador Imagem / Metrologia.")
        a = np.asarray(it.data)
        if a.dtype != np.uint8:
            a = (255 * (a - a.min()) / max(float(a.max() - a.min()), 1e-9)).astype(np.uint8)
        pitch = float(it.meta.get("pitch", 0.1)) if hasattr(it, "meta") and it.meta else 0.1
        v = self.ask("Imagem no desenho", [("w", "Largura no desenho (mm)", round(a.shape[1] * pitch, 2)), ("op", "Opacidade (%)", 60.0, (5, 100))])
        if not v:
            return
        if a.ndim == 2:
            q = QImage(np.ascontiguousarray(a).data, a.shape[1], a.shape[0], a.shape[1], QImage.Format_Grayscale8).copy()
        else:
            b = np.ascontiguousarray(a[..., :3])
            q = QImage(b.data, b.shape[1], b.shape[0], 3 * b.shape[1], QImage.Format_RGB888).copy()
        item = self.canvas.scene().addPixmap(QPixmap.fromImage(q))
        k = v["w"] / a.shape[1]
        item.setScale(k)
        item.setTransform(item.transform().scale(1, -1))
        item.setPos(0, a.shape[0] * k)
        item.setOpacity(v["op"] / 100); item.setZValue(-10)
        self.tabs.setCurrentWidget(self.canvas)
        self.log(f"Imagem inserida no desenho 2D ({v['w']:.1f} mm de largura)")

    def d2_fan(self):
        v = self.ask("Leque", [("cx", "Centro X", 0.0), ("cy", "Centro Y", 0.0), ("r1", "Raio interior", 20.0, (0, 1e6)),
                               ("r2", "Raio exterior", 50.0, (0.01, 1e6)), ("a0", "Ângulo inicial (°)", 0.0), ("a1", "Ângulo final (°)", 90.0)])
        if v:
            self.canvas.add_shapes([s2.Shape("fan", v)]); self.tabs.setCurrentWidget(self.canvas)

    def d2_basic(self):
        v = self.ask("Forma básica", [("k", "Forma", "hexágono", gn.SHAPES_2D + ["retângulo arredondado"]), ("s", "Tamanho (mm)", 40.0),
                                      ("x", "Centro X", 0.0), ("y", "Centro Y", 0.0), ("r", "Raio dos cantos (retângulo)", 5.0),
                                      ("rot", "Rotação (°)", 0.0)])
        if not v:
            return
        from shapely import affinity
        g = gn.rounded_rect(v["s"] * 1.5, v["s"], v["r"]) if v["k"] == "retângulo arredondado" else gn.base_shape(v["k"], v["s"])
        g = affinity.translate(affinity.rotate(g, v["rot"], origin=(0, 0)), v["x"], v["y"])
        self.canvas.add_shapes([s2.Shape("geo", {}, g, v["k"])]); self.tabs.setCurrentWidget(self.canvas)

    def d2_curve(self):
        v = self.ask("Curva", [("k", "Tipo", "espiral", ["espiral", "onda", "dente de serra", "onda quadrada", "triangular"]),
                               ("x", "Início X", 0.0), ("y", "Início Y", 0.0), ("L", "Comprimento / diâmetro (mm)", 100.0),
                               ("a", "Amplitude (mm)", 10.0), ("p", "Período (mm)", 20.0), ("t", "Voltas (espiral)", 5.0, (0.1, 200))])
        if v:
            pts = s2.curve(v["k"], v["x"], v["y"], v["L"], v["a"], v["p"], v["t"])
            self.canvas.add_shapes([s2.Shape("pl", {"pts": pts}, name=v["k"])]); self.tabs.setCurrentWidget(self.canvas)

    def d2_text(self):
        from PySide6.QtGui import QFontDatabase
        fams = QFontDatabase.families()
        v = self.ask("Texto", [("t", "Texto", "Forja"), ("f", "Tipo de letra", "Helvetica" if "Helvetica" in fams else fams[0], fams),
                               ("s", "Altura (mm)", 10.0), ("b", "Negrito", False), ("x", "X", 0.0), ("y", "Y (linha de base)", 0.0),
                               ("a", "Rotação (°)", 0.0)])
        if v:
            self.canvas.add_shapes([s2.Shape("txt", dict(text=v["t"], font=v["f"], size=v["s"], bold=v["b"], x=v["x"], y=v["y"], angle=v["a"]))])
            self.tabs.setCurrentWidget(self.canvas)

    # --- estilo ------------------------------------------------------------
    def d2_set_style(self, **kw):
        sel = self._sel2d()
        if sel:
            self.canvas.push_undo()
            for s in sel:
                s.style.update(kw)
            self.canvas.redraw()

    def d2_color(self, key):
        sel = self._sel2d()
        if sel:
            c = QColorDialog.getColor(QColor(sel[0].style.get(key) or "#1e3c78"), self)
            if c.isValid():
                self.d2_set_style(**{key: c.name()})

    def d2_width(self):
        v = self.ask("Espessura", [("w", "Espessura (px no ecrã; ×0,1 mm no SVG)", 2.0, (0.1, 30))])
        if v: self.d2_set_style(width=v["w"])

    def d2_dash(self):
        v = self.ask("Tipo de linha", [("d", "Tipo", "tracejada", ["contínua", "tracejada", "pontilhada", "traço-ponto", "traço longo"])])
        if v: self.d2_set_style(dash=v["d"])

    def d2_arrows(self):
        v = self.ask("Setas", [("s", "Seta no início", False), ("e", "Seta no fim", True)])
        if v: self.d2_set_style(arrow_start=v["s"], arrow_end=v["e"])

    def d2_order(self, front):
        sel = self._sel2d()
        if sel:
            self.canvas.push_undo()
            ids = {s.id for s in sel}
            rest = [s for s in self.canvas.shapes if s.id not in ids]
            self.canvas.shapes = rest + sel if front else sel + rest
            self.canvas.redraw()

    # --- área de transferência / lista --------------------------------------
    def d2_copy(self):
        if self.tabs.currentWidget() is self.canvas:
            self._clip2d = [s.copy() for s in self.canvas.selected_shapes()]
            self.statusBar().showMessage(f"{len(self._clip2d)} forma(s) copiada(s)")
        elif self.tabs.currentWidget() is self.imgview and self.imgview.item is not None:
            from PySide6.QtWidgets import QApplication
            from .viewimage import array_to_qimage
            d = np.asarray(self.imgview.current_slice())
            rgb = d if d.dtype == np.uint8 else iop.colorize(d, self.imgview.cmap)[0]
            QApplication.clipboard().setImage(array_to_qimage(rgb))
            self.statusBar().showMessage("Imagem copiada para a área de transferência")

    def d2_paste(self):
        if self.tabs.currentWidget() is self.imgview:
            from PySide6.QtWidgets import QApplication
            img = QApplication.clipboard().image()
            if img.isNull():
                return self.error("A área de transferência não tem uma imagem.")
            img = img.convertToFormat(img.Format.Format_RGB888)
            a = np.frombuffer(img.constBits(), np.uint8, img.bytesPerLine() * img.height()).reshape(img.height(), img.bytesPerLine())
            self.add_image("colada", a[:, :img.width() * 3].reshape(img.height(), img.width(), 3).copy())
            return
        clip = getattr(self, "_clip2d", [])
        if clip:
            self.canvas.add_shapes([s2.move(s, 5, -5) if True else s for s in clip])
            for a, b in zip(self.canvas.selected_shapes(), clip):
                a.style = dict(b.style)
            self.canvas.redraw()

    def d2_list(self):
        d = QDialog(self); d.setWindowTitle("Formas"); d.resize(420, 480)
        lay = QVBoxLayout(d)
        lst = QListWidget(); lst.setSelectionMode(QListWidget.ExtendedSelection)
        for s in self.canvas.shapes:
            it = QListWidgetItem(s.name); it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(Qt.Unchecked if s.hidden else Qt.Checked); it.setData(Qt.UserRole, s.id)
            it.setSelected(s.id in self.canvas.selected)
            lst.addItem(it)
        lay.addWidget(lst)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        if d.exec() == QDialog.Accepted:
            byid = {s.id: s for s in self.canvas.shapes}
            sel = set()
            for i in range(lst.count()):
                it = lst.item(i); s = byid[it.data(Qt.UserRole)]
                s.name = it.text(); s.hidden = it.checkState() != Qt.Checked
                if it.isSelected():
                    sel.add(s.id)
            self.canvas.selected = sel
            self.canvas.redraw()

    def d2_rulers(self):
        self.canvas.show_rulers = not getattr(self.canvas, "show_rulers", True)
        self.canvas.viewport().update()

    # --- padrões / vetorização ---------------------------------------------
    def d2_pattern(self):
        sel = [s for s in self._sel2d() if s.closed]
        if not sel:
            return
        v = self.ask("Padrão de recorte", [("k", "Padrão", "hexágonos", d2.PATTERNS), ("c", "Tamanho da célula (mm)", 10.0),
                                           ("w", "Parede entre células (mm)", 2.0), ("b", "Moldura (mm)", 5.0), ("a", "Rotação (°)", 0.0)])
        if not v:
            return
        new = []
        for s in sel:
            try:
                g, n = d2.pattern_fill(s.geom, v["k"], v["c"], v["w"], v["b"], v["a"])
            except ValueError as e:
                return self.error(e)
            new.append(s2.Shape("geo", {}, g, f"{s.name} ({v['k']})"))
            self.log(f"{s.name}: {n} recortes, material reduzido para {g.area / s.geom.area * 100:.0f}%")
        self.canvas.replace_selected(new)

    def d2_halftone(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Meio-tom vetorial", [("w", "Largura (mm)", 100.0), ("c", "Célula (mm)", 1.5, (0.2, 50)), ("k", "Tipo", "pontos", ["pontos", "linhas"])])
        if v:
            g = iop.to_gray(it.data) if it.data.dtype == np.uint8 else iop.to_u8(it.data)
            geoms = d2.vector_halftone(g, v["w"], v["c"], v["k"])
            self.canvas.add_shapes([s2.Shape("geo", {}, geoms[0], f"meio-tom {it.name}")] if geoms else [])
            self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()

    def d2_photo_vector(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Fotografia → vetor", [("n", "Níveis", 4, (1, 16)), ("px", "Tamanho do pixel (mm)", 0.25), ("s", "Suavização (px)", 1.0, (0, 10)),
                                            ("simp", "Simplificação (px)", 0.8, (0, 10))])
        if not v:
            return
        g = iop.to_gray(it.data) if np.asarray(it.data).ndim == 3 else np.asarray(it.data, float)
        res = d2.photo_to_vector(g, v["n"], v["px"], simplify_px=v["simp"], smooth=v["s"])
        shapes = []
        for k, t, geom in res:
            sh = s2.Shape("geo", {}, geom, f"nível {k} (≤ {t:.0f})")
            gray = int(40 + 180 * (k - 1) / max(len(res) - 1, 1))
            sh.style["fill"] = f"#{gray:02x}{gray:02x}{gray:02x}"
            shapes.append(sh)
        self.canvas.add_shapes(shapes[::-1])
        self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()
        self.log(f"{len(shapes)} níveis vetorizados (cada nível numa forma; ideal para gravação em camadas)")

    def d2_export_eps(self):
        shapes = [s for s in (self.canvas.selected_shapes() or self.canvas.shapes) if not s.hidden]
        if not shapes:
            return self.error("Não há formas 2D.")
        p, _ = QFileDialog.getSaveFileName(self, "Exportar EPS", os.path.join(self.last_dir, "desenho.eps"), "EPS (*.eps)")
        if p:
            s2.export_eps(shapes, p); self.log(f"EPS exportado: {p}")
