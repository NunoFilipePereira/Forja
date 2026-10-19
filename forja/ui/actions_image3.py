"""Mais ações de imagem: máscaras, cor, campo vetorial, janelas FFT, interpolação, alinhamento elástico, etc."""
import os
import numpy as np
from PySide6.QtWidgets import QFileDialog
from ..core import imageops3 as io3, imageops as iop, imageops2 as io2


class Image3Actions:
    def menus_image3(self):
        return [("&Imagem", [
            None,
            ("Máscaras", [("Criar máscara por intervalo de valores…", self.i3_mask_range),
                          ("Preencher dentro/fora da máscara (A, máscara B)…", self.i3_fill_mask),
                          ("Usar imagem como máscara de qualidade da atual", self.i3_set_mask)]),
            ("Cor (mais)", [("Camada de cor…", self.i3_color_layer), ("Cor → intensidade…", self.i3_color_intensity),
                            ("Copiar cor de B para a intensidade de A", self.i3_copy_color)]),
            ("Mapa de vetores do gradiente…", self.i3_vector_map),
            ("Janela (Hann/Hamming…) antes da FFT…", self.i3_window),
            ("Redimensionar com interpolação…", self.i3_resize),
            ("Rodar com tela alargada…", self.i3_rotate),
            ("Alinhamento elástico (B → A)", self.i3_elastic),
            ("Histograma 2D (A, B)", self.i3_hist2d),
            ("Inserir imagem B dentro de A…", self.i3_insert),
            ("Projeções em X e Y (gráfico)", self.i3_projections),
            ("Caminho mínimo entre dois pontos…", self.i3_min_path),
            ("Limiar pelo método dos dois picos", self.i3_two_peak),
            ("Integrar ao longo de X/Y…", self.i3_integrate),
            ("Variância local…", self.i3_variance),
            ("Média ponderada (imagens, pesos = última)", self.i3_weighted),
            None,
            ("Lupa", self.i3_magnifier, "Ctrl+Shift+M"),
            ("Pré-visualizar várias imagens", self.i3_preview),
            ("Imagem seguinte", lambda: self.i3_nav(1), "Ctrl+PgDown"),
            ("Imagem anterior", lambda: self.i3_nav(-1), "Ctrl+PgUp"),
            ("Primeira imagem", lambda: self.i3_nav("first"), "Ctrl+Home"),
            ("Última imagem", lambda: self.i3_nav("last"), "Ctrl+End"),
            ("Renomear imagem…", self.i3_rename),
            ("Exportar como ICO/XPM…", self.i3_export_qt),
        ])]

    def i3_mask_range(self):
        it = self.cur_image()
        if not it:
            return
        a = np.asarray(it.data, float)
        v = self.ask("Máscara", [("lo", "Mínimo", float(np.nanpercentile(a, 25))), ("hi", "Máximo", float(np.nanpercentile(a, 100)))])
        if v: self.add_image(f"máscara({it.name})", io3.mask_from_range(it.data, v["lo"], v["hi"]))

    def i3_fill_mask(self):
        it = self.sel_images(2)
        if not it:
            return
        v = self.ask("Preencher", [("v", "Valor", 0.0), ("in", "Dentro da máscara (senão fora)", True)],
                     note=f"Imagem: {it[0].name}\nMáscara: {it[1].name}")
        if v: self.add_image(f"preenchida({it[0].name})", io3.fill_mask(it[0].data, it[1].data, v["v"], v["in"]))

    def i3_set_mask(self):
        it = self.sel_images(2)
        if it:
            it[0].meta["mask"] = np.asarray(it[1].data) > 0
            self.log(f"{it[1].name} passa a ser a máscara de {it[0].name} (usada no desembrulhamento e em Fase → altura).")

    def i3_color_layer(self):
        v = self.ask("Camada de cor", [("c", "Canal", "R", ["R", "G", "B"])])
        if v: self.img_op(f"camada {v['c']}", lambda a: io3.color_layer(a, v["c"]))

    def i3_color_intensity(self):
        v = self.ask("Cor → intensidade", [("m", "Modo", "luminância", ["luminância", "média", "máximo", "mínimo", "saturação"])])
        if v: self.img_op(v["m"], lambda a: io3.color_to_intensity(a, v["m"]))

    def i3_copy_color(self):
        it = self.sel_images(2)
        if it:
            try:
                self.add_image(f"cor de {it[1].name}", io3.copy_true_color(it[0].data, it[1].data))
            except ValueError as e:
                self.error(e)

    def i3_vector_map(self):
        v = self.ask("Mapa de vetores", [("s", "Espaçamento (px)", 16, (4, 200)), ("k", "Escala", 1.0, (0.1, 20))])
        if v: self.img_op("vetores", lambda a: io3.vector_map(a, v["s"], v["k"]))

    def i3_window(self):
        v = self.ask("Janela", [("k", "Tipo", "Hann", ["Hann", "Hamming", "Blackman", "cosseno"])])
        if v: self.img_op(f"janela {v['k']}", lambda a: io3.window(a, v["k"]))

    def i3_resize(self):
        v = self.ask("Redimensionar", [("f", "Fator", 2.0, (0.01, 50)), ("m", "Interpolação", "cúbica", list(io3.INTERP))])
        if v: self.img_op(f"×{v['f']:g}", lambda a: io3.resize_interp(a, v["f"], v["m"]))

    def i3_rotate(self):
        v = self.ask("Rodar", [("a", "Ângulo (°)", 30.0), ("m", "Interpolação", "linear", list(io3.INTERP)), ("e", "Alargar a tela", True)])
        if v: self.img_op(f"rodada {v['a']:g}°", lambda a: io3.rotate_expand(a, v["a"], v["m"], v["e"]))

    def i3_elastic(self):
        it = self.sel_images(2)
        if it:
            out, flow = io3.elastic_align(it[0].data, it[1].data)
            self.add_image(f"deslocamento elástico ({it[1].name})", np.hypot(flow[..., 0], flow[..., 1]), show=False)
            self.add_image(f"alinhada elástica({it[1].name})", out)

    def i3_hist2d(self):
        it = self.sel_images(2)
        if it:
            H, _ = io3.hist2d(it[0].data, it[1].data)
            self.add_image(f"histograma 2D ({it[0].name} × {it[1].name})", H)

    def i3_insert(self):
        it = self.sel_images(2)
        if not it:
            return
        v = self.ask("Inserir", [("x", "X", 0, (-100000, 100000)), ("y", "Y", 0, (-100000, 100000)), ("a", "Opacidade", 1.0, (0, 1))],
                     note=f"Destino: {it[0].name}\nA inserir: {it[1].name}")
        if v:
            try:
                self.add_image(f"inserida({it[0].name})", io3.insert(it[0].data, it[1].data, v["x"], v["y"], v["a"]))
            except ValueError as e:
                self.error(e)

    def i3_projections(self):
        it = self.cur_image()
        if it:
            px, py = io3.projections(it.data)
            self.chart.series += [{"x": np.arange(len(px)), "y": px, "label": f"projeção X ({it.name})", "kind": "linha"},
                                  {"x": np.arange(len(py)), "y": py, "label": f"projeção Y ({it.name})", "kind": "linha"}]
            self.chart.opts.update(xlabel="pixel", ylabel="soma")
            self._show_chart()

    def i3_min_path(self):
        it = self.cur_image()
        if not it:
            return
        h, w = np.asarray(it.data).shape[:2]
        v = self.ask("Caminho mínimo", [("x0", "X inicial", 0, (0, w - 1)), ("y0", "Y inicial", h // 2, (0, h - 1)),
                                        ("x1", "X final", w - 1, (0, w - 1)), ("y1", "Y final", h // 2, (0, h - 1)),
                                        ("d", "Seguir zonas escuras", True)])
        if v:
            path, L, c = io3.min_path(it.data, (v["x0"], v["y0"]), (v["x1"], v["y1"]), v["d"])
            import cv2
            out = io2.draw_shapes(it.data)
            cv2.polylines(out, [path.astype(np.int32)], False, (255, 220, 0), 1)
            self.add_image(f"caminho({it.name})", out)
            self.log(f"Caminho mínimo: {len(path)} pixels, comprimento {L:.1f} px, custo {c:.3g}")

    def i3_two_peak(self):
        it = self.cur_image()
        if it:
            try:
                b, t = io3.two_peak_threshold(it.data)
            except ValueError as e:
                return self.error(e)
            self.add_image(f"dois picos({it.name})", b)
            self.log(f"Limiar pelo método dos dois picos: {t:.4g}")

    def i3_integrate(self):
        v = self.ask("Integrar", [("ax", "Eixo", "X", ["X", "Y"]), ("s", "Passo (tamanho do pixel)", 1.0)])
        if v: self.img_op(f"integral {v['ax']}", lambda a: io3.integrate(a, v["ax"], v["s"]))

    def i3_variance(self):
        v = self.ask("Variância local", [("s", "Janela (px)", 7, (3, 101))])
        if v: self.img_op("variância", lambda a: io3.local_variance(a, v["s"]))

    def i3_weighted(self):
        it = self.sel_images(3)
        if it:
            try:
                self.add_image(f"média ponderada ({len(it) - 1})", io3.weighted_mean([i.data for i in it[:-1]], [it[-1].data] * (len(it) - 1)))
            except Exception as e:
                self.error(e)

    def i3_magnifier(self):
        on = self.imgview.toggle_magnifier()
        self.tabs.setCurrentWidget(self.imgview)
        self.statusBar().showMessage("Lupa ligada (passe o rato sobre a imagem; ⌘⇧M desliga)" if on else "Lupa desligada")

    def i3_preview(self):
        it = self.sel_images(2)
        if it:
            self.add_image(f"pré-visualização ({len(it)})", io3.preview_grid([i.data if not i.meta.get("volume") else i.data[len(i.data) // 2]
                                                                              for i in it], [i.name for i in it]))

    def i3_nav(self, step):
        n = len(self.images)
        if not n:
            return
        r = self.img_list.currentRow()
        r = 0 if step == "first" else n - 1 if step == "last" else max(0, min(n - 1, r + step))
        self.img_list.clearSelection(); self.img_list.setCurrentRow(r)
        self.tabs.setCurrentWidget(self.imgview)

    def i3_rename(self):
        r = self.img_list.currentRow()
        if r < 0:
            return
        from PySide6.QtWidgets import QInputDialog
        t, ok = QInputDialog.getText(self, "Renomear", "Nome:", text=self.images[r].name)
        if ok and t:
            self.images[r].name = t; self.img_list.item(r).setText(t); self.imgview.show_item(self.images[r])

    def i3_export_qt(self):
        it = self.cur_image()
        if not it:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Exportar", os.path.join(self.last_dir, it.name.split(".")[0] + ".ico"),
                                           "Ícone (*.ico);;XPM (*.xpm)")
        if p:
            from .viewimage import array_to_qimage
            d = np.asarray(it.data)
            rgb = d if d.dtype == np.uint8 else iop.colorize(d, self.imgview.cmap)[0]
            img = array_to_qimage(rgb)
            if p.lower().endswith(".ico") and max(img.width(), img.height()) > 256:
                from PySide6.QtCore import Qt
                img = img.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if img.save(p):
                self.log(f"Exportado {p}")
            else:
                self.error("Não foi possível guardar neste formato.")
