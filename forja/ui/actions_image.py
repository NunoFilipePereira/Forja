"""Ações de imagem adicionais e processamento em lote (mistura para a MainWindow)."""
import os
import numpy as np
from PySide6.QtWidgets import QFileDialog
from ..core import imageops as iop, imageops2 as io2
from .widgets import TextDialog

IMG_SAVE = "PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg);;BMP (*.bmp)"


class ImageActions:
    def menus_image(self):
        return [
            ("&Imagem", [
                None,
                ("Matemática por pixel", [(op, (lambda checked=False, op=op: self.act_pixel_math(op))) for op in
                                          ["+ valor", "− valor", "× valor", "÷ valor", "potência", "raiz quadrada", "ln", "exp",
                                           "seno", "cosseno", "valor absoluto", "negativo"]]
                 + [None, ("atan2(A, B)", self.act_atan2), ("Norma |A + jB|", self.act_cnorm)]),
                ("Intensidade", [("Gama…", self.act_gamma), ("Esticar contraste (percentis)…", self.act_stretch),
                                 ("Normalizar 0–1", lambda: self.img_op("normalizada", io2.normalize)),
                                 ("Transformação linear…", self.act_linear), ("Limitar extremos…", self.act_clamp),
                                 ("Substituir valor…", self.act_replace), ("Definir região…", self.act_set_region),
                                 ("Aplicar máscara…", self.act_mask), ("Adicionar ruído…", self.act_noise)]),
                ("Frequência", [("FFT", self.act_fft), ("FFT inversa (da imagem complexa)", self.act_ifft),
                                ("Filtro em frequência…", self.act_fft_filter), ("Hilbert em X", self.act_hilbert),
                                ("Filtro de Wiener…", self.act_wiener)]),
                ("Análise", [("Histograma", self.act_histogram), ("Perfil de linha…", self.act_profile),
                             ("Gradiente…", self.act_gradient), ("Transformada de distância", self.act_distance_tf),
                             ("Etiquetar objetos", self.act_labeling), ("Bacias hidrográficas (watershed)…", self.act_watershed),
                             ("Extremos locais…", self.act_extrema), ("Cantos de Harris", self.act_harris),
                             ("Arestas ISEF (Shen–Castan)…", self.act_isef), ("Linhas (Hough)…", self.act_lines),
                             ("Elipses…", self.act_ellipses), ("Forma envolvente…", self.act_bounding),
                             ("Procurar padrão (imagem B dentro de A)", self.act_template)]),
                ("Cor e geometria", [("Separar RGB", self.act_split_rgb), ("Juntar RGB (3 imagens)", self.act_merge_rgb),
                                     ("Virar", [("Horizontal", lambda: self.img_op("virada H", lambda a: io2.flip(a, "horizontal"))),
                                                ("Vertical", lambda: self.img_op("virada V", lambda a: io2.flip(a, "vertical")))]),
                                     ("Rodar 90° horário", lambda: self.img_op("rot90", lambda a: io2.rot90(a, True))),
                                     ("Rodar 90° anti-horário", lambda: self.img_op("rot-90", lambda a: io2.rot90(a, False))),
                                     ("Mapa polar", lambda: self.img_op("polar", io2.polar)),
                                     ("Polar → cartesiano", lambda: self.img_op("cartesiano", lambda a: io2.polar(a, True))),
                                     ("Transformação projetiva (matriz 3×3)…", self.act_projective),
                                     ("Alinhar B a A (pontos característicos)…", self.act_align),
                                     ("Meio-tom…", self.act_halftone), ("Mosaico…", self.act_mosaic),
                                     ("Transição entre duas fotos…", self.act_morph)]),
                ("Gerar", [("Mandelbrot…", self.act_mandelbrot), ("Imagem sintética…", self.act_synthetic),
                           ("Formas 2D → imagem (vetor → bitmap)…", self.act_rasterize)]),
            ]),
            ("&Lote", [
                ("Carregar pasta de imagens…", self.act_load_folder, "Ctrl+L"),
                ("Importar vídeo (frames)…", self.act_import_video),
                ("Exportar seleção como vídeo…", self.act_export_video),
                ("Exportar seleção para pasta…", self.act_export_batch),
                None,
                ("Estatística do lote", [(k, (lambda checked=False, k=k: self.act_batch_stat(k))) for k in
                                         ["média", "mediana", "mínimo", "máximo", "desvio-padrão", "soma", "amplitude (máx−mín)"]]),
                ("Média progressiva", self.act_progressive),
                ("Lote 1 (op) Lote 2…", self.act_batch_pair),
                ("Cada imagem (op) imagem fixa…", self.act_batch_vs_img),
                None,
                ("Juntar lado a lado", lambda: self._batch_join("horizontal")),
                ("Juntar em coluna", lambda: self._batch_join("vertical")),
                ("Colagem…", self.act_collage),
                ("Panorama / costura (stitch)…", self.act_stitch),
                None,
                ("Aplicar a operação seguinte a todas as selecionadas", self.act_batch_hint),
            ]),
        ]

    # ------------------------------------------------------------------
    def _imgs(self, n=1):
        return self.sel_images(n)

    def act_pixel_math(self, op):
        v = {"value": 1.0}
        if op in ("+ valor", "− valor", "× valor", "÷ valor", "potência"):
            v = self.ask(op, [("value", "Valor", 2.0 if op != "+ valor" else 10.0)])
            if not v:
                return
        self.img_op(op, lambda a: io2.pixel_math(a, op, v["value"]))

    def _pair(self, label, fn):
        it = self._imgs(2)
        if it:
            self.add_image(f"{label}({it[0].name}, {it[1].name})", fn(it[0].data, it[1].data))

    def act_atan2(self): self._pair("atan2", io2.atan2_img)
    def act_cnorm(self): self._pair("norma", io2.complex_norm)

    def act_gamma(self):
        v = self.ask("Gama", [("g", "Gama (<1 clareia, >1 escurece)", 0.6, (0.01, 10))])
        if v: self.img_op(f"gama {v['g']:g}", lambda a: io2.gamma(a, v["g"]))

    def act_stretch(self):
        v = self.ask("Esticar contraste", [("lo", "Percentil baixo", 1.0, (0, 100)), ("hi", "Percentil alto", 99.0, (0, 100))])
        if v: self.img_op("esticada", lambda a: io2.stretch(a, v["lo"], v["hi"]))

    def act_linear(self):
        v = self.ask("Transformação linear", [("g", "Ganho", 1.0), ("o", "Desvio", 0.0)])
        if v: self.img_op("linear", lambda a: io2.linear(a, v["g"], v["o"]))

    def act_clamp(self):
        it = self.cur_image()
        if not it:
            return
        a = np.asarray(it.data, float)
        v = self.ask("Limitar extremos", [("lo", "Mínimo", float(np.nanpercentile(a, 1))), ("hi", "Máximo", float(np.nanpercentile(a, 99)))])
        if v: self.img_op("limitada", lambda a: io2.clamp(a, v["lo"], v["hi"]))

    def act_replace(self):
        v = self.ask("Substituir valor", [("old", "Valor antigo", 0.0), ("new", "Valor novo", float("nan") if False else 255.0),
                                          ("tol", "Tolerância", 0.0)])
        if v: self.img_op("substituída", lambda a: io2.replace_value(a, v["old"], v["new"], v["tol"]))

    def act_set_region(self):
        it = self.cur_image()
        if not it:
            return
        h, w = it.data.shape[:2]
        v = self.ask("Definir região", [("x", "X", 0, (0, w)), ("y", "Y", 0, (0, h)), ("w", "Largura", w // 4, (1, w)),
                                        ("h", "Altura", h // 4, (1, h)), ("val", "Valor", 0.0)])
        if v: self.img_op("região", lambda a: io2.set_region(a, v["x"], v["y"], v["w"], v["h"], v["val"]))

    def act_mask(self):
        it = self._imgs(2)
        if it:
            v = self.ask("Máscara", [("o", "Valor fora da máscara", 0.0)],
                         note=f"Imagem: {it[0].name}\nMáscara (branco = manter): {it[1].name}")
            if v:
                self.add_image(f"mascarada({it[0].name})", io2.apply_mask(it[0].data, it[1].data, v["o"]))

    def act_noise(self):
        v = self.ask("Ruído", [("k", "Tipo", "gaussiano", ["gaussiano", "sal e pimenta", "speckle"]), ("a", "Intensidade", 10.0)])
        if v: self.img_op("ruído", lambda a: io2.add_noise(a, v["k"], v["a"]))

    def act_fft(self):
        it = self.cur_image()
        if it:
            r = io2.fft(it.data)
            for k in ("real", "imaginária", "fase"):
                self.add_image(f"FFT {k}({it.name})", r[k], show=False)
            self.add_image(f"FFT potência({it.name})", r["potência (log)"], {"complex": r["_complex"]})

    def act_ifft(self):
        it = self.cur_image()
        if not it:
            return
        F = it.meta.get("complex")
        if F is None:
            its = self.sel_images(2)
            if not its:
                return self.error("Selecione uma imagem 'FFT potência' ou as imagens FFT real e imaginária.")
            F = np.asarray(its[0].data, float) + 1j * np.asarray(its[1].data, float)
        r = io2.ifft(F)
        for k in ("imaginária", "fase", "módulo"):
            self.add_image(f"IFFT {k}", r[k], show=False)
        self.add_image("IFFT real", r["real"])

    def act_fft_filter(self):
        v = self.ask("Filtro em frequência", [("k", "Tipo", "passa-baixo", ["passa-baixo", "passa-alto", "passa-banda", "rejeita-banda"]),
                                              ("r1", "Raio 1 (px na FFT)", 20.0), ("r2", "Raio 2 (banda)", 60.0)])
        if v: self.img_op(v["k"], lambda a: io2.fft_filter(a, v["k"], v["r1"], v["r2"]))

    def act_hilbert(self):
        it = self.cur_image()
        if it:
            r = io2.hilbert_x(it.data)
            for k in ("real", "imaginária", "potência"):
                self.add_image(f"Hilbert {k}({it.name})", r[k], show=False)
            self.add_image(f"Hilbert fase({it.name})", r["fase"])

    def act_wiener(self):
        v = self.ask("Wiener", [("s", "Janela (px)", 5, (3, 51))])
        if v: self.img_op("wiener", lambda a: io2.wiener(a, v["s"]))

    def act_histogram(self):
        it = self.cur_image()
        if it:
            self.add_image(f"histograma({it.name})", io2.histogram_plot(it.data))
            a = np.asarray(it.data, float)
            self.log(f"Histograma {it.name}: mín {np.nanmin(a):.4g}, máx {np.nanmax(a):.4g}, média {np.nanmean(a):.4g}, "
                     f"desvio {np.nanstd(a):.4g}")

    def act_profile(self):
        it = self.cur_image()
        if not it:
            return
        h, w = it.data.shape[:2]
        v = self.ask("Perfil", [("x1", "X1", 0, (0, w - 1)), ("y1", "Y1", h // 2, (0, h - 1)), ("x2", "X2", w - 1, (0, w - 1)),
                                ("y2", "Y2", h // 2, (0, h - 1))])
        if not v:
            return
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p = io2.profile(it.data, v["x1"], v["y1"], v["x2"], v["y2"])
        fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=100)
        ax.plot(p, lw=1); ax.set_xlabel("distância (px)"); ax.set_ylabel("valor"); ax.grid(alpha=.3); fig.tight_layout()
        fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
        self.add_image(f"perfil({it.name})", img, {"profile": p})
        self.log("Perfil: " + " ".join(f"{x:.4g}" for x in p[:: max(1, len(p) // 40)]))

    def act_gradient(self):
        v = self.ask("Gradiente", [("k", "Componente", "magnitude", ["magnitude", "X", "Y", "direção"]), ("d", "Distância (px)", 1, (1, 100))])
        if v: self.img_op(f"gradiente {v['k']}", lambda a: io2.gradient(a, v["k"], v["d"]))

    def _binary(self, a):
        a = np.asarray(a)
        return a if (a.dtype == np.uint8 and set(np.unique(a)) <= {0, 255}) else iop.threshold(a, "otsu")

    def act_distance_tf(self):
        self.img_op("distância", lambda a: io2.distance_transform(self._binary(a)))

    def act_labeling(self):
        it = self.cur_image()
        if it:
            lab, n = io2.labeling(self._binary(it.data))
            st = io2.label_stats(lab)
            self.add_image(f"etiquetas({it.name})", lab.astype(float))
            self.log(f"{n} objetos: " + "; ".join(f"#{s['id']} área {s['área']} centro ({s['centro'][0]:.1f}, {s['centro'][1]:.1f})" for s in st[:100]))

    def act_watershed(self):
        v = self.ask("Watershed", [("d", "Distância mínima entre centros (px)", 10, (1, 500))])
        if v: self.img_op("watershed", lambda a: io2.watershed(self._binary(a), v["d"]).astype(float))

    def act_extrema(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Extremos locais", [("s", "Janela (px)", 15, (3, 501)), ("k", "Tipo", "máximos", ["máximos", "mínimos"])])
        if v:
            pts = io2.local_extrema(it.data, v["s"], v["k"])
            if len(pts) > 5000:
                pts = pts[:5000]
            self.add_image(f"{v['k']}({it.name})", io2.draw_shapes(it.data, pts=pts))
            self.log(f"{len(pts)} {v['k']} locais")

    def act_harris(self):
        it = self.cur_image()
        if it:
            R, pts = io2.harris(it.data)
            self.add_image(f"harris({it.name})", io2.draw_shapes(it.data, pts=pts[:3000]))
            self.log(f"Harris: {len(pts)} pontos")

    def act_isef(self):
        v = self.ask("ISEF", [("b", "Intensidade do filtro (0,01–0,99)", 0.6, (0.01, 0.99))])
        if v: self.img_op("ISEF", lambda a: io2.isef_edges(a, v["b"]))

    def act_lines(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Linhas", [("t", "Votos mínimos", 80, (1, 10000)), ("l", "Comprimento mínimo (px)", 30, (1, 10000)),
                                ("g", "Intervalo máximo (px)", 10, (0, 1000))])
        if v:
            L = io2.hough_lines(it.data, v["t"], v["l"], v["g"])
            self.add_image(f"linhas({it.name})", io2.draw_shapes(it.data, lines=L))
            self.log(f"{len(L)} linhas: " + "; ".join(f"({a},{b})→({c},{d})" for a, b, c, d in L[:40]))

    def act_ellipses(self):
        it = self.cur_image()
        if not it:
            return
        E = io2.fit_ellipses(self._binary(it.data))
        self.add_image(f"elipses({it.name})", io2.draw_shapes(it.data, ellipses=E))
        self.log(f"{len(E)} elipses: " + "; ".join(f"centro ({e[0][0]:.1f},{e[0][1]:.1f}) eixos {e[1][0]:.1f}×{e[1][1]:.1f} @ {e[2]:.1f}°" for e in E[:40]))

    def act_bounding(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Forma envolvente", [("k", "Tipo", "caixa mínima", ["caixa mínima", "caixa alinhada", "círculo mínimo", "invólucro convexo"])])
        if not v:
            return
        kind, geo, extra = io2.bounding_shapes(self._binary(it.data), v["k"])
        if kind == "circle":
            self.add_image(f"envolvente({it.name})", io2.draw_shapes(it.data, circles=[geo]))
            self.log(f"Círculo mínimo: centro ({geo[0]:.2f}, {geo[1]:.2f}), raio {geo[2]:.2f} px")
        else:
            self.add_image(f"envolvente({it.name})", io2.draw_shapes(it.data, polys=[geo]))
            if v["k"] == "caixa mínima":
                (cx, cy), (w, h), a = extra
                self.log(f"Caixa mínima: centro ({cx:.2f}, {cy:.2f}), {w:.2f} × {h:.2f} px, ângulo {a:.2f}°")
            else:
                self.log(f"{v['k']}: {len(geo)} vértices")

    def act_template(self):
        it = self._imgs(2)
        if it:
            R, (x, y), score = io2.template_match(it[0].data, it[1].data)
            h, w = np.asarray(it[1].data).shape[:2]
            poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            self.add_image(f"encontrado({it[0].name})", io2.draw_shapes(it[0].data, polys=[poly]))
            self.add_image("mapa de correlação", R, show=False)
            self.log(f"Padrão encontrado em ({x}, {y}) com correlação {score:.3f}")

    def act_split_rgb(self):
        it = self.cur_image()
        if it:
            try:
                for c, a in zip("RGB", io2.split_rgb(it.data)):
                    self.add_image(f"{c}({it.name})", a)
            except ValueError as e:
                self.error(e)

    def act_merge_rgb(self):
        it = self._imgs(3)
        if it:
            self.add_image("RGB", io2.merge_rgb(it[0].data, it[1].data, it[2].data))

    def act_projective(self):
        v = self.ask("Matriz projetiva", [("m", "Matriz 3×3 (linhas separadas por ;)", "1 0 0; 0 1 0; 0 0 1")])
        if v:
            try:
                M = np.array([[float(x) for x in r.split()] for r in v["m"].split(";")])
                assert M.shape == (3, 3)
            except Exception:
                return self.error("Matriz inválida.")
            self.img_op("projetiva", lambda a: io2.projective(a, M))

    def act_align(self):
        it = self._imgs(2)
        if not it:
            return
        v = self.ask("Alinhar", [("m", "Modelo", "projetivo", ["projetivo", "semelhança"])],
                     note=f"Referência: {it[0].name}\nA alinhar: {it[1].name}")
        if v:
            out, M = io2.align_images(it[0].data, it[1].data, v["m"])
            self.add_image(f"alinhada({it[1].name})", out)
            self.log(f"Matriz de alinhamento:\n{np.round(M, 5)}")

    def act_halftone(self):
        v = self.ask("Meio-tom", [("k", "Tipo", "pontos", ["pontos", "linhas"]), ("c", "Célula (px)", 8, (2, 100))])
        if v: self.img_op("meio-tom", lambda a: io2.halftone(a, v["k"], v["c"]))

    def act_mosaic(self):
        v = self.ask("Mosaico", [("b", "Bloco (px)", 12, (2, 500))])
        if v: self.img_op("mosaico", lambda a: io2.mosaic(a, v["b"]))

    def act_morph(self):
        it = self._imgs(2)
        if not it:
            return
        v = self.ask("Transição", [("n", "Número de passos", 8, (2, 100))])
        if v:
            for k, t in enumerate(np.linspace(0, 1, v["n"])):
                self.add_image(f"transição {k + 1}/{v['n']}", io2.morph_photos(it[0].data, it[1].data, t), show=k == v["n"] - 1)

    def act_mandelbrot(self):
        v = self.ask("Mandelbrot", [("w", "Largura", 800, (16, 8000)), ("h", "Altura", 600, (16, 8000)), ("cx", "Centro X", -0.6),
                                    ("cy", "Centro Y", 0.0), ("z", "Zoom", 1.0, (0.01, 1e12)), ("i", "Iterações", 200, (10, 5000))])
        if v:
            self.task("A calcular…", lambda: io2.mandelbrot(v["w"], v["h"], v["cx"], v["cy"], v["z"], v["i"]),
                      lambda a: self.add_image("Mandelbrot", a))

    def act_synthetic(self):
        v = self.ask("Imagem sintética", [("k", "Tipo", "anéis", ["gaussiana", "rampa", "xadrez", "dente de serra", "anéis", "aleatória"]),
                                          ("w", "Largura", 512, (2, 10000)), ("h", "Altura", 512, (2, 10000)), ("p", "Período (px)", 32.0)])
        if v:
            self.add_image(v["k"], io2.synthetic(v["k"], v["w"], v["h"], v["p"]))

    def act_rasterize(self):
        sh = self.canvas.selected_shapes() or self.canvas.shapes
        if not sh:
            return self.error("Não há formas 2D.")
        v = self.ask("Vetor → bitmap", [("px", "Tamanho do pixel (mm)", 0.1, (0.001, 100))])
        if v:
            try:
                self.add_image("formas rasterizadas", io2.rasterize_shapes(sh, v["px"]))
            except ValueError as e:
                self.error(e)

    # --- lotes ----------------------------------------------------------
    def act_load_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta de imagens", self.last_dir)
        if not d:
            return
        self.last_dir = d
        self.task("A carregar pasta…", lambda: io2.load_folder(d),
                  lambda items: [self.add_image(n, a, show=i == len(items) - 1) for i, (n, a) in enumerate(items)]
                  or self.log(f"{len(items)} imagens carregadas de {d}"))

    def act_import_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Vídeo", self.last_dir, "Vídeo (*.avi *.mp4 *.mov *.mkv);;Todos (*)")
        if not p:
            return
        v = self.ask("Vídeo", [("s", "Usar 1 frame em cada", 1, (1, 1000)), ("m", "Máximo de frames", 200, (1, 5000))])
        if v:
            base = os.path.splitext(os.path.basename(p))[0]
            self.task("A importar vídeo…", lambda: io2.load_video(p, v["s"], v["m"]),
                      lambda fr: [self.add_image(f"{base} {i + 1:04d}", a, show=i == len(fr) - 1) for i, a in enumerate(fr)])

    def act_export_video(self):
        it = self._imgs(2)
        if not it:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Guardar vídeo", os.path.join(self.last_dir, "video.avi"), "AVI (*.avi);;MP4 (*.mp4)")
        if not p:
            return
        v = self.ask("Vídeo", [("f", "Imagens por segundo", 10.0, (0.1, 240))])
        if v:
            io2.save_video(p, [i.data if i.data.dtype == np.uint8 else iop.colorize(i.data, self.imgview.cmap)[0] for i in it], v["f"])
            self.log(f"Vídeo guardado: {p} ({len(it)} frames)")

    def act_export_batch(self):
        it = self._imgs(1)
        if not it:
            return
        d = QFileDialog.getExistingDirectory(self, "Pasta de destino", self.last_dir)
        if not d:
            return
        v = self.ask("Exportar lote", [("f", "Formato", "png", ["png", "tif", "jpg", "bmp", "f32", "txt"]), ("p", "Prefixo", "img_")])
        if not v:
            return
        for k, i in enumerate(it):
            path = os.path.join(d, f"{v['p']}{k + 1:04d}.{v['f']}")
            if v["f"] == "txt":
                io2.save_ascii(path, i.data if i.data.ndim == 2 else iop.to_gray(i.data))
            elif v["f"] == "f32":
                np.asarray(i.data, np.float32).tofile(path)
            else:
                d8 = i.data if i.data.dtype == np.uint8 else iop.colorize(i.data, self.imgview.cmap)[0]
                iop.save(path, d8)
        self.log(f"{len(it)} imagens exportadas para {d}")

    def act_batch_stat(self, kind):
        it = self._imgs(2)
        if it:
            try:
                self.add_image(f"{kind} do lote ({len(it)})", io2.batch_stat([i.data for i in it], kind))
            except ValueError as e:
                self.error(e)

    def act_progressive(self):
        it = self._imgs(2)
        if it:
            for k, a in enumerate(io2.progressive_mean([i.data for i in it])):
                self.add_image(f"média progressiva {k + 1}", a, show=k == len(it) - 1)

    def act_batch_pair(self):
        it = self._imgs(2)
        if not it or len(it) % 2:
            return self.error("Selecione um número par de imagens: a 1.ª metade é o lote 1 e a 2.ª metade o lote 2.")
        v = self.ask("Lote 1 (op) Lote 2", [("op", "Operação", "subtrair", ["somar", "subtrair", "diferença abs.", "multiplicar",
                                                                             "dividir", "média", "máximo", "mínimo"])])
        if v:
            n = len(it) // 2
            for k, r in enumerate(io2.batch_pairwise([i.data for i in it[:n]], [i.data for i in it[n:]], v["op"])):
                self.add_image(f"{v['op']} {k + 1}", r, show=k == n - 1)

    def act_batch_vs_img(self):
        it = self._imgs(2)
        if not it:
            return
        v = self.ask("Cada imagem (op) imagem fixa", [("op", "Operação", "subtrair", ["somar", "subtrair", "diferença abs.", "multiplicar",
                                                                                       "dividir", "média", "máximo", "mínimo"])],
                     note=f"Imagem fixa: a última selecionada ({it[-1].name})")
        if v:
            for k, i in enumerate(it[:-1]):
                self.add_image(f"{v['op']}({i.name})", iop.arithmetic(i.data, it[-1].data, v["op"]), show=k == len(it) - 2)

    def _batch_join(self, d):
        it = self._imgs(2)
        if it:
            self.add_image(f"junção {d}", io2.join([i.data for i in it], d))

    def act_collage(self):
        it = self._imgs(2)
        if it:
            v = self.ask("Colagem", [("c", "Colunas (0 = automático)", 0, (0, 100)), ("g", "Espaço (px)", 4, (0, 200))])
            if v:
                self.add_image("colagem", io2.collage([i.data for i in it], v["c"], v["g"]))

    def act_stitch(self):
        it = self._imgs(2)
        if not it:
            return
        v = self.ask("Costura", [("m", "Modo", "panorama", ["panorama", "digitalização (planar)"])])
        if v:
            self.task("A costurar…", lambda: io2.stitch([i.data for i in it], "panorama" if v["m"] == "panorama" else "scans"),
                      lambda a: self.add_image("panorama", a))

    def act_batch_hint(self):
        TextDialog.show_text(self, "Lotes", "Selecione várias imagens na lista (Shift/⌘+clique) e escolha qualquer operação "
                             "do menu Imagem: é aplicada a cada uma das imagens selecionadas.\n\nA ordem da lista (arrastável) "
                             "é a ordem do lote.")
