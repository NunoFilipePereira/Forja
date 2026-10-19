"""Metrologia avançada: DIC, shearografia, holografia, desembrulhamento guiado, estéreo, linha laser, nuvens."""
import os
import numpy as np
from PySide6.QtWidgets import QFileDialog
from ..core import metro2 as m2, fringe as fr, meshops as mo
from .widgets import TextDialog


class MetroActions:
    def menus_metro(self):
        return [("Metro&logia", [
            None,
            ("Fase (avançado)", [
                ("Subtrair fases (A − B, embrulhada)", self.mt_sub_phase),
                ("Filtrar fase embrulhada (seno/cosseno)…", self.mt_filter_wrapped),
                ("Filtrar fase desembrulhada…", self.mt_filter_unwrapped),
                ("Média de várias fases", self.mt_avg_phase),
                ("Mapa de variância do gradiente (PGV)", self.mt_pgv),
                ("Desembrulhar guiado por PGV", self.mt_unwrap_pgv),
                ("Desembrulhar guiado por mapa do utilizador (A = fase, B = qualidade)", self.mt_unwrap_user),
            ]),
            ("Correlação digital de imagem (DIC)", [
                ("Deslocamentos por subconjuntos (antes, depois)…", self.mt_dic),
                ("Deslocamentos densos (fluxo ótico)", self.mt_dic_dense),
                ("Deformações a partir de DispX e DispY…", self.mt_strain),
                ("Gerar padrão de speckle e deformação de teste…", self.mt_speckle_demo),
            ]),
            ("Interferometria / shearografia", [
                ("Fase → deslocamento fora do plano…", lambda: self.mt_phase_conv("out")),
                ("Fase → deslocamento no plano…", lambda: self.mt_phase_conv("in")),
                ("Fase → gradiente de deformação (shearografia)…", lambda: self.mt_phase_conv("grad")),
            ]),
            ("Holografia digital", [
                ("Reconstruir holograma…", self.mt_holo),
                ("Gerar holograma de teste", self.mt_holo_demo),
            ]),
            ("Triangulação por linha laser", [
                ("Perfil da linha numa imagem", self.mt_laser_profile),
                ("Varrimento: sequência de imagens → mapa de altura…", self.mt_line_scan),
                ("Gerar sequência de teste", self.mt_laser_demo),
            ]),
            ("Estéreo (2 câmaras)", [
                ("Calibrar par estéreo (metade esquerda, metade direita)…", self.mt_stereo_cal),
                ("Profundidade a partir de um par calibrado (esq., dir.)…", self.mt_stereo_depth),
                ("Disparidade sem calibração (esq., dir.)…", self.mt_disparity),
                ("Gerar imagens de calibração de teste", self.mt_stereo_demo),
            ]),
            ("Nuvens de pontos", [
                ("Importar nuvem (TXT/XYZ/PLY)…", self.mt_import_cloud),
                ("Exportar objeto como nuvem TXT (pontos e normais)…", self.mt_export_cloud),
                ("Triangulação de Delaunay → malha…", self.mt_delaunay),
                ("Alinhar objeto A a B (ICP)", self.mt_icp),
                ("Juntar vistas de mesa rotativa (imagens de altura)…", self.mt_turntable),
            ]),
        ])]

    def _two(self):
        return self.sel_images(2)

    def _arr(self, it):
        a = np.asarray(it.data, float)
        return a.mean(2) if a.ndim == 3 else a

    # --- fase ---------------------------------------------------------
    def mt_sub_phase(self):
        it = self._two()
        if it: self.add_image(f"Δfase({it[0].name} − {it[1].name})", m2.subtract_phase(self._arr(it[0]), self._arr(it[1])))

    def mt_filter_wrapped(self):
        v = self.ask("Filtrar fase embrulhada", [("s", "Janela (px)", 5, (3, 51)), ("n", "Iterações", 2, (1, 50))])
        if v: self.img_op("fase filtrada", lambda a: m2.filter_wrapped(np.asarray(a, float), v["s"], v["n"]))

    def mt_filter_unwrapped(self):
        v = self.ask("Filtrar fase desembrulhada", [("s", "Sigma (px)", 2.0, (0.1, 50))])
        if v:
            from scipy import ndimage
            self.img_op("fase suavizada", lambda a: ndimage.gaussian_filter(np.nan_to_num(np.asarray(a, float)), v["s"]))

    def mt_avg_phase(self):
        it = self._two()
        if it: self.add_image(f"média de {len(it)} fases", m2.average_phase([self._arr(i) for i in it]))

    def mt_pgv(self):
        self.img_op("PGV", lambda a: m2.phase_gradient_variance(np.asarray(a, float)))

    def mt_unwrap_pgv(self):
        it = self.cur_image()
        if not it:
            return
        phi = self._arr(it); mask = it.meta.get("mask")
        q = m2.second_difference_quality(phi) / (1 + 5 * m2.phase_gradient_variance(phi))
        self.task("A desembrulhar…", lambda: m2.unwrap_guided(phi, q, mask),
                  lambda U: self.add_image(f"desembrulhada PGV({it.name})", U, {"mask": mask}))

    def mt_unwrap_user(self):
        it = self._two()
        if it:
            phi, q = self._arr(it[0]), self._arr(it[1])
            self.task("A desembrulhar…", lambda: m2.unwrap_guided(phi, q),
                      lambda U: self.add_image(f"desembrulhada guiada({it[0].name})", U))

    # --- DIC ----------------------------------------------------------
    def mt_dic(self):
        it = self._two()
        if not it:
            return
        v = self.ask("DIC", [("sub", "Subconjunto (px, ímpar)", 31, (7, 201)), ("step", "Passo da grelha (px)", 10, (1, 200)),
                             ("search", "Raio de procura (px)", 20, (1, 500)), ("c", "Correlação mínima", 0.6, (0, 1)),
                             ("px", "Tamanho do pixel (mm, 0 = px)", 0.0)],
                     note=f"Antes: {it[0].name}\nDepois: {it[1].name}")
        if not v:
            return
        A, B = it[0].data, it[1].data

        def done(r):
            DX, DY, C, xs, ys = r
            k = v["px"] or 1.0
            u = "mm" if v["px"] else "px"
            self.add_image(f"DispX ({u})", DX * k, {"grid_step": v["step"], "unit": u}, show=False)
            self.add_image(f"DispY ({u})", DY * k, {"grid_step": v["step"], "unit": u}, show=False)
            self.add_image("coeficiente de correlação", C, show=False)
            self.add_image("vetores de deslocamento", m2.vector_plot(A, DX, DY, xs, ys))
            self.log(f"DIC: {np.isfinite(DX).sum()} pontos; deslocamento médio ({np.nanmean(DX) * k:.4g}, {np.nanmean(DY) * k:.4g}) {u}; "
                     f"correlação média {np.nanmean(C):.3f}")
        self.task("A correlacionar…", lambda: m2.dic(A, B, v["sub"] | 1, v["step"], v["search"], v["c"]), done)

    def mt_dic_dense(self):
        it = self._two()
        if it:
            u, w = m2.dic_dense(it[0].data, it[1].data)
            self.add_image("DispX denso (px)", u, {"grid_step": 1}, show=False)
            self.add_image("DispY denso (px)", w, {"grid_step": 1})

    def mt_strain(self):
        it = self._two()
        if not it:
            return
        step = it[0].meta.get("grid_step", 1)
        v = self.ask("Deformações", [("step", "Distância entre pontos (mesma unidade dos deslocamentos)", float(step)),
                                     ("s", "Suavização", 1.0, (0, 20))], note=f"DispX: {it[0].name}\nDispY: {it[1].name}\nUnidade: mili-deformação")
        if v:
            S = m2.strains(self._arr(it[0]), self._arr(it[1]), v["step"], v["s"])
            for k, a in S.items():
                self.add_image(f"deformação {k} (‰)", a, show=k == "von Mises")
            self.log("Deformação média (‰): " + ", ".join(f"{k} {np.nanmean(a):.4g}" for k, a in S.items()))

    def mt_speckle_demo(self):
        v = self.ask("Speckle de teste", [("e", "Deformação exx", 0.01), ("t", "Translação X (px)", 2.0)])
        if v:
            A = m2.speckle_pattern()
            self.add_image("speckle antes", A, show=False)
            self.add_image("speckle depois", m2.warp_affine_field(A, exx=v["e"], tx=v["t"]))
            self.log("Selecione 'speckle antes' e 'speckle depois' e use DIC ▸ Deslocamentos por subconjuntos.")

    def mt_phase_conv(self, kind):
        F = {"out": [("l", "Comprimento de onda (nm)", 532.0), ("a", "Ângulo de iluminação (°)", 0.0)],
             "in": [("l", "Comprimento de onda (nm)", 532.0), ("a", "Ângulo de iluminação (°)", 30.0)],
             "grad": [("l", "Comprimento de onda (nm)", 532.0), ("s", "Distância de corte/shear (mm)", 5.0)]}[kind]
        v = self.ask("Fase → deformação", F, note="Use uma diferença de fase já desembrulhada.")
        if not v:
            return
        if kind == "out":
            self.img_op("w (nm)", lambda a: m2.phase_to_out_of_plane(np.asarray(a, float), v["l"], v["a"]))
        elif kind == "in":
            self.img_op("u (nm)", lambda a: m2.phase_to_in_plane(np.asarray(a, float), v["l"], v["a"]))
        else:
            self.img_op("∂w/∂x (µm/m)", lambda a: m2.phase_to_gradient(np.asarray(a, float), v["l"], v["s"]))

    # --- holografia ---------------------------------------------------
    def mt_holo(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Holografia", [("l", "Comprimento de onda (nm)", 632.8), ("p", "Tamanho do pixel (µm)", 3.45),
                                    ("z", "Distância de reconstrução (mm, negativa = imagem virtual)", -100.0),
                                    ("m", "Método", "espectro angular", ["espectro angular", "Fresnel (campo próximo)", "Fraunhofer (campo distante)"])])
        if v:
            d = it.data
            def done(r):
                amp, ph = r
                self.add_image(f"fase holo({it.name})", ph, show=False)
                self.add_image(f"amplitude holo z={v['z']:g}({it.name})", amp)
            self.task("A reconstruir…", lambda: m2.holo_reconstruct(d, v["l"], v["p"], v["z"], v["m"]), done)

    def mt_holo_demo(self):
        import cv2
        obj = np.zeros((256, 256), np.uint8)
        cv2.putText(obj, "F", (80, 175), cv2.FONT_HERSHEY_SIMPLEX, 4, 1, 12)
        self.add_image("holograma de teste (z = 100 mm)", m2.simulate_hologram(obj.astype(float)))
        self.log("Reconstrua com distância −100 mm, 632,8 nm e pixel de 3,45 µm.")

    # --- linha laser --------------------------------------------------
    def mt_laser_profile(self):
        it = self.cur_image()
        if it:
            p = m2.laser_line_profile(it.data)
            from ..core import imageops2 as io2
            pts = [(x, y) for x, y in enumerate(p) if np.isfinite(y)]
            self.add_image(f"linha({it.name})", io2.draw_shapes(it.data, pts=pts[::4]))
            self.log(f"Linha laser: {len(pts)} colunas; linha média {np.nanmean(p):.2f} px")

    def mt_line_scan(self):
        it = self._two()
        if not it:
            return
        v = self.ask("Varrimento por linha", [("ref", "Linha de referência (px, vazio = mediana)", ""), ("k", "Escala (mm por px de desvio)", 0.5),
                                              ("t", "Limiar (fração do máximo)", 0.3, (0.01, 1))])
        if v:
            ref = float(v["ref"]) if v["ref"].strip() else None
            imgs = [i.data for i in it]
            self.task("A processar…", lambda: m2.line_scan(imgs, ref, v["k"], v["t"]),
                      lambda H: self.add_image(f"altura linha laser ({len(imgs)} perfis)", H))

    def mt_laser_demo(self):
        Z = fr.demo_object(200, 60)
        for k, im in enumerate(m2.simulate_laser_images(Z, 2.0)):
            self.add_image(f"laser {k + 1:03d}", im, show=False)
        self.img_list.setCurrentRow(len(self.images) - 1)
        self.log("Selecione as imagens 'laser' e use Varrimento (referência 60, escala 0,5).")

    # --- estéreo ------------------------------------------------------
    def mt_stereo_cal(self):
        it = self._two()
        if not it or len(it) % 2:
            return self.error("Selecione um número par de imagens: primeiro todas as da câmara esquerda, depois as da direita.")
        v = self.ask("Calibração estéreo", [("c", "Cantos internos (colunas)", 9, (2, 50)), ("r", "Cantos internos (linhas)", 6, (2, 50)),
                                            ("s", "Tamanho do quadrado (mm)", 10.0)])
        if not v:
            return
        n = len(it) // 2
        L, R = [i.data for i in it[:n]], [i.data for i in it[n:]]

        def done(cal):
            self.stereo_calib = cal
            np.set_printoptions(precision=4, suppress=True)
            txt = (f"Pares usados: {cal['pares']}\nErro RMS: {cal['erro_rms_px']:.4f} px\nDistância entre câmaras: {cal['base_mm']:.3f} mm\n\n"
                   f"K esquerda:\n{cal['K1']}\n\nK direita:\n{cal['K2']}\n\nRotação:\n{cal['R']}\n\nTranslação (mm):\n{cal['T'].ravel()}")
            self.log(txt); TextDialog.show_text(self, "Calibração estéreo", txt)
        self.task("A calibrar…", lambda: m2.stereo_calibrate(L, R, v["c"], v["r"], v["s"]), done)

    def mt_stereo_depth(self):
        if not getattr(self, "stereo_calib", None):
            return self.error("Calibre primeiro o par estéreo.")
        it = self._two()
        if not it:
            return
        v = self.ask("Profundidade estéreo", [("d", "Disparidades (múltiplo de 16)", 128, (16, 1024)), ("b", "Bloco (px, ímpar)", 7, (3, 31))])
        if v:
            disp, Z, _ = m2.stereo_depth(it[0].data, it[1].data, self.stereo_calib, (v["d"] // 16) * 16, v["b"] | 1)
            self.add_image("disparidade", disp, show=False)
            self.add_image("profundidade estéreo (mm)", Z)

    def mt_disparity(self):
        it = self._two()
        if it:
            v = self.ask("Disparidade", [("d", "Disparidades (múltiplo de 16)", 64, (16, 1024)), ("b", "Bloco (px)", 7, (3, 31))])
            if v:
                self.add_image("disparidade", m2.disparity_uncalibrated(it[0].data, it[1].data, (v["d"] // 16) * 16, v["b"] | 1))

    def mt_stereo_demo(self):
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], float)
        rng = np.random.default_rng(1)
        L, R = [], []
        for k in range(6):
            rv = rng.normal(0, 0.25, 3); tv = np.array([-40 + rng.normal(0, 5), -30 + rng.normal(0, 5), 400 + rng.normal(0, 30)])
            L.append(m2.render_chessboard_view(rv, tv, K)); R.append(m2.render_chessboard_view(rv, tv - np.array([60, 0, 0]), K))
        for k, a in enumerate(L):
            self.add_image(f"esquerda {k + 1}", a, show=False)
        for k, a in enumerate(R):
            self.add_image(f"direita {k + 1}", a, show=k == len(R) - 1)
        self.log("Selecione as 12 imagens (esquerda 1–6 e depois direita 1–6) e use Calibrar par estéreo (9×6, 10 mm).")

    # --- nuvens -------------------------------------------------------
    def mt_import_cloud(self):
        p, _ = QFileDialog.getOpenFileName(self, "Nuvem de pontos", self.last_dir, "Pontos (*.txt *.xyz *.asc *.csv *.ply);;Todos (*)")
        if not p:
            return
        try:
            P = m2.load_points(p)
        except Exception as e:
            return self.error(e)
        import trimesh
        pc = trimesh.Trimesh(P, np.zeros((0, 3), int), process=False)
        o = self.add_mesh(pc, os.path.basename(p), place=False)
        self.cloud_pts = getattr(self, "cloud_pts", {}); self.cloud_pts[o.id] = P
        self.log(f"Nuvem com {len(P)} pontos. Use Triangulação de Delaunay para obter uma malha.")
        self._show_points(o)

    def _show_points(self, o):
        self.view3d.refresh(); self.view3d.fit()

    def mt_export_cloud(self):
        o = self.sel_objs(1, True)
        if not o:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Exportar nuvem", os.path.join(self.last_dir, o[0].name + ".txt"), "Texto (*.txt)")
        if p:
            m = o[0].mesh
            m2.save_points_txt(p, m.vertices, m.vertex_normals if len(m.faces) else None)
            self.log(f"Nuvem exportada: {p} ({len(m.vertices)} pontos)")

    def mt_delaunay(self):
        o = self.sel_objs(1, True)
        if not o:
            return
        v = self.ask("Delaunay", [("e", "Aresta máxima (mm, 0 = sem limite)", 0.0)])
        if v:
            P = np.asarray(o[0].mesh.vertices)
            self.task("A triangular…", lambda: m2.delaunay_mesh(P, v["e"] or None),
                      lambda m: self.add_mesh(m, f"Delaunay {o[0].name}", place=False))

    def mt_icp(self):
        o = self.sel_objs(2, True)
        if not o:
            return
        src, dst = np.asarray(o[0].mesh.vertices), np.asarray(o[1].mesh.vertices)
        if len(src) > 5000:
            src = src[np.random.default_rng(0).choice(len(src), 5000, replace=False)]

        def done(r):
            M, cost = r
            self.doc.snapshot(); o[0].mesh.apply_transform(M); self.doc.changed()
            self.log(f"ICP: {o[0].name} alinhado a {o[1].name}; erro médio {cost:.4g} mm\n{np.round(M, 4)}")
        self.task("A alinhar…", lambda: m2.icp_align(src, dst), done)

    def mt_turntable(self):
        it = self._two()
        if not it:
            return
        v = self.ask("Mesa rotativa", [("a0", "Ângulo inicial (°)", 0.0), ("step", "Passo (°)", 360.0 / len(it)),
                                       ("px", "Tamanho do pixel (mm)", 0.5), ("ax", "Eixo de rotação X (px)", float(it[0].data.shape[1] / 2))],
                     note="Cada imagem é um mapa de altura visto de lado; a altura passa a ser a profundidade radial.")
        if not v:
            return
        clouds, angs = [], []
        for k, i in enumerate(it):
            Z = self._arr(i)
            pts, _ = fr.depth_to_points(Z, v["px"])
            # x da imagem → X, altura → Y (profundidade), linha → Z
            P = np.c_[pts[:, 0] - v["ax"] * v["px"], pts[:, 2], pts[:, 1]]
            clouds.append(P); angs.append(v["a0"] + k * v["step"])
        P = m2.merge_turntable(clouds, angs)
        import trimesh
        o = self.add_mesh(trimesh.Trimesh(P, np.zeros((0, 3), int), process=False), "nuvem mesa rotativa", place=False)
        self._show_points(o)
        self.log(f"Nuvem combinada: {len(P)} pontos de {len(it)} vistas")
