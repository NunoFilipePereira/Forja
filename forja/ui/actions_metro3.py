"""Metrologia (3): dois planos de referência, fusão, DIC em sequência, multivista, feixe cónico, nuvem regularizada."""
import numpy as np
from ..core import metro3 as m3, ct
from .widgets import TextDialog


class Metro3Actions:
    def menus_metro3(self):
        return [
            ("Metro&logia", [
                None,
                ("Fase → altura com dois planos de referência (objeto, plano 1, plano 2)…", self.m3_two_plane),
                ("Fundir dois mapas de profundidade (Z1, Z2[, Q1, Q2])…", self.m3_fuse),
                ("DIC numa sequência de imagens…", self.m3_dic_seq),
                ("Calibração multivista (uma câmara)…", self.m3_multiview),
                ("Corrigir distorção radial (manual)…", self.m3_radial),
                ("Nuvem (objeto 3D) → grelha regular (mapa de altura)…", self.m3_regular),
            ]),
            ("&CT / Volume", [
                None,
                ("Feixe cónico: projeções → volume (FDK)…", self.m3_fdk),
                ("Feixe cónico: fantasma + projeções de teste…", self.m3_cone_demo),
            ]),
        ]

    def _arr2(self, it):
        a = np.asarray(it.data, float)
        return a.mean(2) if a.ndim == 3 else a

    def m3_two_plane(self):
        it = self.sel_images(3)
        if not it:
            return
        v = self.ask("Dois planos de referência", [("h1", "Altura do plano 1 (mm)", 0.0), ("h2", "Altura do plano 2 (mm)", 10.0)],
                     note=f"Fase do objeto: {it[0].name}\nFase do plano 1: {it[1].name}\nFase do plano 2: {it[2].name}\n(fases já desembrulhadas)")
        if v:
            H = m3.two_plane_height(self._arr2(it[0]), self._arr2(it[1]), self._arr2(it[2]), v["h1"], v["h2"])
            self.add_image(f"altura absoluta mm({it[0].name})", H)

    def m3_fuse(self):
        it = self.sel_images(2)
        if not it:
            return
        v = self.ask("Fundir profundidade", [("m", "Modo", "média ponderada", ["média ponderada", "melhor qualidade"])],
                     note="Selecione Z1, Z2 e, opcionalmente, as qualidades Q1 e Q2 (por esta ordem).")
        if v:
            q1 = self._arr2(it[2]) if len(it) >= 4 else None
            q2 = self._arr2(it[3]) if len(it) >= 4 else None
            self.add_image("profundidade fundida", m3.fuse_depth(self._arr2(it[0]), self._arr2(it[1]), q1, q2, v["m"]))

    def m3_dic_seq(self):
        it = self.sel_images(2)
        if not it:
            return
        v = self.ask("DIC em sequência", [("m", "Correlacionar", "com a primeira", ["com a primeira", "cada par consecutivo"]),
                                          ("s", "Subconjunto (px)", 31, (7, 201)), ("st", "Passo (px)", 16, (2, 200)), ("r", "Procura (px)", 20, (1, 500))])
        if not v:
            return
        frames = [i.data for i in it]

        def done(res):
            x = [r["frame"] for r in res]
            self.chart.series += [{"x": x, "y": [r["dx"] for r in res], "label": "deslocamento X médio (px)", "kind": "linha e pontos"},
                                  {"x": x, "y": [r["dy"] for r in res], "label": "deslocamento Y médio (px)", "kind": "linha e pontos"}]
            self.chart.opts.update(xlabel="imagem", ylabel="deslocamento (px)", title="DIC em sequência")
            self._show_chart()
            self.log("DIC em sequência: " + "; ".join(f"#{r['frame']}: ({r['dx']:.3f}, {r['dy']:.3f}) corr {r['corr']:.2f}" for r in res))
        self.task("A correlacionar a sequência…", lambda: m3.dic_sequence(frames, v["m"], v["s"] | 1, v["st"], v["r"]), done)

    def m3_multiview(self):
        it = self.sel_images(3)
        if not it:
            return
        v = self.ask("Calibração multivista", [("p", "Alvo", "tabuleiro", ["tabuleiro", "grelha de círculos"]), ("c", "Colunas", 9, (2, 50)),
                                               ("r", "Linhas", 6, (2, 50)), ("s", "Espaçamento (mm)", 10.0)])
        if not v:
            return
        imgs = [i.data for i in it]

        def done(cal):
            self.calib = {"K": cal["K"], "distorção": cal["distorção"], "erro_rms_px": cal["erro_rms_px"]}
            np.set_printoptions(precision=4, suppress=True)
            lines = [f"Erro RMS: {cal['erro_rms_px']:.4f} px", f"K:\n{cal['K']}", f"Distorção: {cal['distorção']}", "", "Poses do alvo por vista:"]
            for p in cal["poses"]:
                lines.append(f"  vista {p['vista']}: distância {p['distância (mm)']:.1f} mm, rotação {np.round(p['rotação (°)'], 2)}, "
                             f"câmara em {np.round(p['posição da câmara (mm)'], 1)}")
            TextDialog.show_text(self, "Calibração multivista", "\n".join(lines)); self.log("\n".join(lines))
        self.task("A calibrar…", lambda: m3.multiview_calibration(imgs, v["c"], v["r"], v["s"], "tabuleiro" if v["p"] == "tabuleiro" else "círculos"), done)

    def m3_radial(self):
        v = self.ask("Distorção radial", [("k1", "k1 (negativo corrige barril)", -0.1, (-5, 5)), ("k2", "k2", 0.0, (-5, 5)),
                                          ("f", "Distância focal (px, 0 = automática)", 0.0, (0, 1e6))])
        if v: self.img_op("sem distorção", lambda a: m3.radial_undistort(a, v["k1"], v["k2"], f=v["f"] or None))

    def m3_regular(self):
        o = self.sel_objs(1, True)
        if not o:
            return
        v = self.ask("Grelha regular", [("s", "Espaçamento (mm)", 0.5, (0.001, 1000))])
        if v:
            P = np.asarray(o[0].mesh.vertices)
            self.task("A reamostrar…", lambda: m3.regularize_points(P, v["s"]),
                      lambda r: (self.add_image(f"grelha regular({o[0].name}) {r[2]:g} mm", r[0], {"px_mm": r[2]}),
                                 self.log(f"Nuvem regularizada: {r[0].shape[1]}×{r[0].shape[0]} pontos, origem {np.round(r[1], 3)}")))

    def m3_fdk(self):
        it = self.sel_images(4)
        if not it:
            return
        v = self.ask("FDK (feixe cónico)", [("sdd", "Distância fonte–detetor (mm)", 400.0), ("sod", "Distância fonte–eixo (mm)", 250.0),
                                            ("px", "Pixel do detetor (mm)", 1.0), ("n", "Voxels XY (0 = automático)", 0, (0, 1024)),
                                            ("nz", "Voxels Z (0 = automático)", 0, (0, 1024))],
                     note=f"{len(it)} projeções numa volta completa (360°), por ordem da lista.")
        if v:
            P = [i.data for i in it]
            self.task("A reconstruir (FDK)…", lambda: m3.fdk(P, v["sdd"], v["sod"], v["px"], v["n"] or None, v["nz"] or None),
                      lambda V: self.add_volume(f"volume FDK ({len(P)} projeções)", V))

    def m3_cone_demo(self):
        v = self.ask("Feixe cónico de teste", [("n", "Resolução (voxels)", 32, (16, 96)), ("a", "Projeções (360°)", 60, (8, 720))])
        if not v:
            return
        V = ct.phantom3d(v["n"])
        det = (int(v["n"] * 1.5), int(v["n"] * 1.5))
        self.add_volume("fantasma 3D (cone)", V)
        self.task("A projetar…", lambda: m3.cone_forward(V, v["a"], det_shape=det),
                  lambda proj: [self.add_image(f"cone {k + 1:03d}", p, show=k == len(proj) - 1) for k, p in enumerate(proj)]
                  or self.log(f"{len(proj)} projeções cónicas (fonte–detetor 400 mm, fonte–eixo 250 mm). Selecione-as e use FDK."))
