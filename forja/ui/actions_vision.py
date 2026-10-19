"""Menu Visão por computador."""
import numpy as np
from ..core import vision as vs
from .widgets import TextDialog


class VisionActions:
    def menus_vision(self):
        return [("Vi&são", [
            ("Rostos", [("Detetar rostos e pontos de referência…", self.vi_faces),
                        ("Anonimizar rostos…", self.vi_anon),
                        ("Comparar rostos (A, B) — semelhança aproximada", self.vi_face_cmp)]),
            ("Marcadores e códigos", [("Detetar marcadores ArUco…", self.vi_aruco),
                                      ("Gerar marcador ArUco…", self.vi_aruco_gen),
                                      ("Ler códigos QR", self.vi_qr),
                                      ("Gerar código QR…", self.vi_qr_gen)]),
            ("Alvos de referência", [("Quadrados pretos…", self.vi_squares),
                                     ("Círculos pretos…", self.vi_circles),
                                     ("Grelha de círculos (calibração)…", self.vi_circle_grid),
                                     ("Gerar grelha de círculos…", self.vi_circle_grid_gen)]),
            None,
            ("Foco total (pilha de fotos com focos diferentes)…", self.vi_all_focus),
            ("Movimento entre duas imagens (A, B)…", self.vi_motion),
            ("Subtração de fundo (sequência)…", self.vi_bg),
            ("Média temporal (sequência)", self.vi_time_avg),
            ("Segmentação por cor (HSV)…", self.vi_color),
            ("Efeito artístico…", self.vi_effect),
        ])]

    def vi_faces(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Rostos", [("s", "Confiança mínima", 0.7, (0.05, 0.99))])
        if not v:
            return
        F = vs.detect_faces(it.data, v["s"])
        self.add_image(f"rostos({it.name})", vs.draw_faces(it.data, F))
        lines = [f"{len(F)} rosto(s)"]
        for k, f in enumerate(F, 1):
            x, y, w, h = f["caixa"]
            lines.append(f"#{k}: caixa ({x:.0f}, {y:.0f}, {w:.0f}×{h:.0f}), confiança {f['confiança']:.2f}, "
                         f"olhos ({f['pontos'][0][0]:.0f},{f['pontos'][0][1]:.0f}) e ({f['pontos'][1][0]:.0f},{f['pontos'][1][1]:.0f})")
        self.log("\n".join(lines))

    def vi_anon(self):
        v = self.ask("Anonimizar", [("m", "Modo", "desfocar", ["desfocar", "pixelizar"])])
        if v: self.img_op("anonimizada", lambda a: vs.anonymize_faces(a, vs.detect_faces(a, 0.6), v["m"]))

    def vi_face_cmp(self):
        it = self.sel_images(2)
        if it:
            try:
                s = vs.face_similarity(it[0].data, it[1].data)
            except ValueError as e:
                return self.error(e)
            msg = f"Semelhança entre os rostos: {s:.3f} (1 = idêntico). Indicação aproximada, não é identificação biométrica."
            self.log(msg); self.statusBar().showMessage(msg)

    def vi_aruco(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("ArUco", [("d", "Dicionário", "4x4 (50)", list(vs.ARUCO_DICTS))])
        if v:
            res, out = vs.aruco_detect(it.data, v["d"])
            self.add_image(f"aruco({it.name})", out)
            self.log(f"{len(res)} marcadores: " + "; ".join(f"id {r['id']} em ({r['centro'][0]:.1f}, {r['centro'][1]:.1f}), lado {r['lado_px']:.1f} px" for r in res))

    def vi_aruco_gen(self):
        v = self.ask("Gerar ArUco", [("id", "Identificador", 0, (0, 999)), ("s", "Tamanho (px)", 300, (50, 4000)),
                                     ("d", "Dicionário", "4x4 (50)", list(vs.ARUCO_DICTS))])
        if v:
            try:
                self.add_image(f"ArUco {v['id']}", vs.aruco_generate(v["id"], v["s"], v["d"]))
            except Exception as e:
                self.error(f"Identificador fora do dicionário: {e}")

    def vi_qr(self):
        it = self.cur_image()
        if it:
            res, out = vs.qr_detect(it.data)
            self.add_image(f"QR({it.name})", out)
            msg = f"{len(res)} código(s): " + " | ".join(r["texto"] or "(ilegível)" for r in res)
            self.log(msg); self.statusBar().showMessage(msg)

    def vi_qr_gen(self):
        v = self.ask("Gerar QR", [("t", "Texto", "https://"), ("s", "Escala (px por módulo)", 8, (1, 50))])
        if v:
            self.add_image("QR", vs.qr_generate(v["t"], v["s"]))

    def _report(self, title, items, fmt):
        self.log(f"{len(items)} {title}: " + "; ".join(fmt(x) for x in items[:100]))

    def vi_squares(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Quadrados", [("a", "Área mínima (px²)", 100.0), ("d", "Escuros sobre claro", True)])
        if v:
            from ..core import imageops2 as io2
            R = vs.black_squares(it.data, v["a"], dark=v["d"])
            self.add_image(f"quadrados({it.name})", io2.draw_shapes(it.data, polys=[r["cantos"] for r in R]))
            self._report("quadrados", R, lambda r: f"({r['centro'][0]:.1f}, {r['centro'][1]:.1f}) lado {r['lado']:.1f}")

    def vi_circles(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Círculos", [("a", "Área mínima (px²)", 30.0), ("d", "Escuros sobre claro", True)])
        if v:
            from ..core import imageops2 as io2
            R = vs.black_circles(it.data, v["a"], v["d"])
            self.add_image(f"círculos({it.name})", io2.draw_shapes(it.data, circles=[(*r["centro"], r["diâmetro"] / 2) for r in R]))
            self._report("círculos", R, lambda r: f"({r['centro'][0]:.1f}, {r['centro'][1]:.1f}) ⌀ {r['diâmetro']:.1f}")

    def vi_circle_grid(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Grelha de círculos", [("c", "Colunas", 7, (2, 50)), ("r", "Linhas", 6, (2, 50)), ("s", "Simétrica", True)])
        if v:
            ok, C, out = vs.circle_grid(it.data, v["c"], v["r"], v["s"])
            self.add_image(f"grelha({it.name})", out)
            self.log(f"Grelha {'encontrada' if ok else 'NÃO encontrada'}" + (f": {len(C)} centros" if ok else ""))

    def vi_circle_grid_gen(self):
        v = self.ask("Gerar grelha", [("c", "Colunas", 7, (2, 50)), ("r", "Linhas", 6, (2, 50)), ("sp", "Espaçamento (px)", 40, (5, 1000)),
                                      ("ra", "Raio (px)", 12, (1, 500))])
        if v:
            self.add_image("grelha de círculos", vs.circle_grid_pattern(v["c"], v["r"], v["sp"], v["ra"]))

    def vi_all_focus(self):
        it = self.sel_images(2)
        if not it:
            return
        v = self.ask("Foco total", [("w", "Janela (px)", 9, (3, 51)), ("s", "Suavização", 2.0, (0, 20))])
        if v:
            stack = [i.data for i in it]
            def done(r):
                out, k = r
                self.add_image("índice da foto mais nítida", k, show=False)
                self.add_image(f"foco total ({len(stack)} fotos)", out)
            self.task("A combinar…", lambda: vs.all_in_focus(stack, v["w"], v["s"]), done)

    def vi_motion(self):
        it = self.sel_images(2)
        if not it:
            return
        v = self.ask("Movimento", [("t", "Limiar", 25, (1, 255)), ("a", "Área mínima (px²)", 50, (1, 1000000))])
        if v:
            m, boxes, out = vs.motion(it[0].data, it[1].data, v["t"], v["a"])
            self.add_image("máscara de movimento", m, show=False)
            self.add_image("movimento", out)
            self.log(f"{len(boxes)} zona(s) em movimento: " + "; ".join(f"({x},{y}) {w}×{h}" for x, y, w, h in boxes[:50]))

    def vi_bg(self):
        it = self.sel_images(3)
        if it:
            masks, bg = vs.background_subtraction([i.data for i in it])
            if bg is not None:
                self.add_image("fundo estimado", bg, show=False)
            for k, m in enumerate(masks):
                self.add_image(f"primeiro plano {k + 1}", m, show=k == len(masks) - 1)

    def vi_time_avg(self):
        it = self.sel_images(2)
        if it:
            try:
                self.add_image(f"média temporal ({len(it)})", vs.time_average([i.data for i in it]))
            except ValueError:
                self.error("As imagens têm de ter o mesmo tamanho.")

    def vi_color(self):
        v = self.ask("Segmentação por cor", [("h0", "Matiz mínima (0–179)", 0, (0, 179)), ("h1", "Matiz máxima (0–179)", 20, (0, 179)),
                                             ("s", "Saturação mínima", 80, (0, 255)), ("v", "Brilho mínimo", 60, (0, 255))],
                     note="Matiz: vermelho ≈ 0/170, amarelo ≈ 30, verde ≈ 60, azul ≈ 120. Se a mínima > máxima, o intervalo dá a volta.")
        if v: self.img_op("cor", lambda a: vs.color_segment(a, v["h0"], v["h1"], v["s"], v["v"]))

    def vi_effect(self):
        v = self.ask("Efeito", [("k", "Efeito", "pintura", ["pintura", "desenho a lápis", "banda desenhada", "realçar detalhe", "sépia"])])
        if v: self.img_op(v["k"], lambda a: vs.effect(a, v["k"]))
