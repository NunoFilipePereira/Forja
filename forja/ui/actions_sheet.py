"""Menu Chapa: corte laser/CNC."""
import numpy as np
from shapely import affinity
from shapely.ops import unary_union
from ..core import sheet as sh, shapes2d as s2, toolpath as tp
from .widgets import TextDialog


class SheetActions:
    def menus_sheet(self):
        return [("C&hapa", [
            ("Caixa com juntas em dente…", self.sh_finger_box),
            ("Leque: planificar tronco de cone…", self.sh_unfan),
            ("Leque: medir a partir de 2 linhas selecionadas", self.sh_fan_measure),
            None,
            ("Escultura em grelha de fatias (objeto 3D)…", self.sh_hash),
            ("Escultura em fatias radiais (objeto 3D)…", self.sh_radial),
            ("Camadas empilhadas (objeto 3D)…", self.act_slice_layers),
            None,
            ("Dividir forma selecionada", [
                ("Em grelha…", lambda: self.sh_split("grelha")),
                ("Em grelha desfasada (tijolo)…", lambda: self.sh_split("desfasada")),
                ("Radial…", lambda: self.sh_split("radial")),
                ("Pelas linhas selecionadas", lambda: self.sh_split("linhas")),
            ]),
            ("Compensar largura de corte (kerf)…", self.sh_kerf),
            None,
            ("Pontes no percurso (micro-juntas)…", self.sh_bridges),
            ("Entrada/saída de corte…", self.sh_lead),
            ("Tempo de máquina estimado…", self.sh_time),
        ])]

    def _add2d(self, items, name_fn=None, layout_gap=0.0, select=True):
        """items: lista de geometrias ou (nome, geometria)."""
        shapes = []
        x = 0.0
        for k, it in enumerate(items):
            name, g = it if isinstance(it, tuple) else (f"peça {k + 1}", it)
            if layout_gap:
                b = g.bounds
                g = affinity.translate(g, x - b[0], -b[1]); x += b[2] - b[0] + layout_gap
            shapes.append(s2.Shape("geo", {}, g, name))
        self.canvas.add_shapes(shapes, select=select)
        self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()
        return shapes

    def sh_finger_box(self):
        v = self.ask("Caixa com juntas em dente", [("w", "Largura exterior (mm)", 100.0), ("d", "Profundidade (mm)", 80.0),
                                                   ("h", "Altura (mm)", 60.0), ("t", "Espessura da chapa (mm)", 3.0),
                                                   ("f", "Largura do dente (mm)", 10.0), ("tol", "Folga (mm)", 0.1),
                                                   ("open", "Sem tampa", False)])
        if not v:
            return
        try:
            panels = sh.finger_box(v["w"], v["d"], v["h"], v["t"], v["f"], v["tol"], v["open"])
        except ValueError as e:
            return self.error(e)
        self._add2d(panels)
        self.log(f"Caixa: {len(panels)} painéis para chapa de {v['t']} mm")

    def sh_unfan(self):
        v = self.ask("Planificar tronco de cone", [("d1", "Diâmetro em baixo (mm)", 100.0), ("d2", "Diâmetro em cima (mm)", 60.0),
                                                   ("h", "Altura (mm)", 80.0), ("ov", "Aba de colagem (mm)", 0.0)])
        if v:
            poly, info = sh.unfan(v["d1"], v["d2"], v["h"], overlap=v["ov"])
            self._add2d([("leque", poly)])
            self.log("Leque: " + ", ".join(f"{k} {val:.3f}" if isinstance(val, float) else f"{k} {val}" for k, val in info.items()))

    def sh_fan_measure(self):
        lines = [s for s in self.canvas.selected_shapes() if s.geom.geom_type == "LineString"]
        if len(lines) != 2:
            return self.error("Desenhe duas linhas sobre as arestas laterais do leque (Desenho 2D ▸ Linha) e selecione-as.")
        try:
            info = sh.fan_from_lines(list(lines[0].geom.coords)[:2], list(lines[1].geom.coords)[:2])
        except Exception as e:
            return self.error(f"As linhas não convergem: {e}")
        TextDialog.show_text(self, "Leque", "\n".join(f"{k}: {np.round(val, 3)}" for k, val in info.items()))

    def _mesh_one(self):
        o = self.sel_objs(1, True)
        return o[0] if o else None

    def sh_hash(self):
        o = self._mesh_one()
        if not o:
            return
        v = self.ask("Grelha de fatias", [("nx", "Fatias em X", 5, (1, 100)), ("ny", "Fatias em Y", 5, (1, 100)),
                                          ("t", "Espessura da chapa (mm)", 3.0), ("tol", "Folga (mm)", 0.1)],
                     note="Cria perfis que encaixam em grelha (ranhuras a meia altura). Escale o objeto ao tamanho final primeiro.")
        if not v:
            return
        mesh = o.mesh.copy()
        self.task("A fatiar…", lambda: sh.hash_slices(mesh, v["nx"], v["ny"], v["t"], v["tol"]),
                  lambda res: self._add2d([(f"fatia {a}={p:.1f}", g) for a, p, g in res if g is not None and not g.is_empty], layout_gap=5))

    def sh_radial(self):
        o = self._mesh_one()
        if not o:
            return
        v = self.ask("Fatias radiais", [("n", "Número de fatias", 8, (2, 72)), ("t", "Espessura da chapa (mm)", 3.0),
                                        ("tol", "Folga (mm)", 0.1)])
        if not v:
            return
        mesh = o.mesh.copy()

        def done(res):
            pieces, base = res
            self._add2d([(n, g) for n, a, g in pieces] + [("base com ranhuras", base)], layout_gap=5)
        self.task("A fatiar…", lambda: sh.radial_slices(mesh, v["n"], v["t"], v["tol"]), done)

    def sh_split(self, mode):
        sel = self.canvas.selected_shapes()
        shapes = [s for s in sel if s.closed]
        if not shapes:
            return self.error("Selecione a forma fechada a dividir no separador 2D.")
        g = unary_union([s.geom for s in shapes])
        if mode in ("grelha", "desfasada"):
            v = self.ask("Dividir em grelha", [("w", "Largura máxima da peça (mm)", 200.0), ("h", "Altura máxima (mm)", 200.0),
                                               ("j", "Encaixes", "cauda de andorinha", ["nenhum", "cauda de andorinha", "botão"]),
                                               ("js", "Tamanho do encaixe (mm)", 8.0)])
            if not v:
                return
            pieces = sh.grid_cut(g, v["w"], v["h"], staggered=mode == "desfasada")
        elif mode == "radial":
            v = self.ask("Dividir radialmente", [("n", "Setores", 6, (2, 360)), ("rot", "Rotação (°)", 0.0),
                                                 ("j", "Encaixes", "nenhum", ["nenhum", "cauda de andorinha", "botão"]), ("js", "Tamanho do encaixe (mm)", 8.0)])
            if not v:
                return
            pieces = sh.radial_cut(g, v["n"], rotation=v["rot"])
        else:
            lines = [list(s.geom.coords) for s in sel if s.geom.geom_type == "LineString"]
            if not lines:
                return self.error("Selecione também as linhas de corte (Desenho 2D ▸ Linha).")
            v = {"j": "nenhum", "js": 8.0}
            pieces = sh.line_cut(g, lines)
        if v["j"] != "nenhum":
            pieces = sh.add_joints(pieces, v["j"], v["js"])
        self.canvas.replace_selected([s2.Shape("geo", {}, p, f"peça {k + 1}") for k, p in enumerate(pieces)])
        self.log(f"Dividido em {len(pieces)} peças")

    def sh_kerf(self):
        v = self.ask("Kerf", [("k", "Largura do corte (mm)", 0.15, (0, 10))],
                     note="As formas fechadas selecionadas crescem metade do kerf para ficarem com a medida certa depois do corte.")
        if v:
            self._map2(lambda s: s2.offset(s, v["k"] / 2) if s.closed else s)

    def sh_bridges(self):
        if not self.toolpaths:
            return self.error("Crie primeiro um percurso (Desenho 2D ▸ Percurso de ferramenta).")
        v = self.ask("Pontes", [("n", "Pontes por contorno", 2, (1, 20)), ("w", "Largura (mm)", 1.0), ("min", "Só em contornos com mais de (mm)", 20.0)])
        if v:
            self.toolpaths = sh.add_bridges(self.toolpaths, v["n"], v["w"], v["min"])
            self.canvas.clear_overlay(); self.canvas.add_overlay_paths(self.toolpaths)
            self.log(f"Pontes adicionadas: {len(self.toolpaths)} trajetos")

    def sh_lead(self):
        if not self.toolpaths:
            return self.error("Crie primeiro um percurso (Desenho 2D ▸ Percurso de ferramenta).")
        v = self.ask("Entrada/saída", [("l", "Comprimento (mm)", 2.0), ("m", "Tipo", "entrada e saída", ["entrada", "saída", "entrada e saída"])])
        if v:
            self.toolpaths = sh.add_lead(self.toolpaths, v["l"], v["m"])
            self.canvas.clear_overlay(); self.canvas.add_overlay_paths(self.toolpaths)

    def sh_time(self):
        if not self.toolpaths:
            return self.error("Crie primeiro um percurso.")
        v = self.ask("Tempo de máquina", [("f", "Velocidade de corte (mm/min)", 1000.0), ("tr", "Velocidade em vazio (mm/min)", 3000.0),
                                          ("p", "Passagens", 1, (1, 100))])
        if v:
            t = tp.estimate_time(self.toolpaths, v["f"], v["p"], v["tr"])
            L = tp.path_length(self.toolpaths)
            self.log(f"Comprimento de corte {L:.1f} mm × {v['p']} passagem(ns); tempo estimado {t / 60:.1f} min")
            self.statusBar().showMessage(f"Tempo estimado: {t / 60:.1f} min")
