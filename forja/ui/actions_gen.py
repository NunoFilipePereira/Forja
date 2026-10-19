"""Menu Gerar: modelos 3D paramétricos."""
import numpy as np
from shapely.ops import unary_union
from ..core import generators as gn, meshops as mo, imageops as iop, shapes2d as s2


class GenActions:
    def menus_gen(self):
        return [("&Gerar", [
            ("Formas básicas", [
                ("Prisma / forma extrudida…", self.gen_prism),
                ("Pirâmide…", self.gen_pyramid),
                ("Extrusão com torção (forma 2D ou básica)…", self.gen_twist),
                ("Sólido de revolução (perfil 2D)…", self.gen_lathe),
            ]),
            ("Caixas", [
                ("Caixa com tampa…", self.gen_box),
                ("Caixas encaixadas (umas dentro das outras)…", self.gen_nested),
            ]),
            ("Vaso…", self.gen_vase),
            ("Mola…", self.gen_spring),
            ("Cavilha…", self.gen_dowel),
            None,
            ("Fotografia", [
                ("Relevo com moldura…", self.gen_framed),
                ("Litofania cilíndrica (candeeiro)…", self.gen_cyl_litho),
                ("Litofania esférica…", self.gen_sph_litho),
                ("Base de candeeiro…", self.gen_lamp_base),
            ]),
            ("Porta-chaves…", self.gen_keychain),
            ("Cortador de bolachas (forma 2D)…", self.gen_cutter),
            ("Carimbo (forma 2D / texto)…", self.gen_stamp),
            ("Puzzle 3D…", self.gen_puzzle3d),
            ("Puzzle 2D (para laser)…", self.gen_puzzle),
            None,
            ("Molde em duas partes (objeto 3D)…", self.gen_mold),
            ("Low poly (objeto 3D)…", self.gen_lowpoly),
            ("Estrutura em arame (objeto 3D)…", self.gen_wire),
            ("Pilha de cópias (objeto 3D)…", self.gen_stack),
            ("Suportes de impressão (objeto 3D)…", self.gen_supports),
            ("Furos para ímanes (objeto 3D)…", self.gen_magnets),
        ])]

    # ------------------------------------------------------------------
    def _new(self, mesh, name, label="A gerar…"):
        if callable(mesh):
            return self.task(label, mesh, lambda m: self._new(m, name))
        meshes = mesh if isinstance(mesh, (list, tuple)) else [mesh]
        objs = [self.add_mesh(m, name if len(meshes) == 1 else f"{name} {k + 1}", select=False) for k, m in enumerate(meshes)]
        self._select_objs(objs)
        self.tabs.setCurrentWidget(self.view3d); self.view3d.fit()
        self.log(f"Gerado: {name}" + (f" ({len(meshes)} peças)" if len(meshes) > 1 else ""))

    def _shape_from_canvas_or_basic(self, title, extra):
        sel = [s for s in self.canvas.selected_shapes() if s.closed]
        fields = ([] if sel else [("shape", "Forma", "estrela", gn.SHAPES_2D), ("size", "Tamanho (mm)", 40.0)]) + extra
        v = self.ask(title, fields, note=("Usa as formas 2D selecionadas." if sel else
                                          "Dica: selecione formas 2D (ou texto) no separador 2D para usar o seu contorno."))
        if not v:
            return None, None
        g = unary_union([s.geom for s in sel]) if sel else gn.base_shape(v["shape"], v["size"])
        return g, v

    def _gray_image(self):
        it = self._image_for_3d()
        if it is None:
            return None, None
        g = iop.to_gray(it.data) if it.data.dtype == np.uint8 else iop.to_u8(it.data)
        return it, g

    def gen_prism(self):
        v = self.ask("Prisma", [("k", "Forma", "hexágono", gn.SHAPES_2D), ("s", "Tamanho (mm)", 30.0), ("h", "Altura (mm)", 20.0)])
        if v: self._new(gn.prism(v["k"], v["s"], v["h"]), v["k"])

    def gen_pyramid(self):
        v = self.ask("Pirâmide", [("n", "Lados", 4, (3, 8)), ("s", "Base (mm)", 40.0), ("h", "Altura (mm)", 30.0)])
        if v: self._new(gn.pyramid(v["n"], v["s"], v["h"]), "pirâmide")

    def gen_twist(self):
        g, v = self._shape_from_canvas_or_basic("Extrusão com torção", [("h", "Altura (mm)", 60.0), ("t", "Torção (°)", 120.0),
                                                                       ("sc", "Escala no topo", 0.7, (0.0, 10))])
        if g is not None:
            self._new(lambda: mo.center_on_plate(gn.twisted_extrusion(g, v["h"], v["t"], v["sc"])), "extrusão torcida")

    def gen_lathe(self):
        sel = [s for s in self.canvas.selected_shapes() if s.closed]
        if not sel:
            return self.error("Desenhe e selecione no separador 2D um perfil fechado (X = raio, Y = altura) à direita do eixo Y.")
        v = self.ask("Revolução", [("seg", "Segmentos", 96, (3, 720)), ("deg", "Ângulo (°)", 360.0, (1, 360))])
        if v:
            g = unary_union([s.geom for s in sel])
            self._new(lambda: mo.drop_to_floor(gn.lathe(g, v["seg"], v["deg"])), "revolução")

    def gen_box(self):
        v = self.ask("Caixa com tampa", [("w", "Largura (mm)", 60.0), ("d", "Profundidade (mm)", 40.0), ("h", "Altura (mm)", 30.0),
                                         ("wall", "Parede (mm)", 2.0), ("bottom", "Fundo/tampo (mm)", 2.0), ("r", "Raio dos cantos (mm)", 4.0),
                                         ("type", "Tipo de tampa", "encaixe", ["encaixe", "sobreposta"]), ("lip", "Rebordo da tampa (mm)", 4.0),
                                         ("lh", "Altura tampa sobreposta (mm)", 8.0), ("cl", "Folga (mm)", 0.25)])
        if v:
            base, lid = gn.box_with_lid(v["w"], v["d"], v["h"], v["wall"], v["bottom"], v["r"], v["lh"], v["lip"], v["cl"], v["type"])
            self._new([base, lid], "caixa")

    def gen_nested(self):
        v = self.ask("Caixas encaixadas", [("n", "Número", 3, (1, 20)), ("w", "Largura exterior (mm)", 80.0), ("d", "Profundidade (mm)", 60.0),
                                           ("h", "Altura (mm)", 40.0), ("wall", "Parede (mm)", 1.6), ("gap", "Folga (mm)", 1.0)])
        if v: self._new(gn.nested_boxes(v["n"], v["w"], v["d"], v["h"], v["wall"], v["gap"]), "caixa encaixada")

    def gen_vase(self):
        v = self.ask("Vaso", [("h", "Altura (mm)", 120.0), ("rb", "Raio na base (mm)", 30.0), ("rm", "Raio a meio (mm)", 45.0),
                              ("rt", "Raio no topo (mm)", 28.0), ("lobes", "Lóbulos", 6, (0, 40)), ("amp", "Ondulação (fração)", 0.08, (0, 0.5)),
                              ("tw", "Torção (°)", 90.0), ("wall", "Parede (mm)", 1.6), ("hollow", "Oco", True)])
        if v:
            self._new(lambda: gn.vase(v["h"], v["rb"], v["rm"], v["rt"], v["lobes"], v["amp"], v["tw"], v["wall"], hollow=v["hollow"]), "vaso")

    def gen_spring(self):
        v = self.ask("Mola", [("r", "Raio da mola (mm)", 10.0), ("w", "Raio do arame (mm)", 1.5), ("p", "Passo (mm)", 6.0),
                              ("n", "Voltas", 5.0, (0.5, 200)), ("flat", "Pontas planas", True)])
        if v: self._new(lambda: gn.spring(v["r"], v["w"], v["p"], v["n"], flat_ends=v["flat"]), "mola")

    def gen_dowel(self):
        v = self.ask("Cavilha", [("d", "Diâmetro (mm)", 5.0), ("l", "Comprimento (mm)", 20.0), ("n", "Quantidade", 4, (1, 100))])
        if v:
            ms = []
            for k in range(v["n"]):
                m = gn.dowel(v["d"], v["l"]); m.apply_translation(((k % 10) * v["d"] * 2, (k // 10) * v["d"] * 2, 0)); ms.append(m)
            import trimesh
            self._new(trimesh.util.concatenate(ms), "cavilhas")

    def gen_framed(self):
        it, g = self._gray_image()
        if it is None:
            return
        v = self.ask("Relevo com moldura", [("w", "Largura (mm)", 100.0), ("rel", "Altura do relevo (mm)", 3.0), ("base", "Base (mm)", 1.5),
                                            ("b", "Moldura (mm, 0 = sem)", 5.0), ("inv", "Claro = alto", True), ("hole", "Furo para pendurar (mm, 0 = sem)", 4.0)])
        if v: self._new(lambda: gn.framed_relief(g, v["w"], v["rel"], v["base"], v["b"], None, v["inv"], hole_d=v["hole"]), f"relevo {it.name}")

    def gen_cyl_litho(self):
        it, g = self._gray_image()
        if it is None:
            return
        v = self.ask("Litofania cilíndrica", [("d", "Diâmetro (mm)", 80.0), ("h", "Altura (mm, 0 = proporcional)", 0.0),
                                              ("tmin", "Espessura mínima (mm)", 0.8), ("tmax", "Espessura máxima (mm)", 3.0),
                                              ("base", "Anel de base (mm, 0 = sem)", 3.0), ("res", "Resolução (px)", 500, (50, 2000))],
                     note="Imprima na vertical para melhor qualidade.")
        if v: self._new(lambda: gn.cylinder_lithophane(g, v["d"], v["h"] or None, v["tmin"], v["tmax"], v["res"], v["base"]), f"litofania {it.name}")

    def gen_sph_litho(self):
        it, g = self._gray_image()
        if it is None:
            return
        v = self.ask("Litofania esférica", [("d", "Diâmetro (mm)", 100.0), ("tmin", "Espessura mínima (mm)", 0.8),
                                            ("tmax", "Espessura máxima (mm)", 3.0), ("op", "Abertura em baixo (mm)", 40.0)])
        if v: self._new(lambda: gn.sphere_lithophane(g, v["d"], v["tmin"], v["tmax"], opening=v["op"]), f"litofania esférica {it.name}")

    def gen_lamp_base(self):
        v = self.ask("Base de candeeiro", [("d", "Diâmetro da litofania (mm)", 80.0), ("h", "Altura (mm)", 30.0), ("sock", "Furo do casquilho (mm)", 28.0),
                                           ("cab", "Furo do cabo (mm)", 8.0), ("t", "Espessura máx. da litofania (mm)", 3.0)])
        if v: self._new(lambda: gn.lamp_base(v["d"], v["h"], socket_d=v["sock"], cable_hole=v["cab"], shade_t=v["t"]), "base de candeeiro")

    def gen_keychain(self):
        from ..core import text3d
        v = self.ask("Porta-chaves", [("t", "Texto", "Forja"), ("shape", "Forma", "retângulo arredondado", ["retângulo arredondado"] + gn.SHAPES_2D),
                                      ("w", "Largura (mm)", 50.0), ("d", "Altura (mm)", 20.0), ("th", "Espessura (mm)", 3.0),
                                      ("tt", "Relevo do texto (mm)", 1.5), ("hole", "Furo (mm)", 4.0)])
        if v:
            tg = text3d.text_polygons(v["t"], bold=True) if v["t"] else None
            self._new(gn.keychain(tg, v["shape"], v["w"], v["d"], v["th"], v["tt"], v["hole"]), f"porta-chaves {v['t']}")

    def gen_cutter(self):
        g, v = self._shape_from_canvas_or_basic("Cortador", [("fit", "Tamanho máximo (mm, 0 = manter)", 70.0), ("h", "Altura (mm)", 15.0),
                                                              ("blade", "Espessura da lâmina (mm)", 0.8), ("fw", "Largura da aba (mm)", 4.0),
                                                              ("ft", "Espessura da aba (mm)", 1.6)])
        if g is not None:
            self._new(lambda: gn.cookie_cutter(g, v["h"], v["blade"], v["fw"], v["ft"], v["fit"] or None), "cortador")

    def gen_stamp(self):
        from ..core import text3d
        sel = [s for s in self.canvas.selected_shapes() if s.closed]
        v = self.ask("Carimbo", ([] if sel else [("t", "Texto", "FORJA")]) + [("d", "Relevo (mm)", 2.0), ("b", "Base (mm)", 4.0),
                                                                              ("m", "Margem (mm)", 3.0), ("hh", "Pega (mm)", 15.0)])
        if v:
            g = unary_union([s.geom for s in sel]) if sel else text3d.text_polygons(v["t"], size_mm=15, bold=True)
            self._new(gn.stamp(g, v["d"], v["b"], v["m"], v["hh"]), "carimbo")

    def gen_puzzle(self):
        v = self.ask("Puzzle", [("w", "Largura (mm)", 150.0), ("h", "Altura (mm)", 100.0), ("c", "Colunas", 4, (1, 50)), ("r", "Linhas", 3, (1, 50)),
                                ("t", "Espessura 3D (mm, 0 = só 2D para laser)", 3.0), ("gap", "Folga (mm)", 0.15), ("seed", "Variante", 1, (0, 9999))])
        if not v:
            return
        pieces = gn.puzzle_pieces(v["w"], v["h"], v["c"], v["r"], 0.2, v["seed"])
        shapes = [s2.Shape("geo", {}, p.buffer(-v["gap"] / 2), f"peça {k + 1}") for k, p in enumerate(pieces)]
        self.canvas.add_shapes(shapes)
        if v["t"] > 0:
            from ..core.text3d import extrude
            ms = []
            for k, p in enumerate(pieces):
                m = extrude(p.buffer(-v["gap"] / 2), v["t"]); m.apply_translation(((k % v["c"]) * 4, (k // v["c"]) * 4, 0)); ms.append(m)
            self._new(ms, "peça puzzle")
        else:
            self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()

    def _one(self):
        o = self.sel_objs(1, True)
        return o[0] if o else None

    def gen_mold(self):
        o = self._one()
        if not o:
            return
        v = self.ask("Molde", [("w", "Parede (mm)", 8.0), ("pin", "Raio dos pinos (mm)", 3.0), ("pour", "Diâmetro do canal (mm)", 8.0),
                               ("ax", "Divisão perpendicular a", "Z", ["X", "Y", "Z"])])
        if v:
            mesh = o.mesh.copy()
            def done(r):
                a, b = r
                b.apply_translation((0, 0, 0))
                self._new([a, b], f"molde {o.name}")
            self.task("A criar molde…", lambda: gn.two_part_mold(mesh, v["w"], v["pin"], v["pour"], axis="XYZ".index(v["ax"])), done)

    def gen_lowpoly(self):
        o = self._one()
        if not o:
            return
        v = self.ask("Low poly", [("r", "Fração de triângulos a manter", 0.05, (0.001, 1.0))])
        if v:
            mesh = o.mesh.copy()
            self._new(lambda: gn.low_poly(mesh, v["r"]), f"low poly {o.name}")

    def gen_wire(self):
        o = self._one()
        if not o:
            return
        v = self.ask("Estrutura em arame", [("s", "Raio das barras (mm)", 1.0), ("e", "Máximo de arestas", 1500, (10, 20000))])
        if v:
            mesh = o.mesh.copy()
            self._new(lambda: gn.wireframe(mesh, v["s"], v["e"]), f"arame {o.name}")

    def gen_stack(self):
        o = self._one()
        if not o:
            return
        v = self.ask("Pilha", [("n", "Cópias", 3, (2, 100)), ("g", "Intervalo (mm)", 0.3)])
        if v: self._new(gn.stack_copies(o.mesh.copy(), v["n"], v["g"]), f"pilha {o.name}")

    def gen_supports(self):
        o = self._one()
        if not o:
            return
        v = self.ask("Suportes", [("a", "Ângulo de saliência (°)", 45.0, (1, 89)), ("s", "Espaçamento (mm)", 4.0), ("r", "Raio na base (mm)", 0.8)])
        if not v:
            return
        mesh = o.mesh.copy()

        def done(m):
            if m is None:
                return self.statusBar().showMessage("Não há zonas que precisem de suporte.")
            s = self.add_mesh(m, f"suportes {o.name}", place=False)
            s.color = (0.6, 0.6, 0.6); self.view3d.refresh()
            self.log(f"Suportes: {len(m.split(only_watertight=False))} pilares")
        self.task("A calcular suportes…", lambda: gn.supports(mesh, v["a"], v["s"], v["r"]), done)

    def gen_magnets(self):
        o = self._one()
        if not o:
            return
        v = self.ask("Ímanes", [("d", "Diâmetro (mm, com folga)", 6.2), ("h", "Profundidade (mm)", 2.2), ("n", "Quantidade", 1, (1, 10)),
                                ("s", "Espaçamento (mm)", 20.0)])
        if v:
            self.doc.snapshot(); o.mesh = gn.magnet_holes(o.mesh, v["d"], v["h"], v["n"], v["s"]); self.doc.changed()
