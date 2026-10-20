"""Mais ações 3D: posição/orientação, cortes avançados, encaixes, relevo projetado, secção na vista, impressora."""
import numpy as np
import trimesh
from PySide6.QtCore import QSettings
from ..core import mesh2 as m2, meshops as mo, generators as gn, imageops as iop
from .widgets import TextDialog


class Mesh2Actions:
    def menus_mesh2(self):
        return [
            ("&Transformar", [
                None,
                ("Virar", [("‹esquerda, direita›", lambda: self.m2_flip(0)), ("‹frente, trás›", lambda: self.m2_flip(1)),
                           ("‹cima, baixo›", lambda: self.m2_flip(2))]),
                ("Rodar 90°", [("Em X", lambda: self._apply(lambda m: mo.rotate(m, 90, 0, 0), "Rodado")),
                               ("Em Y", lambda: self._apply(lambda m: mo.rotate(m, 0, 90, 0), "Rodado")),
                               ("Em Z", lambda: self._apply(lambda m: mo.rotate(m, 0, 0, 90), "Rodado"))]),
                ("Repor coordenadas (centro na origem)", lambda: self._apply(m2.reset_coords, "Coordenadas repostas")),
                ("Alinhar à caixa envolvente mínima", self.m2_align_min_box),
                None,
                ("Vista explodida…", self.m2_explode, "Ctrl+Shift+E"),
                ("Montar (desfazer explosão)", self.m2_assemble),
                ("Juntar dois objetos", [("‹esquerda, direita›", lambda: self.m2_join(0)), ("‹frente, trás›", lambda: self.m2_join(1)),
                                         ("‹cima, baixo›", lambda: self.m2_join(2))]),
                ("Embutir objeto 1 no objeto 2…", self.m2_bury),
            ]),
            ("&Malha", [
                None,
                ("Cortes avançados", [
                    ("Corte com encaixes (cavilha / pino / íman)…", self.m2_cut_connectors),
                    ("Corte radial (fatias de bolo)…", self.m2_radial),
                    ("Corte modular (pelo tamanho da impressora)…", self.m2_modular),
                    ("Corte natural (nos pontos mais estreitos)…", self.m2_natural),
                    ("Corte por região (caixa)…", self.m2_regional),
                    ("Corte desfasado (tijolos)…", self.m2_staggered),
                ]),
                ("Relevo projetado (tatuagem / gravado / embutido)…", self.m2_project_relief),
                ("Numerar peças (gravar número)…", self.m2_number_parts),
                ("Malha → mapa de altura (imagem)…", self.m2_heightmap),
            ]),
            ("&Ver", [
                None,
                ("Vista em corte (plano de secção)…", self.m2_section_view, "Ctrl+Shift+X"),
                ("Desligar vista em corte", self.m2_section_off),
            ]),
            ("Medi&r", [
                None,
                ("Caixa envolvente mínima (orientada)", self.m2_min_box_info),
                ("Informação das peças (lista)", self.m2_parts_info),
                ("Impressora: definições…", self.m2_printer),
                ("Impressora: cabe na mesa?", self.m2_fits),
                ("Impressora: filamento e tempo estimados…", self.m2_estimate),
                ("Impressora: distribuir por mesas…", self.m2_pack),
            ]),
            ("&Gerar", [
                None,
                ("Acessórios", [("Suporte em pé para foto/litofania…", self.m2_gen_stand), ("Gancho para pendurar…", self.m2_gen_hanger),
                                ("Garra / clip em U…", self.m2_gen_clip), ("Pilar decorativo…", self.m2_gen_pillar)]),
            ]),
        ]

    # --- impressora --------------------------------------------------------
    def printer(self):
        st = QSettings("Forja", "Forja")
        return tuple(float(st.value(f"printer/{k}", d)) for k, d in (("x", 220), ("y", 220), ("z", 250)))

    def m2_printer(self):
        x, y, z = self.printer()
        v = self.ask("Impressora", [("x", "Volume X (mm)", x, (10, 5000)), ("y", "Volume Y (mm)", y, (10, 5000)),
                                    ("z", "Volume Z (mm)", z, (10, 5000))])
        if v:
            st = QSettings("Forja", "Forja")
            for k in "xyz":
                st.setValue(f"printer/{k}", v[k])
            self.view3d.set_grid(max(v["x"], v["y"]))
            self.log(f"Impressora: {v['x']:g} × {v['y']:g} × {v['z']:g} mm")

    def m2_fits(self):
        objs = self.sel_objs()
        if not objs:
            return
        pr = self.printer()
        lines = []
        for o in objs:
            direct, rot = m2.fits_printer(o.mesh, pr)
            e = o.mesh.extents
            msg = "cabe" if direct else ("cabe se for rodado" if rot else "NÃO cabe — use Malha ▸ Cortes avançados ▸ Corte modular")
            lines.append(f"{o.name} ({e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f}): {msg}")
        self.log("\n".join(lines)); TextDialog.show_text(self, "Impressora", "\n".join(lines))

    def m2_estimate(self):
        objs = self.sel_objs()
        if not objs:
            return
        v = self.ask("Estimativa", [("inf", "Enchimento (%)", 20.0, (0, 100)), ("wall", "Espessura das paredes (mm)", 1.2),
                                    ("d", "Diâmetro do filamento (mm)", 1.75), ("rho", "Densidade (g/cm³, PLA 1.24, PETG 1.27, ABS 1.04)", 1.24),
                                    ("flow", "Caudal médio (mm³/s)", 8.0), ("lay", "Altura de camada (mm)", 0.2)])
        if not v:
            return
        tot = {}
        lines = []
        for o in objs:
            e = m2.print_estimate(o.mesh, v["inf"] / 100, v["wall"], v["d"], v["rho"], v["flow"], v["lay"])
            lines.append(f"{o.name}: {e['filamento (m)']:.2f} m, {e['peso (g)']:.1f} g, ≈ {e['tempo estimado (h)']:.2f} h")
            for k, val in e.items():
                tot[k] = tot.get(k, 0) + val
        lines.append(f"\nTotal: {tot['filamento (m)']:.2f} m de filamento, {tot['peso (g)']:.1f} g, ≈ {tot['tempo estimado (h)']:.2f} h "
                     f"(estimativa aproximada; o fatiador dá o valor exato)")
        self.log("\n".join(lines)); TextDialog.show_text(self, "Estimativa de impressão", "\n".join(lines))

    def m2_pack(self):
        objs = [o for o in self.doc.objects if o.visible and len(o.mesh.faces)]
        if not objs:
            return
        x, y, _ = self.printer()
        v = self.ask("Distribuir por mesas", [("gap", "Espaço entre peças (mm)", 5.0), ("o", "Ordem", "volume", ["volume", "área", "irregularidade"])],
                     note=f"Mesa de {x:g} × {y:g} mm (Medir ▸ Impressora: definições).")
        if not v:
            return
        meshes = [o.mesh for o in objs]
        res, un, plates = m2.pack_on_plates(meshes, (x, y), v["gap"], v["o"])
        self.doc.snapshot()
        from shapely.geometry import MultiPoint
        for i, p, g, a in res:
            m = meshes[i]
            if a:
                m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(a), [0, 0, 1], m.bounds.mean(axis=0)))
            hb = MultiPoint(m.vertices[:, :2]).convex_hull.bounds
            m.apply_translation((g.bounds[0] - hb[0] - x / 2 + p * (x + 30), g.bounds[1] - hb[1] - y / 2, -m.bounds[0][2]))
        self.doc.changed(); self.view3d.fit()
        self.log(f"{len(res)} peças em {len(plates)} mesa(s)" + (f"; não cabem: {[objs[i].name for i in un]}" if un else ""))

    # --- posição -------------------------------------------------------------
    def m2_flip(self, axis):
        self._apply(lambda m: m2.flip(m, axis), "Virado")

    def m2_align_min_box(self):
        self._apply(lambda m: m2.align_to_min_box(m)[0], "Alinhado à caixa mínima")

    def m2_explode(self):
        objs = self.sel_objs(2) if len([o for o in self.doc.objects]) > 1 else None
        if not objs:
            return
        v = self.ask("Vista explodida", [("f", "Afastamento (fração)", 0.5, (0.01, 10))])
        if v:
            self._explode_moves = getattr(self, "_explode_moves", {})
            moves = m2.explode([o.mesh for o in objs], v["f"])
            for o, d in zip(objs, moves):
                self._explode_moves[o.id] = self._explode_moves.get(o.id, np.zeros(3)) + d
            self.doc.changed(); self.view3d.fit()

    def m2_assemble(self):
        mv = getattr(self, "_explode_moves", {})
        for o in self.doc.objects:
            if o.id in mv:
                o.mesh.apply_translation(-mv[o.id])
        self._explode_moves = {}
        self.doc.changed(); self.view3d.fit()

    def m2_join(self, axis):
        objs = self.sel_objs(2, True)
        if objs:
            self.doc.snapshot()
            m = m2.join_along(objs[0].mesh, objs[1].mesh, axis)
            self.doc.remove(objs); self.add_mesh(m, f"{objs[0].name} + {objs[1].name}", place=False)

    def m2_bury(self):
        objs = self.sel_objs(2, True)
        if not objs:
            return
        v = self.ask("Embutir", [("d", "Profundidade (mm)", 2.0)], note=f"{objs[0].name} fica assente e enterrado no topo de {objs[1].name}.")
        if v:
            self.doc.snapshot()
            m = m2.bury(objs[0].mesh, objs[1].mesh, v["d"])
            self.doc.remove(objs); self.add_mesh(m, f"{objs[1].name} com {objs[0].name}", place=False)

    # --- cortes ------------------------------------------------------------
    def _one3d(self):
        o = self.sel_objs(1, True)
        return o[0] if o else None

    def m2_cut_connectors(self):
        o = self._one3d()
        if not o:
            return
        c = o.mesh.bounds.mean(axis=0)
        v = self.ask("Corte com encaixes", [("ax", "Normal do plano", "Z", ["X", "Y", "Z"]), ("pos", "Posição (mm)", float(round(c[2], 2))),
                                            ("k", "Encaixe", "cavilha", ["cavilha", "pino", "íman"]), ("r", "Raio (mm)", 3.0),
                                            ("d", "Profundidade (mm)", 8.0), ("cl", "Folga (mm)", 0.2), ("n", "Quantidade", 2, (1, 10)),
                                            ("gap", "Afastar as partes (mm)", 10.0)],
                     note="Para ímanes, use o raio do íman + folga (p.ex. 3,1 mm para ímanes de 6 mm).")
        if not v:
            return
        ax = "XYZ".index(v["ax"]); n = np.zeros(3); n[ax] = 1; org = c.copy(); org[ax] = v["pos"]
        mesh = o.mesh.copy()

        def done(r):
            P, N, extras = r
            self.doc.snapshot()
            P.apply_translation(n * v["gap"] / 2); N.apply_translation(-n * v["gap"] / 2)
            self._replace_with(o, [N, P], "parte ")
            for e in extras:
                self.add_mesh(e, "cavilhas")
        self.task("A cortar…", lambda: m2.cut_with_connectors(mesh, org, n, v["k"], v["r"], v["d"], v["cl"], v["n"]), done)

    def m2_radial(self):
        o = self._one3d()
        if not o:
            return
        v = self.ask("Corte radial", [("n", "Número de fatias", 4, (2, 72)), ("a", "Ângulo inicial (°)", 0.0), ("gap", "Afastar (mm)", 6.0)])
        if not v:
            return
        mesh = o.mesh.copy()

        def done(parts):
            self.doc.snapshot()
            c = mesh.bounds.mean(axis=0)
            for p in parts:
                d = p.bounds.mean(axis=0) - c; d[2] = 0
                if np.linalg.norm(d):
                    p.apply_translation(d / np.linalg.norm(d) * v["gap"])
            self._replace_with(o, parts, "fatia ")
        self.task("A cortar…", lambda: m2.radial_cut(mesh, v["n"], v["a"]), done)

    def m2_modular(self):
        o = self._one3d()
        if not o:
            return
        pr = self.printer()
        nx, ny, nz = m2.modular_counts(o.mesh, pr)
        v = self.ask("Corte modular", [("m", "Margem de segurança (mm)", 5.0), ("con", "Encaixes", True), ("r", "Raio do pino (mm)", 3.0)],
                     note=f"Impressora {pr[0]:g}×{pr[1]:g}×{pr[2]:g} mm → {nx}×{ny}×{nz} módulos para {o.name}.")
        if not v:
            return
        nx, ny, nz = m2.modular_counts(o.mesh, pr, v["m"])
        mesh = o.mesh.copy()

        def done(parts):
            self.doc.snapshot()
            ms = []
            for (i, j, k), m in parts:
                m.apply_translation((i * 10, j * 10, k * 10)); ms.append(m)
            self._replace_with(o, ms, "módulo ")
            self.log(f"Corte modular: {len(ms)} módulos ({nx}×{ny}×{nz})")
        self.task("A cortar…", lambda: mo.split_grid(mesh, nx, ny, nz, v["con"], v["r"]), done)

    def m2_natural(self):
        o = self._one3d()
        if not o:
            return
        v = self.ask("Corte natural", [("ax", "Eixo", "Z", ["X", "Y", "Z"]), ("n", "Número de cortes", 1, (1, 20)), ("gap", "Afastar (mm)", 6.0)],
                     note="Procura as secções mais estreitas (pescoços) para os cortes ficarem discretos e fáceis de colar.")
        if not v:
            return
        ax = "XYZ".index(v["ax"]); mesh = o.mesh.copy()

        def work():
            pos, _, _ = m2.natural_cut_positions(mesh, ax, v["n"])
            return pos, split_by_planes_safe(mesh, ax, pos)

        def done(r):
            pos, pieces = r
            self.doc.snapshot()
            for i, p in enumerate(pieces):
                d = np.zeros(3); d[ax] = i * v["gap"]; p.apply_translation(d)
            self._replace_with(o, pieces, "parte ")
            self.log(f"Corte natural em {v['ax']} = " + ", ".join(f"{p:.2f}" for p in pos))
        from ..core.meshops import split_by_planes as split_by_planes_safe
        self.task("A procurar pontos estreitos…", work, done)

    def m2_regional(self):
        o = self._one3d()
        if not o:
            return
        b = o.mesh.bounds
        c = b.mean(axis=0)
        v = self.ask("Corte por região", [("x0", "X mín", float(round(b[0][0], 2))), ("x1", "X máx", float(round(c[0], 2))),
                                          ("y0", "Y mín", float(round(b[0][1], 2))), ("y1", "Y máx", float(round(b[1][1], 2))),
                                          ("z0", "Z mín", float(round(b[0][2], 2))), ("z1", "Z máx", float(round(b[1][2], 2)))])
        if v:
            mesh = o.mesh.copy()
            def done(r):
                a, rest = r
                self.doc.snapshot()
                a.apply_translation((0, 0, 0)); rest.apply_translation((0, 0, 0))
                self._replace_with(o, [m for m in (a, rest) if len(m.faces)], "região ")
            self.task("A cortar…", lambda: m2.regional_cut(mesh, [v["x0"], v["y0"], v["z0"]], [v["x1"], v["y1"], v["z1"]]), done)

    def m2_staggered(self):
        o = self._one3d()
        if not o:
            return
        e = o.mesh.extents
        v = self.ask("Corte desfasado", [("cx", "Largura do bloco (mm)", float(round(e[0] / 2, 1))), ("cy", "Profundidade do bloco (mm)", float(round(e[1], 1))),
                                         ("h", "Altura da camada (mm)", float(round(e[2] / 2, 1)))])
        if v:
            mesh = o.mesh.copy()
            self.task("A cortar…", lambda: m2.staggered_cut(mesh, v["cx"], v["cy"], v["h"]),
                      lambda parts: (self.doc.snapshot(), self._replace_with(o, parts, "bloco ")))

    # --- relevo / números / mapa de altura --------------------------------
    def m2_project_relief(self):
        o = self._one3d()
        if not o:
            return
        it = self.cur_image() if self.images else None
        b = o.mesh.bounds; c = b.mean(axis=0); e = o.mesh.extents
        v = self.ask("Relevo projetado", [("src", "Fonte", "imagem atual" if it else "texto", ["imagem atual", "texto"]),
                                          ("t", "Texto (se fonte = texto)", "Forja 3D"),
                                          ("dir", "Projetar de", "cima", ["cima", "frente", "lado"]),
                                          ("u", "Centro horizontal (mm)", float(round(c[0], 2))), ("v", "Centro vertical (mm)", float(round(c[1], 2))),
                                          ("w", "Largura (mm)", float(round(min(e[0], e[1]) * 0.6, 1))), ("h", "Altura (mm)", float(round(min(e[0], e[1]) * 0.6, 1))),
                                          ("d", "Profundidade (mm, + relevo, − gravado)", 1.0), ("inv", "Claro = alto", False)],
                     note="Para 'frente' o centro vertical é Z; para 'lado' o horizontal é Y.")
        if not v:
            return
        if v["src"] == "texto":
            import cv2
            img = np.full((200, 200 * max(1, len(v["t"]) // 3 + 1)), 255, np.uint8)
            cv2.putText(img, v["t"], (10, 140), cv2.FONT_HERSHEY_DUPLEX, 4, 0, 10)
            gray = img
        else:
            if not it:
                return self.error("Abra ou selecione uma imagem.")
            gray = iop.to_gray(it.data) if it.data.dtype == np.uint8 else iop.to_u8(it.data)
        mesh = o.mesh.copy()

        def done(m):
            self.doc.snapshot(); o.mesh = m; self.doc.changed()
            self.log(f"Relevo projetado em {o.name}: {len(m.faces)} faces")
        self.task("A projetar…", lambda: m2.project_relief(mesh, gray, v["dir"], (v["u"], v["v"]), (v["w"], v["h"]), v["d"], invert=v["inv"]), done)

    def m2_number_parts(self):
        from ..core import text3d
        objs = self.sel_objs()
        if not objs:
            return
        v = self.ask("Numerar peças", [("s", "Altura dos números (mm)", 6.0), ("d", "Profundidade (mm)", 0.6), ("p", "Prefixo", ""),
                                       ("start", "Começar em", 1, (0, 100000))])
        if not v:
            return
        self.doc.snapshot()
        for k, o in enumerate(objs):
            label = f"{v['p']}{v['start'] + k}"
            t = text3d.text_mesh(label, v["s"], v["d"] * 2)
            t.apply_translation(-t.bounds.mean(axis=0))
            b = o.mesh.bounds; c = b.mean(axis=0)
            t.apply_translation((c[0], c[1], b[0][2] + v["d"] - t.extents[2] / 2))
            t = mo.mirror(t, 0)  # lê-se corretamente vendo a peça por baixo
            try:
                o.mesh = mo.boolean(o.mesh, t, "difference")
                o.name = f"{o.name} [{label}]"
            except Exception as e:
                self.log(f"Não foi possível numerar {o.name}: {e}")
        self.doc.changed()

    def m2_heightmap(self):
        o = self._one3d()
        if not o:
            return
        v = self.ask("Mapa de altura", [("px", "Tamanho do pixel (mm)", float(round(o.mesh.scale / 300, 3)), (0.001, 100)),
                                        ("dir", "Vista de", "cima", ["cima", "frente", "lado"])])
        if v:
            mesh = o.mesh
            self.task("A calcular…", lambda: m2.mesh_heightmap(mesh, v["px"], v["dir"]),
                      lambda r: (self.add_image(f"altura({o.name}) {r[1]:.3g} mm/px", r[0], {"px_mm": r[1]}),
                                 self.log(f"Mapa de altura com {r[1]:.4g} mm por pixel")))

    # --- secção na vista ---------------------------------------------------
    def m2_section_view(self):
        b = self.doc.bounds(); c = b.mean(axis=0)
        v = self.ask("Vista em corte", [("ax", "Normal", "Y", ["X", "Y", "Z"]), ("pos", "Posição (mm)", float(round(c[1], 2))),
                                        ("side", "Mostrar", "lado negativo", ["lado negativo", "lado positivo"])])
        if v:
            ax = "XYZ".index(v["ax"]); n = np.zeros(3); n[ax] = 1 if v["side"] == "lado negativo" else -1
            o = np.zeros(3); o[ax] = v["pos"]
            self.view3d.clip = (o, n)
            self.view3d.refresh()
            # câmara do lado da face cortada
            pos_side = v["side"] == "lado negativo"
            if ax == 2:
                self.view3d.setCameraPosition(elevation=60 if pos_side else -60, azimuth=-60)
            else:
                az = (0 if ax == 0 else 90) + (0 if pos_side else 180)
                self.view3d.setCameraPosition(elevation=25, azimuth=az + 20)
            self.tabs.setCurrentWidget(self.view3d)
            self.statusBar().showMessage(f"Vista em corte: {v['ax']} = {v['pos']}  (Ver ▸ Desligar vista em corte)")

    def m2_section_off(self):
        self.view3d.clip = None; self.view3d.refresh()

    # --- informação --------------------------------------------------------
    def m2_min_box_info(self):
        objs = self.sel_objs()
        if objs:
            lines = []
            for o in objs:
                T, ext = m2.oriented_box(o.mesh)
                e = o.mesh.extents
                lines.append(f"{o.name}: caixa mínima {ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f} mm "
                             f"(alinhada aos eixos: {e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f})")
            self.log("\n".join(lines)); TextDialog.show_text(self, "Caixa mínima", "\n".join(lines))

    def m2_parts_info(self):
        lines = ["N.º  Nome                          X×Y×Z (mm)                Volume (cm³)  Fechada"]
        for k, o in enumerate(self.doc.objects, 1):
            e = o.mesh.extents if len(o.mesh.faces) else np.zeros(3)
            vol = abs(o.mesh.volume) / 1000 if o.mesh.is_watertight else float("nan")
            lines.append(f"{k:<4} {o.name[:28]:<29} {e[0]:7.2f}×{e[1]:7.2f}×{e[2]:7.2f}   {vol:10.3f}   {'sim' if o.mesh.is_watertight else 'não'}")
        TextDialog.show_text(self, "Peças", "\n".join(lines))

    # --- acessórios --------------------------------------------------------
    def m2_gen_stand(self):
        v = self.ask("Suporte em pé", [("w", "Largura da foto (mm)", 100.0), ("t", "Espessura da foto (mm)", 3.0), ("a", "Inclinação (°)", 10.0, (0, 45))])
        if v: self._new(gn.photo_stand(v["w"], v["t"], angle=v["a"]), "suporte de foto")

    def m2_gen_hanger(self):
        v = self.ask("Gancho", [("w", "Largura (mm)", 30.0), ("t", "Espessura (mm)", 3.0), ("h", "Furo (mm)", 5.0)])
        if v: self._new(gn.hanger(v["w"], v["t"], v["h"]), "gancho")

    def m2_gen_clip(self):
        v = self.ask("Garra", [("t", "Espessura a prender (mm)", 3.0), ("l", "Comprimento (mm)", 25.0), ("o", "Folga (mm)", 0.4)])
        if v: self._new(gn.claw_clip(v["t"], v["l"], v["o"]), "garra")

    def m2_gen_pillar(self):
        v = self.ask("Pilar", [("d", "Diâmetro (mm)", 20.0), ("h", "Altura (mm)", 80.0), ("f", "Caneluras", 12, (0, 60))])
        if v: self._new(gn.pillar(v["d"], v["h"], v["f"]), "pilar")
