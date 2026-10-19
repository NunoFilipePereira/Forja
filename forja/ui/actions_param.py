"""Criação de objetos paramétricos (ficam ajustáveis no painel Parâmetros)."""
import numpy as np
from PySide6.QtCore import Qt
from ..core import params as P, imageops as iop
from ..core.document import mbounds
from .parampanel import ParamPanel, new_gen

OPTIONAL_GEOM = {"cortador", "carimbo", "extrusao_torcida"}


class ParamActions:
    def setup_param_panel(self):
        from PySide6.QtWidgets import QStackedWidget
        from .cutpanel import CutPanel
        self.param_panel = ParamPanel(self)
        from .toolpanel import ToolPanel
        self.cut_panel = CutPanel(self)
        self.tool_panel = ToolPanel(self)
        self.side_stack = QStackedWidget()
        self.side_stack.addWidget(self.param_panel); self.side_stack.addWidget(self.cut_panel); self.side_stack.addWidget(self.tool_panel)
        # o painel da ferramenta fica à esquerda da vista 3D (separador junto a "Objetos 3D")
        self.param_dock = self._dock("Ferramenta / Parâmetros", self.side_stack, Qt.LeftDockWidgetArea)
        self.param_dock.setMinimumWidth(340)
        if hasattr(self, "obj_dock"):
            self.tabifyDockWidget(self.obj_dock, self.param_dock)
            if hasattr(self, "img_dock"):
                self.tabifyDockWidget(self.obj_dock, self.img_dock)
            self.obj_dock.raise_()
        if hasattr(self, "log_dock"):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.resizeDocks([self.log_dock], [110], Qt.Vertical))
        from .batch_panel import BatchPanel
        self.batch_panel = BatchPanel(self)
        self.batch_dock = self._dock("Lote", self.batch_panel, Qt.LeftDockWidgetArea)
        self.batch_dock.hide()

    def menus_batch(self):
        bp = self.batch_panel

        def show(fn):
            def f():
                self.batch_dock.show(); self.batch_dock.raise_(); fn()
            return f
        return [("&Lote", [
            ("Carregar ficheiros…", show(bp.load_files)),
            ("Carregar pasta…", show(bp.load_folder)),
            None,
            ("Primeiro", show(bp.first), "Alt+Home"), ("Último", show(bp.last), "Alt+End"),
            ("Página acima", show(bp.page_up), "Alt+PgUp"), ("Página abaixo", show(bp.page_down), "Alt+PgDown"),
            ("Anterior", show(bp.prev), "Alt+Up"), ("Seguinte", show(bp.next), "Alt+Down"),
            None,
            ("Ficheiros", [("Listar nomes", show(bp.list_names)), ("Renomear…", show(bp.rename)),
                           ("Apagar (para o Lixo)", show(bp.delete))]),
            ("Exportar todos…", show(bp.export_all)),
            ("Operação em lote (malha / objeto)…", show(bp.batch_op)),
            ("Mostrar o painel Lote", lambda: (self.batch_dock.show(), self.batch_dock.raise_())),
        ])]

    def open_tool(self, spec, need_obj=True, key=None, obj=None, edit=None):
        if need_obj and obj is None:
            objs = self.sel_objs(1, True)
            if not objs:
                return
            obj = objs[0]
            # escolher uma peça já feita por esta ferramenta reabre-a com os valores usados (sobre o original)
            t = getattr(obj, "tool", None)
            if t and t.get("kind") == "tool" and t.get("key") == key and edit is None:
                return self.reopen_tool(obj)
        self.side_stack.setCurrentWidget(self.tool_panel)
        self._show_tool_dock()
        self.tabs.setCurrentWidget(self.view3d)
        self.tool_panel.start(spec, obj, key=key, edit=edit)

    @staticmethod
    def get_spec(key):
        """Ferramenta a partir do nome guardado nas peças ("conv", "create:LITHO", …)."""
        if key == "conv":
            from .conv_tool import SPEC
            return SPEC
        if key and key.startswith("create:"):
            from . import create_tools as CT
            return getattr(CT, key.split(":", 1)[1], None)
        return None

    def reopen_tool(self, o):
        """Reabre a ferramenta com que a peça foi feita, com os mesmos valores, para ajustar (o resultado é substituído)."""
        t = getattr(o, "tool", None)
        if not t:
            return
        if t.get("kind") == "cut":
            if self.side_stack.currentWidget() is self.cut_panel and self.cut_panel.edit_group == t["group"] and self.cut_panel.obj is not None:
                return self._show_tool_dock()
            self.side_stack.setCurrentWidget(self.cut_panel)
            self._show_tool_dock()
            self.cut_panel.start(None, edit=t)
            return
        tp = self.tool_panel
        if self.side_stack.currentWidget() is tp and tp.spec is not None and getattr(tp, "edit_group", None) == t["group"]:
            return self._show_tool_dock()
        spec = self.get_spec(t.get("key"))
        if spec is None:
            return
        src = next((x for x in self.doc.objects if x.id == t.get("source_id")), None)
        if t.get("source_id") is not None and src is None:
            return self.statusBar().showMessage("O objeto original desta peça já não existe — não é possível reajustar.", 6000)
        if t.get("key", "").startswith("create:"):
            from ..core import cimg
            it = self.imgview.item
            cimg.CURRENT["rgb"] = np.asarray(it.data) if it is not None else None
        self.open_tool(spec, need_obj=False, key=t.get("key"), obj=src, edit=t)

    def _show_tool_dock(self):
        d = self.param_dock
        if d.isFloating():
            d.setFloating(False)
        d.show(); d.raise_()
        try:
            self.resizeDocks([d], [max(d.width(), 360)], Qt.Horizontal)
        except Exception:
            pass

    def act_convert(self):
        from .conv_tool import SPEC, run
        it = self.imgview.item
        run.image = None
        if it is not None:
            from ..core import imageops as iop
            a = np.asarray(it.data)
            run.image = iop.to_gray(a) if a.ndim == 3 else iop.to_u8(a)
        self.open_tool(SPEC, key="conv")

    def act_create(self, name):
        """Ferramentas do menu Criar (sem precisar de objeto selecionado)."""
        from . import create_tools as CT
        from ..core import cimg
        it = self.imgview.item
        cimg.CURRENT["rgb"] = np.asarray(it.data) if it is not None else None
        sel = [o for o in self.doc.objects if o.id in self.view3d.selected_ids]
        cimg.CURRENT["mesh"] = sel[0].mesh.copy() if sel else None
        self.open_tool(getattr(CT, name), need_obj=False, key=f"create:{name}")

    # --- Nesting 3D, Transformar e Alinhar -------------------------
    def act_nesting3d(self):
        """Arruma as peças selecionadas (ou todas as visíveis) dentro de caixas X × Y × Z."""
        from ..core import nesting3d as N3
        from ..core import meshops as mo
        rows = sorted(i.row() for i in self.obj_list.selectedIndexes())
        objs = [self.doc.objects[r] for r in rows if r < len(self.doc.objects)] or [o for o in self.doc.objects if o.visible]
        if not objs:
            return self.error("Não há peças para arrumar: abra ou crie objetos 3D primeiro.")
        v = self.ask("Nesting 3D (arrumar peças em caixas)", [
            ("seq", "Sequência", "Volume descending", N3.SEQUENCES),
            ("bx", "Caixa X (mm)", 300.0), ("by", "Caixa Y (mm)", 200.0), ("bz", "Caixa Z (mm)", 250.0),
            ("gb", "Folga peça–caixa (mm)", 0.0), ("gp", "Folga peça–peça (mm)", 0.0),
            ("rot", "Permitir rodar 90° em Z", True), ("export", "Exportar cada caixa para uma pasta (STL)", False)],
            note="As peças são arrumadas pela caixa envolvente, de baixo para cima.")
        if not v:
            return
        B = np.array([v["bx"], v["by"], v["bz"]])
        sizes = [o.mesh.extents for o in objs]
        res, nb = N3.pack(sizes, B, v["gb"], v["gp"], v["seq"], v["rot"])
        self.doc.snapshot()
        bins = {}
        miss = []
        for o, (b, p, r) in zip(objs, res):
            if b < 0:
                miss.append(o.name); continue
            m = o.mesh.copy()
            if r:
                m = mo.rotate(m, 0, 0, 90)
            lo = mbounds(m)[0]
            m.apply_translation(np.array([b * (B[0] + 20), 0, 0]) + p - lo)
            o.mesh = m
            bins.setdefault(b, []).append(o)
        self.doc.changed()
        msg = f"Nesting 3D: {len(objs) - len(miss)} peça(s) em {nb} caixa(s) de {B[0]:.0f}×{B[1]:.0f}×{B[2]:.0f} mm"
        if miss:
            msg += f"; não cabem: {', '.join(miss)}"
        self.log(msg)
        if v["export"] and bins:
            import os
            import trimesh
            from PySide6.QtWidgets import QFileDialog
            d = QFileDialog.getExistingDirectory(self, "Pasta para exportar as caixas", self.last_dir)
            if d:
                for b, os_ in sorted(bins.items()):
                    mm = trimesh.util.concatenate([o.mesh for o in os_])
                    mm.apply_translation((-b * (B[0] + 20), 0, 0))
                    mm.export(os.path.join(d, f"caixa_{b + 1}.stl"))
                self.log(f"Caixas exportadas para {d}")

    def act_transform_panel(self):
        """Transformar: rodar, deslocar, escala ou tamanho (uniforme ou não) e espelhar."""
        from ..core import meshops as mo
        objs = self.sel_objs()
        if not objs:
            return
        e = objs[0].mesh.extents
        v = self.ask("Transformar", [
            ("rx", "Rodar em X (°)", 0.0), ("ry", "Rodar em Y (°)", 0.0), ("rz", "Rodar em Z (°)", 0.0),
            ("tx", "Deslocar X (mm)", 0.0), ("ty", "Deslocar Y (mm)", 0.0), ("tz", "Deslocar Z (mm)", 0.0),
            ("mode", "Modo", "Escala", ["Escala", "Tamanho"]), ("uni", "Uniforme", True),
            ("k", "Escala X, Y, Z (uniforme)", 1.0), ("kx", "Escala X", 1.0), ("ky", "Escala Y", 1.0), ("kz", "Escala Z", 1.0),
            ("sx", "Tamanho X (mm)", float(e[0])), ("sy", "Tamanho Y (mm)", float(e[1])), ("sz", "Tamanho Z (mm)", float(e[2])),
            ("mx", "Espelhar X", False), ("my", "Espelhar Y", False), ("mz", "Espelhar Z", False)])
        if not v:
            return

        def fn(m):
            if v["rx"] or v["ry"] or v["rz"]:
                m = mo.rotate(m, v["rx"], v["ry"], v["rz"])
            ex = np.maximum(m.extents, 1e-9)
            if v["mode"] == "Escala":
                k = [v["k"]] * 3 if v["uni"] else [v["kx"], v["ky"], v["kz"]]
            else:
                want = np.array([v["sx"], v["sy"], v["sz"]])
                k = list(want / ex)
                if v["uni"]:
                    ch = [i for i in range(3) if abs(want[i] - e[i]) > 1e-6]
                    k = [k[ch[0]] if ch else 1.0] * 3
            if any(abs(x - 1) > 1e-12 for x in k):
                m = mo.scale(m, *k)
            for ax, key in enumerate(("mx", "my", "mz")):
                if v[key]:
                    m = mo.mirror(m, ax)
            m.apply_translation((v["tx"], v["ty"], v["tz"]))
            return m
        self._apply(fn, "Transformado")

    def act_align_objs(self):
        """Alinhar objetos: alinha os selecionados pelo primeiro."""
        objs = self.sel_objs(2)
        if not objs:
            return
        v = self.ask("Alinhar objetos", [("x", "Em X", "centro", ["não", "mínimo", "centro", "máximo"]),
                                         ("y", "Em Y", "centro", ["não", "mínimo", "centro", "máximo"]),
                                         ("z", "Em Z", "mínimo", ["não", "mínimo", "centro", "máximo"])],
                     note="O primeiro objeto selecionado serve de referência.")
        if not v:
            return
        ref = mbounds(objs[0].mesh)
        self.doc.snapshot()
        for o in objs[1:]:
            b = mbounds(o.mesh)
            d = np.zeros(3)
            for i, k in enumerate("xyz"):
                if v[k] == "mínimo":
                    d[i] = ref[0][i] - b[0][i]
                elif v[k] == "máximo":
                    d[i] = ref[1][i] - b[1][i]
                elif v[k] == "centro":
                    d[i] = (ref[0][i] + ref[1][i]) / 2 - (b[0][i] + b[1][i]) / 2
            o.mesh = o.mesh.copy(); o.mesh.apply_translation(d)
        self.doc.changed()
        self.log(f"Alinhados {len(objs) - 1} objeto(s) por {objs[0].name}")

    # --- Malha: ferramentas com janela própria -------------------------------------
    def act_mesh_depth(self):
        """Mapa de profundidade: direção e largura em píxeis, ex. "+x 500"."""
        import re
        import trimesh
        from ..core.mesh2 import mesh_heightmap
        objs = self.sel_objs(1, True)
        if not objs:
            return
        v = self.ask("Mapa de profundidade", [("d", "Direção e resolução", "+x 500")],
                     note="Exemplos: +x 300 (direção +x, 300 píxeis de largura), -y 400, +z 700.")
        if not v:
            return
        mt = re.match(r"\s*([+-])\s*([xyzXYZ])\s+(\d+)", v["d"])
        if not mt:
            return self.error("Escreva a direção e a largura, por exemplo: +x 500")
        sgn, ax, w = mt.group(1), "xyz".index(mt.group(2).lower()), int(mt.group(3))
        m = objs[0].mesh.copy()
        # roda o modelo para que a direção pedida fique a apontar para +Z (vista de cima)
        R = {0: trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]), 1: trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]),
             2: np.eye(4)}[ax]
        m.apply_transform(R)
        if sgn == "-":
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        px = max(m.extents[0], m.extents[1]) / max(w, 8)
        Z, px = mesh_heightmap(m, px, "cima", max_px=max(w, 8))
        ok = np.isfinite(Z)
        img = np.zeros(Z.shape, np.uint8)
        if ok.any():
            z0, z1 = np.nanmin(Z), np.nanmax(Z)
            img[ok] = np.clip(40 + 215 * (Z[ok] - z0) / max(z1 - z0, 1e-9), 0, 255).astype(np.uint8)
        self.add_image(f"{objs[0].name} profundidade {v['d'].strip()}", img, {"pitch": px})
        self.log(f"Mapa de profundidade {v['d'].strip()}: {img.shape[1]} × {img.shape[0]} píxeis ({px:.3f} mm/píxel)")

    def act_hole_fill(self):
        """Preencher buracos."""
        import trimesh
        objs = self.sel_objs()
        if not objs:
            return
        self.doc.snapshot()
        msgs = []
        for o in objs:
            m = o.mesh.copy()
            if m.is_watertight:
                msgs.append(f"{o.name}: não foram encontrados buracos na malha.")
                continue
            before = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
            trimesh.repair.fill_holes(m)
            if not m.is_watertight:
                try:
                    from ..core import meshops as mo
                    m, _ = mo.repair(m)
                except Exception:
                    pass
            after = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
            o.mesh = m
            msgs.append(f"{o.name}: {before} arestas abertas → {after}" + (" (fechada)" if m.is_watertight else ""))
        self.doc.changed()
        self.log("Preencher buracos — " + "; ".join(msgs))
        if len(msgs) == 1 and "não foram" in msgs[0]:
            self.statusBar().showMessage(msgs[0], 6000)

    def act_mesh_stats(self):
        """Estatísticas da malha."""
        from .widgets import TextDialog
        objs = self.sel_objs()
        if not objs:
            return
        out = []
        for o in objs:
            m = o.mesh
            lo, hi = m.bounds; e = hi - lo
            vol = abs(m.volume) if m.is_volume else float("nan")
            c = m.center_mass if m.is_volume else m.centroid
            out.append(f"{o.name}\n"
                       f"Caixa envolvente X: {e[0]:.4g}, [{lo[0]:.4g}, {hi[0]:.4g}]\n"
                       f"Caixa envolvente Y: {e[1]:.4g}, [{lo[1]:.4g}, {hi[1]:.4g}]\n"
                       f"Caixa envolvente Z: {e[2]:.4g}, [{lo[2]:.4g}, {hi[2]:.4g}]\n\n"
                       f"Área da superfície: {m.area:.4g} mm²\n"
                       f"Volume da caixa: {np.prod(e):.4g} mm³\n"
                       f"Volume da malha: {vol:.4g} mm³\n"
                       f"Centroide: ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})\n\n"
                       f"Número de vértices: {len(m.vertices)}\n"
                       f"Número de triângulos: {len(m.faces)}\n"
                       f"Fechada (estanque): {'sim' if m.is_watertight else 'não'}")
        TextDialog.show_text(self, "Estatísticas", "\n\n".join(out))

    def act_zero_xyz(self):
        """Zero XYZ: leva o canto mínimo da caixa envolvente para a origem."""
        def fn(m):
            m.apply_translation(-m.bounds[0]); return m
        self._apply(fn, "Zero XYZ")

    def act_min_support(self):
        """Suporte mínimo: roda o modelo para a orientação com menos área a precisar
        de suporte (faces inclinadas mais de 45° para baixo, fora da mesa)."""
        import trimesh
        objs = self.sel_objs(1, True)
        if not objs:
            return
        o = objs[0]
        mesh = o.mesh.copy()

        def overhang(m):
            n = m.face_normals; a = m.area_faces
            zc = m.triangles_center[:, 2]
            down = n[:, 2] < -np.cos(np.radians(45))
            on_bed = zc < m.bounds[0][2] + 0.2
            return float(a[down & ~on_bed].sum())

        def work():
            cands = []
            dirs = [np.array(d, float) for d in ((0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
            h = mesh.convex_hull
            order = np.argsort(-h.area_faces)[:40]
            dirs += [h.face_normals[i] for i in order]
            best = None
            cur = overhang(mesh)
            for d in dirs:
                d = d / np.linalg.norm(d)
                R = trimesh.geometry.align_vectors(d, [0, 0, -1])
                m = mesh.copy(); m.apply_transform(R)
                val = overhang(m)
                cands.append(val)
                if best is None or val < best[0] - 1e-6:
                    best = (val, R)
            return cur, best

        def done(r):
            cur, (val, R) = r
            if val >= cur * 0.97:
                self.statusBar().showMessage("Não é preciso mudar a orientação.", 6000)
                self.log(f"Suporte mínimo: a orientação atual já é a melhor ({cur:.0f} mm² a suportar).")
                return
            m = mesh.copy(); m.apply_transform(R); m.apply_translation((0, 0, -m.bounds[0][2]))
            self.doc.snapshot(); o.mesh = m; self.doc.changed()
            self.log(f"Suporte mínimo: área a suportar {cur:.0f} → {val:.0f} mm²")
        self.task("A procurar a melhor orientação…", work, done)

    # --- Lote -------------------------------------------------------------
    BATCH_OPS = ["Simplificar", "Separar em componentes", "Estatísticas (CSV)", "Zero XYZ", "Suporte mínimo", "Reescalar",
                 "Rodar X", "Rodar Y", "Rodar Z", "Deslocar", "Converter formato"]

    def act_batch(self):
        """Aplica uma operação a todos os modelos de uma pasta e grava o resultado noutra pasta."""
        import os, glob
        from PySide6.QtWidgets import QFileDialog
        src = QFileDialog.getExistingDirectory(self, "Pasta com os modelos (STL, OBJ, PLY, OFF, 3MF)", self.last_dir)
        if not src:
            return
        files = sorted(f for ext in ("stl", "obj", "ply", "off", "3mf", "glb") for f in glob.glob(os.path.join(src, f"*.{ext}")) +
                       glob.glob(os.path.join(src, f"*.{ext.upper()}")))
        if not files:
            return self.error("A pasta não tem modelos 3D.")
        self.run_batch(files, src)

    def run_batch(self, files, src=None):
        """Operação em lote sobre uma lista de modelos (usada pelo menu e pelo painel Lote)."""
        import os, csv
        import trimesh
        from PySide6.QtWidgets import QFileDialog
        from ..core import meshops as mo
        src = src or os.path.dirname(files[0])
        v = self.ask(f"Lote — {len(files)} modelo(s)", [
            ("op", "Operação", "Simplificar", self.BATCH_OPS),
            ("r", "Simplificar: fração a reduzir", 0.3, (0.0, 0.999)),
            ("k", "Reescalar: fator", 1.0, (1e-4, 1e4)), ("a", "Rodar: ângulo (°)", 90.0, (-360, 360)),
            ("dx", "Deslocar X (mm)", 0.0), ("dy", "Deslocar Y (mm)", 0.0), ("dz", "Deslocar Z (mm)", 0.0),
            ("fmt", "Formato de saída", "stl", ["stl", "obj", "ply", "off", "3mf"])],
            note="A mesma operação em todos os ficheiros da pasta.")
        if not v:
            return
        dst = QFileDialog.getExistingDirectory(self, "Pasta para os resultados", src)
        if not dst:
            return

        def min_support(m):
            best = None
            for d in [(0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
                R = trimesh.geometry.align_vectors(d, [0, 0, -1]); mm = m.copy(); mm.apply_transform(R)
                n = mm.face_normals; down = n[:, 2] < -0.7071; bed = mm.triangles_center[:, 2] < mm.bounds[0][2] + 0.2
                val = mm.area_faces[down & ~bed].sum()
                if best is None or val < best[0]:
                    best = (val, mm)
            m2 = best[1]; m2.apply_translation((0, 0, -m2.bounds[0][2])); return m2

        def work():
            rows, done = [], 0
            for f in files:
                m = trimesh.load(f, force="mesh")
                base = os.path.splitext(os.path.basename(f))[0]
                op = v["op"]
                outs = [m]
                if op == "Simplificar":
                    outs = [mo.simplify(m, len(m.faces) * (1 - v["r"]))]
                elif op == "Separar em componentes":
                    outs = mo.separate(m)
                elif op == "Estatísticas (CSV)":
                    e = m.extents
                    rows.append([base, f"{e[0]:.3f}", f"{e[1]:.3f}", f"{e[2]:.3f}", f"{m.area:.2f}",
                                 f"{abs(m.volume) if m.is_volume else float('nan'):.2f}", len(m.vertices), len(m.faces), m.is_watertight])
                    outs = []
                elif op == "Zero XYZ":
                    m.apply_translation(-m.bounds[0])
                elif op == "Suporte mínimo":
                    outs = [min_support(m)]
                elif op == "Reescalar":
                    m.apply_scale(v["k"])
                elif op.startswith("Rodar"):
                    ax = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[op[-1]]
                    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(v["a"]), ax, m.bounds.mean(0)))
                elif op == "Deslocar":
                    m.apply_translation((v["dx"], v["dy"], v["dz"]))
                for k, mm in enumerate(outs):
                    name = base + (f"_{k + 1}" if len(outs) > 1 else "") + "." + v["fmt"]
                    mm.export(os.path.join(dst, name))
                done += 1
            if rows:
                with open(os.path.join(dst, "estatisticas.csv"), "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["ficheiro", "X (mm)", "Y (mm)", "Z (mm)", "área (mm²)", "volume (mm³)", "vértices", "triângulos", "fechada"])
                    w.writerows(rows)
            return done
        self.task(f"Lote: {v['op']}…", work, lambda n: self.log(f"Lote «{v['op']}»: {n} ficheiro(s) → {dst}"))

    def act_slicer(self):
        from .slicer_tool import SPEC
        x, y, z = self.printer()
        self.open_tool(SPEC)
        if self.tool_panel.obj is None:
            return
        e = self.tool_panel.obj.mesh.extents
        for k, val in (("px", x), ("py", y), ("pz", z), ("sx", e[0]), ("sy", e[1]), ("sz", e[2])):
            w = self.tool_panel.widgets[k]
            w.blockSignals(True); w.setValue(float(val)); w.blockSignals(False)
        self.tool_panel.changed()

    def show_param_page(self):
        self.side_stack.setCurrentWidget(self.param_panel)

    def act_cut(self):
        """Cortar: painel com métodos, plano e conectores, e pré-visualização ao vivo."""
        objs = [o for o in self.doc.objects if o.id in self.view3d.selected_ids]
        if not objs:
            vis = [o for o in self.doc.objects if o.visible]
            objs = vis if len(vis) == 1 else []
        if not objs:
            return self.error("Selecione o objeto a cortar: clique nele na vista 3D ou na lista Objetos 3D.")
        # peças de um corte já feito: volta a esse corte (ajustar e cortar de novo)
        cut_parts = [o for o in objs if (getattr(o, "tool", None) or {}).get("kind") == "cut"]
        if cut_parts:
            return self.reopen_tool(cut_parts[0])
        if len(objs) > 1:
            names = [f"{o.name}" for o in objs]
            v = self.ask("Cortar", [("o", "Objeto a cortar", names[0], names)],
                         note="Há vários objetos selecionados; o corte é feito num de cada vez.")
            if not v:
                return
            objs = [objs[names.index(v["o"])]]
        self.side_stack.setCurrentWidget(self.cut_panel)
        self._show_tool_dock()
        self.tabs.setCurrentWidget(self.view3d)
        self.cut_panel.start(objs[0])

    # --- criar -------------------------------------------------------------
    def _src_for(self, gid):
        needs = P.REGISTRY[gid]["needs"]
        src = {}
        if needs == "image":
            it = self._image_for_3d()
            if it is None:
                return None
            a = np.asarray(self.imgview.current_slice() if it.meta.get("volume") else it.data)
            src["image"] = iop.to_gray(a) if a.ndim == 3 else (a if a.dtype == np.uint8 else iop.to_u8(a))
            src["image_name"] = it.name
        if needs == "geom" or gid in OPTIONAL_GEOM:
            from shapely.ops import unary_union
            sel = [s for s in self.canvas.selected_shapes() if s.closed]
            if sel:
                src["geom"] = unary_union([s.geom for s in sel]).wkt
            elif needs == "geom":
                self.error("Desenhe e selecione formas 2D fechadas no separador 2D." if gid != "revolucao" else
                           "Desenhe e selecione no separador 2D um perfil fechado (X = raio, Y = altura) à direita do eixo Y.")
                return None
        return src

    def create_param(self, gid, values=None, src=None, name=None, ask=False):
        spec = P.REGISTRY[gid]
        if src is None:
            src = self._src_for(gid)
            if src is None:
                return
        vals = P.defaults(gid); vals.update(values or {})
        if ask:
            v = self.ask(spec["name"], [p[:4] for p in spec["params"]])
            if not v:
                return
            vals.update(v)
        name = name or spec["name"]

        def done(result):
            meshes = result if isinstance(result, list) else [result]
            gen = new_gen(gid, vals, src)
            objs = []
            # coloca o conjunto ao lado dos objetos existentes, mantendo a disposição relativa das peças
            import trimesh
            whole = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0].copy()
            b0 = whole.bounds.copy()
            self._place_new(whole)
            shift = whole.bounds[0] - b0[0]
            for k, m in enumerate(meshes):
                if len(meshes) > 1:
                    m.apply_translation(shift)
                elif k == 0:
                    m.apply_translation(shift)
                o = self.add_mesh(m, name if len(meshes) == 1 else f"{name} {k + 1}", select=False, place=False)
                o.gen = dict(gen, part=k)
                o.gen["values"] = gen["values"]  # partilhado pelo grupo
                objs.append(o)
            self._select_objs(objs[:1])
            self.tabs.setCurrentWidget(self.view3d); self.view3d.fit()
            self._show_tool_dock()
            self.log(f"Criado: {name}. Ajuste os valores no painel Parâmetros.")
        self.task(f"A criar {name}…", lambda: P.build(gid, vals, src), done)

    # --- atualizar ---------------------------------------------------------
    def group_objs(self, group):
        objs = [o for o in self.doc.objects if getattr(o, "gen", None) and o.gen.get("group") == group]
        return sorted(objs, key=lambda o: o.gen.get("part", 0))

    def replace_group(self, group, meshes, final=True):
        objs = self.group_objs(group)
        if not objs:
            return
        old = np.array([mbounds(o.mesh) for o in objs]); new = np.array([mbounds(m) for m in meshes])
        olo, ohi = old[:, 0].min(0), old[:, 1].max(0)
        nlo, nhi = new[:, 0].min(0), new[:, 1].max(0)
        d = np.array([(olo[0] + ohi[0]) / 2 - (nlo[0] + nhi[0]) / 2, (olo[1] + ohi[1]) / 2 - (nlo[1] + nhi[1]) / 2, olo[2] - nlo[2]])
        for m in meshes:
            m.apply_translation(d)
        if final:
            self.doc.snapshot()
        gen = objs[0].gen
        for k, m in enumerate(meshes):
            if k < len(objs):
                objs[k].mesh = m
                objs[k].vcolors = None
            else:
                o = self.doc.add(m, f"{objs[0].name.rsplit(' ', 1)[0]} {k + 1}")
                o.gen = dict(gen, part=k); o.gen["values"] = gen["values"]
        for o in objs[len(meshes):]:
            self.doc.objects.remove(o)
        self.doc.changed()

    # --- menus convertidos (substituem os diálogos de uma só vez) ----------
    def gen_prism(self): self.create_param("prisma")
    def gen_pyramid(self): self.create_param("piramide")
    def gen_twist(self): self.create_param("extrusao_torcida")
    def gen_lathe(self): self.create_param("revolucao")
    def gen_box(self): self.create_param("caixa_tampa")
    def gen_nested(self): self.create_param("caixas_enc")
    def gen_vase(self): self.create_param("vaso")
    def gen_spring(self): self.create_param("mola")
    def gen_dowel(self): self.create_param("cavilha")
    def gen_framed(self): self.create_param("relevo_moldura")
    def gen_cyl_litho(self): self.create_param("lito_cil")
    def gen_sph_litho(self): self.create_param("lito_esf")
    def gen_lamp_base(self): self.create_param("candeeiro_base")
    def gen_keychain(self): self.create_param("porta_chaves")
    def gen_cutter(self): self.create_param("cortador")
    def gen_stamp(self): self.create_param("carimbo")
    def gen_puzzle3d(self): self.create_param("puzzle")
    def m2_gen_stand(self): self.create_param("suporte_foto")
    def m2_gen_hanger(self): self.create_param("gancho")
    def m2_gen_clip(self): self.create_param("garra")
    def m2_gen_pillar(self): self.create_param("pilar")
