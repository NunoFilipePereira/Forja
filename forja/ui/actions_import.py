"""Ficheiro ▸ Importar e afins:
histograma de 1 e 2 valores, curva 2D, formas básicas, estatística progressiva, nuvem de pontos, pontos e normais,
rasterizar malha STL; juntar ficheiros num só projeto; recetor de dados (TCP); abrir e exportar figuras (.fgs + STL)."""
import os
import re
import numpy as np
import trimesh
from PySide6.QtWidgets import QFileDialog

DATA_FILTER = "Dados (*.txt *.csv *.dat *.asc *.xyz);;Todos (*)"


def _load_cols(p):
    from ..core.imageops2 import load_ascii
    A = np.asarray(load_ascii(p), float)
    if A.ndim == 1:
        A = A[:, None]
    return A


class ImportActions:
    def menus_import(self):
        return [("&Ficheiro", [
            None,
            ("Importar (dados)", [
                ("Histograma de 1 valor…", self.imp_hist1),
                ("Histograma de 2 valores…", self.imp_hist2),
                ("Curva 2D…", self.imp_curve2d),
                ("Formas básicas…", self.imp_basic_shapes),
                ("Estatística progressiva…", self.imp_progressive),
                ("Nuvem de pontos…", self.imp_point_cloud),
                ("Pontos e normais…", self.imp_points_normals),
                ("Rasterizar malha STL (volume)…", self.imp_rasterize),
            ]),
            ("Juntar ficheiros num projeto (Assemble)…", self.imp_assemble),
            ("Recetor de dados (rede)…", self.imp_receiver),
            ("Palavra-passe do projeto…", self.act_set_password),
            ("Guardar projeto com palavra-passe…", self.imp_save_protected),
            None,
            ("Abrir pasta de figura (.fgs + STL)…", self.imp_open_fgs),
            ("Exportar como pasta de figura (.fgs + STL)…", self.imp_export_fgs),
            ("Exportar um STL por objeto…", self.imp_stl_per_object),
        ])]

    # ------------------------------------------------------------------ dados
    def _ask_data(self, title):
        p, _ = QFileDialog.getOpenFileName(self, title, self.last_dir, DATA_FILTER)
        if not p:
            return None, None
        self._dir(p)
        try:
            return p, _load_cols(p)
        except Exception as e:
            self.error(f"{os.path.basename(p)}: {e}")
            return None, None

    def imp_hist1(self):
        """Histograma de 1 valor: uma coluna de valores → histograma no separador Gráficos."""
        p, A = self._ask_data("Histograma de 1 valor")
        if A is None:
            return
        v = self.ask("Histograma de 1 valor", [("col", "Coluna", 1, (1, max(A.shape[1], 1))), ("bins", "Número de classes", 30, (2, 1000))],
                     note=f"{A.shape[0]} linhas × {A.shape[1]} colunas")
        if not v:
            return
        x = A[:, int(v["col"]) - 1]; x = x[np.isfinite(x)]
        h, e = np.histogram(x, bins=int(v["bins"]))
        c = (e[:-1] + e[1:]) / 2
        self.chart.series.append({"x": c, "y": h.astype(float), "label": f"{os.path.basename(p)} [col {int(v['col'])}]", "kind": "barras"})
        self._show_chart()
        self.log(f"Histograma: n={len(x)}, média {x.mean():.4g}, desvio-padrão {x.std():.4g}, mín {x.min():.4g}, máx {x.max():.4g}")

    def imp_hist2(self):
        """Histograma de 2 valores: duas colunas → histograma 2D (imagem com barra de cores)."""
        p, A = self._ask_data("Histograma de 2 valores")
        if A is None:
            return
        if A.shape[1] < 2:
            return self.error("São precisas pelo menos 2 colunas.")
        v = self.ask("Histograma de 2 valores", [("a", "Coluna X", 1, (1, A.shape[1])), ("b", "Coluna Y", 2, (1, A.shape[1])),
                                                  ("bins", "Classes por eixo", 64, (2, 2000))])
        if not v:
            return
        x, y = A[:, int(v["a"]) - 1], A[:, int(v["b"]) - 1]
        ok = np.isfinite(x) & np.isfinite(y)
        H, xe, ye = np.histogram2d(x[ok], y[ok], bins=int(v["bins"]))
        self.add_image(f"histograma 2D {os.path.basename(p)}", np.flipud(H.T).astype(np.float32),
                       {"x0": float(xe[0]), "x1": float(xe[-1]), "y0": float(ye[0]), "y1": float(ye[-1])})

    def imp_curve2d(self):
        """Curva 2D: colunas X e Y → polilinha no desenho 2D e série no gráfico."""
        p, A = self._ask_data("Curva 2D")
        if A is None:
            return
        if A.shape[1] < 2:
            return self.error("São precisas 2 colunas (X e Y).")
        from shapely.geometry import LineString, Polygon
        from ..core.shapes2d import Shape
        xy = A[:, :2][np.all(np.isfinite(A[:, :2]), 1)]
        closed = len(xy) > 3 and np.allclose(xy[0], xy[-1])
        geom = Polygon(xy) if closed else LineString(xy)
        self.canvas.add_shapes([Shape("geo", {}, geom, os.path.basename(p))])
        self.chart.series.append({"x": xy[:, 0], "y": xy[:, 1], "label": os.path.basename(p), "kind": "linha"})
        self.tabs.setCurrentWidget(self.canvas)
        self.log(f"Curva 2D: {len(xy)} pontos ({'fechada' if closed else 'aberta'})")

    def imp_basic_shapes(self):
        """Formas básicas: ficheiro de texto com uma forma por linha:
        point x y · line x1 y1 x2 y2 · rect x y w h · circle cx cy r · ellipse cx cy rx ry ·
        polygon x1 y1 x2 y2 … · polyline x1 y1 … · arc cx cy r a0 a1 · text x y texto"""
        p, _ = QFileDialog.getOpenFileName(self, "Formas básicas", self.last_dir, "Texto (*.txt *.csv);;Todos (*)")
        if not p:
            return
        from shapely.geometry import Point, LineString, Polygon, box
        from shapely import affinity
        from ..core.shapes2d import Shape
        shapes, bad = [], 0
        for k, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,;]+", line)
            kind = parts[0].lower()
            try:
                if kind == "text":
                    from ..core.text3d import text_polygons
                    g = text_polygons(" ".join(parts[3:]), size_mm=5)
                    b = g.bounds
                    g = affinity.translate(g, float(parts[1]) - b[0], float(parts[2]) - b[1])
                else:
                    n = [float(x) for x in parts[1:]]
                    if kind == "point":
                        g = Point(n[0], n[1]).buffer(0.3)
                    elif kind == "line":
                        g = LineString([(n[0], n[1]), (n[2], n[3])])
                    elif kind in ("rect", "rectangle"):
                        g = box(n[0], n[1], n[0] + n[2], n[1] + n[3])
                    elif kind == "circle":
                        g = Point(n[0], n[1]).buffer(n[2], quad_segs=32)
                    elif kind == "ellipse":
                        g = affinity.scale(Point(n[0], n[1]).buffer(1, quad_segs=32), n[2], n[3])
                    elif kind == "polygon":
                        g = Polygon(np.reshape(n, (-1, 2)))
                    elif kind == "polyline":
                        g = LineString(np.reshape(n, (-1, 2)))
                    elif kind == "arc":
                        a = np.radians(np.linspace(n[3], n[4], 48))
                        g = LineString(np.c_[n[0] + n[2] * np.cos(a), n[1] + n[2] * np.sin(a)])
                    else:
                        raise ValueError(kind)
                shapes.append(Shape("geo", {}, g, f"{kind} {k}"))
            except Exception:
                bad += 1
        if shapes:
            self.canvas.add_shapes(shapes); self.tabs.setCurrentWidget(self.canvas)
        self.log(f"Formas básicas: {len(shapes)} importadas" + (f", {bad} linha(s) não reconhecidas" if bad else ""))

    def imp_progressive(self):
        """Estatística progressiva: média, desvio-padrão, mínimo e máximo acumulados ao longo das amostras."""
        p, A = self._ask_data("Estatística progressiva")
        if A is None:
            return
        v = self.ask("Estatística progressiva", [("col", "Coluna", 1, (1, max(A.shape[1], 1)))])
        if not v:
            return
        x = A[:, int(v["col"]) - 1]; x = x[np.isfinite(x)]
        n = np.arange(1, len(x) + 1)
        mean = np.cumsum(x) / n
        var = np.maximum(np.cumsum(x ** 2) / n - mean ** 2, 0)
        name = os.path.basename(p)
        for y, lab in ((x, "valor"), (mean, "média"), (mean + np.sqrt(var), "média + σ"), (mean - np.sqrt(var), "média − σ"),
                       (np.minimum.accumulate(x), "mínimo"), (np.maximum.accumulate(x), "máximo")):
            self.chart.series.append({"x": n.astype(float), "y": y, "label": f"{name}: {lab}", "kind": "linha"})
        self._show_chart()
        self.log(f"Estatística progressiva ({name}): final média {mean[-1]:.4g}, σ {np.sqrt(var[-1]):.4g}")

    def imp_point_cloud(self):
        p, A = self._ask_data("Nuvem de pontos")
        if A is None:
            return
        if A.shape[1] < 3:
            return self.error("São precisas 3 colunas (X Y Z).")
        P = A[:, :3][np.all(np.isfinite(A[:, :3]), 1)]
        m = trimesh.Trimesh(P, np.zeros((0, 3), int), process=False)
        o = self.add_mesh(m, os.path.basename(p), place=False)
        if A.shape[1] >= 6 and A[:, 3:6].max() > 1.0:  # colunas de cor RGB 0–255
            c = np.clip(A[:len(P), 3:6] / 255.0, 0, 1)
            o.vcolors = (id(o.mesh), np.c_[c, np.ones(len(c))].astype(np.float32))
        self.doc.changed()
        self.log(f"Nuvem de pontos: {len(P)} pontos")

    def imp_points_normals(self):
        """Pontos e normais (x y z nx ny nz): nuvem com as normais; opcionalmente reconstrói uma superfície."""
        p, A = self._ask_data("Pontos e normais")
        if A is None:
            return
        if A.shape[1] < 6:
            return self.error("São precisas 6 colunas (x y z nx ny nz).")
        ok = np.all(np.isfinite(A[:, :6]), 1)
        P, N = A[ok, :3], A[ok, 3:6]
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        v = self.ask("Pontos e normais", [("mode", "Mostrar como", "pontos", ["pontos", "pontos + normais", "superfície (reconstruir)"]),
                                          ("len", "Comprimento das normais (mm)", float(np.ptp(P, 0).max() / 40))])
        if not v:
            return
        name = os.path.basename(p)
        if v["mode"] == "superfície (reconstruir)":
            from ..core import mesh2
            try:
                m = mesh2.reconstruct_surface(P, N) if hasattr(mesh2, "reconstruct_surface") else None
            except Exception:
                m = None
            if m is None:
                # reconstrução simples: casca convexa orientada pelas normais (aproximação)
                m = trimesh.convex.convex_hull(P)
            self.add_mesh(m, f"{name} superfície", place=False)
        else:
            o = self.add_mesh(trimesh.Trimesh(P, np.zeros((0, 3), int), process=False), name, place=False)
            o.normals = N
            if v["mode"] == "pontos + normais":
                L = v["len"]
                segs = np.stack([P, P + N * L], 1)
                tube = trimesh.load_path(segs)
                verts = np.asarray(tube.vertices)
                m2 = trimesh.Trimesh(verts, np.zeros((0, 3), int), process=False)
                self.add_mesh(m2, f"{name} normais", place=False)
        self.log(f"Pontos e normais: {len(P)} pontos")

    def imp_rasterize(self):
        """Rasterizar malha STL: o modelo (ficheiro ou objeto selecionado) em voxels → volume no separador CT."""
        objs = [o for o in self.doc.objects if o.id in self.view3d.selected_ids]
        if objs:
            m, name = objs[0].mesh, objs[0].name
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Malha STL", self.last_dir, "Malhas (*.stl *.obj *.ply *.off *.3mf)")
            if not p:
                return
            m, name = trimesh.load(p, force="mesh"), os.path.basename(p)
        e = m.extents
        v = self.ask("Rasterizar malha", [("px", "Tamanho do voxel (mm)", float(max(e) / 200), (1e-3, 1e3)),
                                          ("fill", "Preencher o interior", True)],
                     note=f"{name}: {e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm")
        if not v:
            return

        def work():
            vox = m.voxelized(v["px"])
            if v["fill"]:
                vox = vox.fill()
            return np.transpose(vox.matrix.astype(np.uint8) * 255, (2, 1, 0))[::-1]
        self.task("A rasterizar…", work, lambda vol: (self.add_volume(f"{name} rasterizado", vol, {"pitch": v["px"]})
                                                       if hasattr(self, "add_volume") else self.add_image(f"{name} rasterizado", vol[vol.shape[0] // 2]),
                                                       self.log(f"Rasterizado: {vol.shape[2]} × {vol.shape[1]} × {vol.shape[0]} voxels")))

    # ------------------------------------------------------------------ projeto
    def imp_assemble(self):
        """Juntar ficheiros: abre vários modelos, imagens e desenhos de uma vez no mesmo projeto,
        mantendo as posições originais dos modelos."""
        ps, _ = QFileDialog.getOpenFileNames(self, "Juntar ficheiros", self.last_dir,
                                             "Suportados (*.stl *.obj *.ply *.off *.3mf *.glb *.png *.jpg *.bmp *.tif *.dxf *.svg);;Todos (*)")
        if not ps:
            return
        n = 0
        for p in ps:
            ext = p.lower().rsplit(".", 1)[-1]
            if ext in ("stl", "obj", "ply", "off", "3mf", "glb"):
                try:
                    self.add_mesh(trimesh.load(p, force="mesh"), os.path.basename(p), select=False, place=False); n += 1
                except Exception as e:
                    self.error(f"{os.path.basename(p)}: {e}")
            else:
                self.open_path(p); n += 1
        self.view3d.fit()
        self.log(f"Juntar ficheiros: {n} de {len(ps)} ficheiro(s) no projeto")

    def imp_save_protected(self):
        v = self.ask("Guardar com palavra-passe", [("p", "Palavra-passe", ""), ("q", "Repetir", "")])
        if not v:
            return
        if not v["p"] or v["p"] != v["q"]:
            return self.error("Escreva a mesma palavra-passe nos dois campos.")
        self.act_save_project(password=v["p"])

    def imp_stl_per_object(self):
        objs = [o for o in self.doc.objects if o.visible]
        if not objs:
            return self.error("Não há objetos 3D.")
        d = QFileDialog.getExistingDirectory(self, "Pasta para os STL", self.last_dir)
        if not d:
            return
        for o in objs:
            safe = re.sub(r"[^\w\-. ]+", "_", o.name).strip() or f"objeto_{o.id}"
            o.mesh.export(os.path.join(d, safe + ".stl"))
        self.log(f"{len(objs)} STL gravados em {d}")

    # ------------------------------------------------------------------ recetor de dados
    def imp_receiver(self):
        """Recetor de dados: escuta numa porta TCP; cada linha recebida com números é
        acrescentada — 1 valor: série no gráfico; 2 valores: X Y no gráfico; 3 ou mais: ponto 3D numa nuvem."""
        from PySide6.QtNetwork import QTcpServer, QHostAddress
        srv = getattr(self, "_rx_server", None)
        if srv is not None and srv.isListening():
            srv.close(); self._rx_server = None
            return self.log("Recetor de dados parado.")
        v = self.ask("Recetor de dados", [("port", "Porta TCP", 5555, (1024, 65535)), ("local", "Só ligações deste computador", True)],
                     note="Envie linhas de texto com números separados por espaço, vírgula ou ponto e vírgula. "
                          "Volte a escolher esta opção para parar.")
        if not v:
            return
        srv = QTcpServer(self)
        addr = QHostAddress(QHostAddress.LocalHost) if v["local"] else QHostAddress(QHostAddress.Any)
        if not srv.listen(addr, int(v["port"])):
            return self.error(f"Não foi possível escutar na porta {int(v['port'])}: {srv.errorString()}")
        self._rx_server = srv
        self._rx = {"vals": [], "xy": [], "pts": [], "obj": None}

        def on_conn():
            while srv.hasPendingConnections():
                c = srv.nextPendingConnection()
                c.readyRead.connect(lambda c=c: self._rx_read(c))
                self.log(f"Recetor: ligação de {c.peerAddress().toString()}")
        srv.newConnection.connect(on_conn)
        self.log(f"Recetor de dados à escuta na porta {int(v['port'])}")

    def _rx_read(self, c):
        rx = self._rx
        changed = set()
        while c.canReadLine():
            line = bytes(c.readLine()).decode("utf-8", "replace").strip()
            nums = [float(x) for x in re.split(r"[\s,;]+", line) if re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$", x)]
            if len(nums) == 1:
                rx["vals"].append(nums[0]); changed.add("vals")
            elif len(nums) == 2:
                rx["xy"].append(nums); changed.add("xy")
            elif len(nums) >= 3:
                rx["pts"].append(nums[:3]); changed.add("pts")
        if "vals" in changed or "xy" in changed:
            self.chart.series = [s for s in self.chart.series if not s.get("label", "").startswith("recetor")]
            if rx["vals"]:
                y = np.asarray(rx["vals"]); self.chart.series.append({"x": np.arange(len(y), dtype=float), "y": y, "label": "recetor", "kind": "linha"})
            if rx["xy"]:
                a = np.asarray(rx["xy"]); self.chart.series.append({"x": a[:, 0], "y": a[:, 1], "label": "recetor X-Y", "kind": "pontos"})
            self.chart.draw()
        if "pts" in changed:
            P = np.asarray(rx["pts"], float)
            m = trimesh.Trimesh(P, np.zeros((0, 3), int), process=False)
            o = rx["obj"]
            if o is None or o not in self.doc.objects:
                rx["obj"] = self.add_mesh(m, "recetor (pontos)", select=False, place=False)
            else:
                o.mesh = m; self.doc.changed()

    # ------------------------------------------------------------------ figuras .fgs
    def imp_open_fgs(self):
        """Abre uma pasta de figura: lê o .fgs (nomes, cores, transparência, o que se mostra, vista) e carrega
        os STL de cada peça gravados ao lado (Part_*.stl, Shape_STL/…)."""
        from ..core import fgs as LF
        p, _ = QFileDialog.getOpenFileName(self, "Ficheiro .fgs", self.last_dir, "Figura (*.fgs);;Todos (*)")
        if not p:
            return
        self._dir(p)
        fig = LF.read_fgs(p)
        folder = os.path.dirname(p)
        n, missing = 0, []
        for sh in fig["shapes"]:
            f = LF.find_stl(folder, sh["Name"])
            if not f:
                missing.append(sh["Name"]); continue
            m = trimesh.load(f, force="mesh")
            o = self.add_mesh(m, sh["Name"], select=False, place=False)
            LF.apply_shape_style(o, sh)
            n += 1
        LF.apply_view(self.view3d, fig["view"])
        self.doc.changed(); self.view3d.refresh(); self.view3d.fit()
        self.log(f"Figura {os.path.basename(p)}: {n} peça(s)" + (f"; sem STL para: {', '.join(missing)}" if missing else ""))

    def imp_export_fgs(self):
        from ..core import fgs as LF
        objs = [o for o in self.doc.objects if o.visible]
        if not objs:
            return self.error("Não há objetos 3D.")
        d = QFileDialog.getExistingDirectory(self, "Pasta para a figura .fgs", self.last_dir)
        if not d:
            return
        LF.write_fgs(d, objs, self.view3d)
        self.log(f"Exportado como figura .fgs: {d} (Forja.fgs + um STL por objeto)")
