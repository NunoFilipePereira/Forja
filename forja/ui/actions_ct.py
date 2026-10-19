"""Menu CT / Volume."""
import os
import numpy as np
from PySide6.QtWidgets import QFileDialog
from ..core import ct
from .widgets import TextDialog


class CTActions:
    def menus_ct(self):
        return [("&CT / Volume", [
            ("Transformada de Radon (imagem → sinograma)…", self.ct_radon),
            ("Retroprojeção filtrada (sinograma → imagem)…", self.ct_fbp),
            ("Retroprojeção simples (sem filtro)", lambda: self.ct_fbp(simple=True)),
            ("Reconstrução iterativa SART…", self.ct_sart),
            ("Projeções → volume 3D (feixe paralelo)…", self.ct_volume),
            None,
            ("Abrir volume", [("Pasta de fatias (PNG/TIFF/DICOM)…", self.ct_load_stack),
                              ("Ficheiro .im3…", self.ct_load_im3),
                              ("Imagens selecionadas como fatias", self.ct_from_selection)]),
            ("Guardar volume .im3…", self.ct_save_im3),
            ("Exportar fatias do volume para pasta…", self.ct_export_slices),
            None,
            ("Corte atual → imagem 2D", self.ct_current_slice),
            ("Projeção do volume…", self.ct_project),
            ("Isosuperfície → objeto 3D…", self.ct_iso),
            ("Objeto 3D → volume (voxelizar)…", self.ct_voxelize),
            ("Estatísticas do volume…", self.ct_stats),
            ("Ver o volume em 3D", [("Completo (Full)", lambda: self.ct_show3d("full")), ("Três planos (Triplane)", lambda: self.ct_show3d("tri")),
                                    ("Retirar da vista 3D", lambda: self.ct_show3d(None))]),
            ("Mapa de cor do volume…", self.ct_colormap),
            None,
            ("Fantasma 2D (Shepp–Logan)", lambda: self.add_image("fantasma Shepp-Logan", ct.phantom2d(256))),
            ("Fantasma 3D + projeções de teste…", self.ct_phantom3d),
        ])]

    # --- Volume em 3D ---------------------
    def _vol_item(self):
        r = self.img_list.currentRow()
        it = self.images[r] if 0 <= r < len(self.images) else None
        if it is None or not it.meta.get("volume"):
            vols = [i for i in self.images if i.meta.get("volume") and np.asarray(i.data).ndim == 3]
            it = vols[-1] if vols else None
        if it is None or not it.meta.get("volume") or np.asarray(it.data).ndim != 3:
            self.error("Selecione um volume (lista de imagens) — por exemplo CT ▸ Abrir volume ou Ficheiro ▸ Importar ▸ Rasterizar malha.")
            return None
        return it

    def ct_colormap(self):
        from ..core import colormaps as CM
        v = self.ask("Mapa de cor do volume", [("c", "Escala", getattr(self, "vol_cmap", "Gray"), CM.SCHEMES[1:-1]),
                                               ("t", "Limiar de transparência (0 – 1)", float(getattr(self, "vol_thr", 0.1)), (0, 1))])
        if v:
            self.vol_cmap, self.vol_thr = v["c"], v["t"]
            if getattr(self, "_vol3d_mode", None):
                self.ct_show3d(self._vol3d_mode)

    def ct_show3d(self, mode):
        import pyqtgraph.opengl as gl
        from ..core import colormaps as CM
        v3 = self.view3d
        for item in getattr(self, "_vol3d_items", []):
            if item in v3.items:
                v3.removeItem(item)
        self._vol3d_items = []
        self._vol3d_mode = mode
        if mode is None:
            v3.update(); return
        it = self._vol_item()
        if it is None:
            return
        V = np.asarray(it.data, np.float32)  # (z, y, x)
        px = float(it.meta.get("pitch", 1.0))
        step = max(1, int(np.ceil(max(V.shape) / (160 if mode == "full" else 400))))
        V = V[::step, ::step, ::step]
        lo, hi = np.nanpercentile(V, 1), np.nanpercentile(V, 99.5)
        T = np.clip((V - lo) / max(hi - lo, 1e-9), 0, 1)
        cmap = getattr(self, "vol_cmap", "Gray"); thr = getattr(self, "vol_thr", 0.1)
        s = px * step

        def rgba(t2d):
            c = CM.colors(t2d.ravel(), cmap)
            return (c.reshape(t2d.shape + (4,)) * 255).astype(np.ubyte)
        if mode == "full":
            Tx = np.transpose(T, (2, 1, 0))  # (x, y, z)
            data = rgba(Tx)
            data[..., 3] = (np.clip((Tx - thr) / max(1 - thr, 1e-6), 0, 1) * 90).astype(np.ubyte)
            item = gl.GLVolumeItem(data, smooth=True, glOptions="translucent")
            item.scale(s, s, s)
            v3.addItem(item); self._vol3d_items.append(item)
        else:
            nz, ny, nx = T.shape
            planes = [(T[nz // 2], "z"), (T[:, ny // 2, :], "y"), (T[:, :, nx // 2], "x")]
            for img, ax in planes:
                d = rgba(np.ascontiguousarray(img.T))  # (w, h, 4)
                d[..., 3] = 230
                item = gl.GLImageItem(d, glOptions="translucent")
                item.scale(s, s, 1)
                if ax == "z":
                    item.translate(0, 0, (nz // 2) * s)
                elif ax == "y":
                    item.rotate(90, 1, 0, 0); item.translate(0, (ny // 2) * s, 0)
                else:
                    item.rotate(90, 1, 0, 0); item.rotate(90, 0, 0, 1); item.translate((nx // 2) * s, 0, 0)
                v3.addItem(item); self._vol3d_items.append(item)
        self.tabs.setCurrentWidget(v3); v3.update()
        self.log(f"Volume em 3D ({'completo' if mode == 'full' else 'três planos'}): {T.shape[2]} × {T.shape[1]} × {T.shape[0]} voxels, escala {cmap}")

    def add_volume(self, name, vol, meta=None):
        m = dict(meta or {}); m["volume"] = True
        return self.add_image(name, np.asarray(vol), m)

    def _cur_volume(self):
        it = self.imgview.item
        if not (it and self.imgview.is_volume()):
            self.error("Selecione um volume 3D na lista Imagens.")
            return None
        return it

    def ct_radon(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Radon", [("n", "Número de ângulos", 180, (2, 3600)), ("a", "Ângulo máximo (°)", 180.0, (1, 360))])
        if v:
            S, th = ct.radon(it.data, v["n"], v["a"])
            self.add_image(f"sinograma({it.name})", S, {"max_angle": v["a"]})

    def ct_fbp(self, simple=False):
        it = self.cur_image()
        if not it:
            return
        if simple:
            v = {"f": "sem filtro", "a": it.meta.get("max_angle", 180.0), "s": 0}
        else:
            v = self.ask("Retroprojeção filtrada", [("f", "Filtro", "ramp", ct.FILTERS), ("a", "Ângulo máximo (°)", float(it.meta.get("max_angle", 180.0))),
                                                    ("s", "Tamanho de saída (px, 0 = automático)", 0, (0, 10000))],
                         note="Sinograma: linhas = posição no detetor, colunas = ângulos.")
            if not v:
                return
        S = it.data
        self.task("A reconstruir…", lambda: ct.backproject(S, v["a"], v["f"], v["s"] or None),
                  lambda R: self.add_image(f"{'FBP' if v['f'] != 'sem filtro' else 'BP'}({it.name})", R))

    def ct_sart(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("SART", [("n", "Iterações", 3, (1, 50)), ("a", "Ângulo máximo (°)", float(it.meta.get("max_angle", 180.0)))])
        if v:
            S = it.data
            self.task("A reconstruir…", lambda: ct.sart(S, v["a"], v["n"]), lambda R: self.add_image(f"SART({it.name})", R))

    def ct_volume(self):
        items = self.sel_images(2)
        if not items:
            return
        v = self.ask("Projeções → volume", [("a", "Ângulo total coberto (°)", 180.0, (1, 360)), ("f", "Filtro", "ramp", ct.FILTERS)],
                     note=f"{len(items)} projeções por ordem da lista, igualmente espaçadas em ângulo.")
        if v:
            P = [i.data for i in items]
            self.task("A reconstruir volume…", lambda: ct.projections_to_volume(P, v["a"], v["f"]),
                      lambda V: self.add_volume(f"volume CT ({len(P)} projeções)", V))

    def ct_load_stack(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta de fatias", self.last_dir)
        if d:
            self.task("A carregar fatias…", lambda: ct.load_image_stack(d), lambda V: self.add_volume(os.path.basename(d), V))

    def ct_load_im3(self):
        p, _ = QFileDialog.getOpenFileName(self, "Volume", self.last_dir, "Volume 3D (*.im3);;Todos (*)")
        if p:
            try:
                self.add_volume(os.path.basename(p), ct.load_im3(p))
            except Exception as e:
                self.error(e)

    def ct_from_selection(self):
        items = self.sel_images(2)
        if items:
            shp = np.shape(items[0].data)[:2]
            sl = [np.asarray(i.data, float) if np.ndim(i.data) == 2 else np.asarray(i.data, float).mean(2) for i in items
                  if np.shape(i.data)[:2] == shp]
            self.add_volume(f"volume ({len(sl)} fatias)", np.stack(sl))

    def ct_save_im3(self):
        it = self._cur_volume()
        if it:
            p, _ = QFileDialog.getSaveFileName(self, "Guardar volume", os.path.join(self.last_dir, "volume.im3"), "Volume 3D (*.im3)")
            if p:
                ct.save_im3(p, it.data); self.log(f"Volume guardado: {p}")

    def ct_export_slices(self):
        it = self._cur_volume()
        if not it:
            return
        d = QFileDialog.getExistingDirectory(self, "Pasta de destino", self.last_dir)
        if not d:
            return
        from ..core import imageops as iop
        V = np.asarray(it.data, float)
        lo, hi = V.min(), V.max()
        for k, sl in enumerate(V):
            iop.save(os.path.join(d, f"fatia_{k:04d}.png"), ((sl - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8))
        self.log(f"{len(V)} fatias exportadas para {d}")

    def ct_current_slice(self):
        it = self._cur_volume()
        if it:
            ax = ["Z", "Y", "X"][self.imgview.vol_axis.currentIndex()]
            self.add_image(f"corte {ax}={self.imgview.vol_slider.value()}({it.name})", np.array(self.imgview.current_slice()))

    def ct_project(self):
        it = self._cur_volume()
        if not it:
            return
        v = self.ask("Projeção", [("ax", "Direção", "Z", ["Z", "Y", "X"]), ("k", "Tipo", "máxima", ["máxima", "média", "soma", "mínima"])])
        if v:
            self.add_image(f"projeção {v['k']} {v['ax']}({it.name})", ct.project(it.data, "ZYX".index(v["ax"]), v["k"]))

    def ct_iso(self):
        it = self._cur_volume()
        if not it:
            return
        V = np.asarray(it.data, float)
        lo, hi = float(V.min()), float(V.max())
        v = self.ask("Isosuperfície", [("l", "Nível", (lo + hi) / 2, (lo, hi)), ("sx", "Espaçamento X/Y (mm)", 1.0), ("sz", "Espaçamento Z (mm)", 1.0),
                                       ("st", "Passo (1 = máximo detalhe)", 1, (1, 10))])
        if v:
            self.task("A extrair superfície…", lambda: ct.isosurface(V, v["l"], (v["sz"], v["sx"], v["sx"]), v["st"]),
                      lambda m: (self.add_mesh(m, f"iso {it.name}"), self.tabs.setCurrentWidget(self.view3d), self.view3d.fit()))

    def ct_voxelize(self):
        o = self.sel_objs(1, True)
        if not o:
            return
        v = self.ask("Voxelizar", [("p", "Tamanho do voxel (mm)", float(round(o[0].mesh.scale / 64, 3)), (1e-4, 1e4))])
        if v:
            mesh = o[0].mesh
            self.task("A voxelizar…", lambda: ct.voxelize(mesh, v["p"]), lambda V: self.add_volume(f"voxels {o[0].name}", V, {"pitch": v["p"]}))

    def ct_stats(self):
        it = self._cur_volume()
        if not it:
            return
        V = np.asarray(it.data, float)
        v = self.ask("Estatísticas", [("t", "Limiar", float((V.min() + V.max()) / 2)), ("s", "Tamanho do voxel (mm)", float(it.meta.get("pitch", 1.0)))])
        if v:
            st = ct.volume_stats(V, v["t"], v["s"])
            TextDialog.show_text(self, "Volume", "\n".join(f"{k}: {val}" for k, val in st.items()))

    def ct_phantom3d(self):
        v = self.ask("Fantasma 3D", [("n", "Resolução (voxels)", 64, (16, 256)), ("a", "Projeções", 90, (4, 720))])
        if not v:
            return
        V = ct.phantom3d(v["n"])
        self.add_volume("fantasma 3D", V)
        for k, p in enumerate(ct.forward_project_volume(V, v["a"])):
            self.add_image(f"projeção {k + 1:03d}", p, show=False)
        self.log(f"Fantasma 3D e {v['a']} projeções (0–180°). Selecione as projeções e use Projeções → volume 3D.")
