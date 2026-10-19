"""Opções de visualização: iluminação, projeção, régua 3D, textura, legenda, eixos, tema, unidades."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QColorDialog


def texture_colors(mesh, img, mode="topo"):
    """Cores por vértice a partir de uma imagem projetada (topo, frente, lado ou cilíndrica)."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    if a.dtype != np.uint8:
        lo, hi = np.nanmin(a), np.nanmax(a)
        a = ((np.nan_to_num(a) - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8)
    V = np.asarray(mesh.vertices)
    lo, hi = V.min(0), V.max(0)
    ext = np.maximum(hi - lo, 1e-9)
    if mode == "topo":
        u, v = (V[:, 0] - lo[0]) / ext[0], 1 - (V[:, 1] - lo[1]) / ext[1]
    elif mode == "frente":
        u, v = (V[:, 0] - lo[0]) / ext[0], 1 - (V[:, 2] - lo[2]) / ext[2]
    elif mode == "lado":
        u, v = (V[:, 1] - lo[1]) / ext[1], 1 - (V[:, 2] - lo[2]) / ext[2]
    else:  # cilíndrica em torno de Z
        c = (lo + hi) / 2
        u = (np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]) + np.pi) / (2 * np.pi)
        v = 1 - (V[:, 2] - lo[2]) / ext[2]
    h, w = a.shape[:2]
    px = np.clip((u * (w - 1)).astype(int), 0, w - 1); py = np.clip((v * (h - 1)).astype(int), 0, h - 1)
    rgb = a[py, px, :3].astype(np.float32) / 255
    return np.c_[rgb, np.ones(len(rgb), np.float32)]


def QApplication_process():
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()


class ViewActions:
    def menus_view(self):
        return [("&Ver", [
            None,
            ("Projeção ortográfica / perspetiva", self.vw_toggle_ortho, "F1"),
            ("Iluminação…", self.vw_lighting),
            ("Ângulo da luz (X Y)…", self.vw_light_angle),
            ("Sequência de vistas (exportar imagens)…", self.vw_multiview),
            ("Tamanho da vista 3D (largura altura)…", self.vw_view_size),
            ("Régua 3D (clique em 2 pontos)", self.vw_ruler, "Ctrl+Shift+R"),
            ("Mostrar/ocultar eixos", lambda: self.view3d.axis.setVisible(not self.view3d.axis.visible())),
            ("Mostrar/ocultar grelha", lambda: self.view3d.grid.setVisible(not self.view3d.grid.visible())),
            ("Tamanho dos pontos…", self.vw_point_size),
            ("Textura a partir da imagem atual…", self.vw_texture),
            ("Mostrar todos os objetos", self.vw_show_all),
            ("Ocultar os selecionados", self.vw_hide_sel),
            ("Isolar os selecionados", self.vw_isolate),
            None,
            ("Tema claro / escuro", self.vw_theme),
            ("Ecrã inteiro", self.vw_fullscreen, "F5"),
            ("Transparência da janela…", self.vw_window_opacity),
            ("Unidades das medições…", self.vw_units),
        ])]

    def vw_toggle_ortho(self):
        self.view3d.set_ortho(not self.view3d.ortho)
        self.statusBar().showMessage("Projeção ortográfica" if self.view3d.ortho else "Projeção em perspetiva")

    def vw_lighting(self):
        v = self.ask("Iluminação", [("amb", "Luz ambiente (0–1)", 0.30, (0, 2)), ("dif", "Luz difusa (0–1)", 0.55, (0, 2)),
                                    ("spe", "Brilho especular (0–1)", 0.18, (0, 2)), ("az", "Ângulo horizontal da luz (°)", 40.0, (-180, 180)),
                                    ("el", "Elevação da luz (°)", 45.0, (-90, 90))])
        if v:
            self.view3d.set_lighting(v["amb"], v["dif"], v["spe"], v["az"], v["el"])

    def vw_light_angle(self):
        """Ângulo da luz: "X Y" — X entre 0 e 180°, Y entre −180 e 180°."""
        v = self.ask("Ângulo da luz", [("a", "Ângulo X [0, 180] e Y [−180, 180]", "90 0")],
                     note="Exemplo 1: 30 (X 30°, Y 0°). Exemplo 2: 30 20 (X 30°, Y 20°).")
        if not v:
            return
        try:
            nums = [float(x) for x in str(v["a"]).replace(",", " ").split()]
        except ValueError:
            return self.error("Escreva um ou dois ângulos, por exemplo: 30 20")
        ax = max(0.0, min(180.0, nums[0])) if nums else 90.0
        ay = max(-180.0, min(180.0, nums[1])) if len(nums) > 1 else 0.0
        # X = 90° → luz de frente/por cima; converte para elevação e azimute da luz
        self.view3d.set_lighting(azimuth=ay, elevation=90.0 - ax if ax <= 90 else -(ax - 90.0))
        self.statusBar().showMessage(f"Luz: X {ax:g}°, Y {ay:g}°")

    def vw_multiview(self):
        """Sequência de vistas: "N passo" — exporta N imagens a rodar 'passo' graus cada."""
        import os
        from PySide6.QtWidgets import QFileDialog
        v = self.ask("Sequência de vistas", [("s", "Número de vistas e ângulo por passo (°)", "4 90")],
                     note="Exemplo: 4 90 (4 vistas, 90° por passo). As imagens são exportadas em PNG.")
        if not v:
            return
        try:
            n, step = str(v["s"]).split()[:2]; n = int(n); step = float(step)
        except Exception:
            return self.error("Escreva o número de vistas e o ângulo, por exemplo: 4 90")
        d = QFileDialog.getExistingDirectory(self, "Pasta para as imagens", self.last_dir)
        if not d:
            return
        v3 = self.view3d
        az0 = v3.opts["azimuth"]
        for k in range(max(n, 1)):
            v3.opts["azimuth"] = az0 + k * step
            v3.update(); QApplication_process()
            img = v3.grabFramebuffer()
            img.save(os.path.join(d, f"vista_{k + 1:03d}.png"))
        v3.opts["azimuth"] = az0; v3.update()
        self.log(f"Sequência de vistas: {n} imagens em {d}")

    def vw_view_size(self):
        """Tamanho da vista: "largura altura" em píxeis para a vista 3D."""
        v3 = self.view3d
        v = self.ask("Tamanho da vista 3D", [("s", "Largura e altura (píxeis)", f"{v3.width()} {v3.height()}")],
                     note="Exemplo: 500 400 (largura 500, altura 400). A janela ajusta-se para a vista ficar com esse tamanho.")
        if not v:
            return
        try:
            w, h = [int(float(x)) for x in str(v["s"]).split()[:2]]
        except Exception:
            return self.error("Escreva a largura e a altura, por exemplo: 500 400")
        dw, dh = w - v3.width(), h - v3.height()
        self.resize(max(self.width() + dw, 300), max(self.height() + dh, 300))

    def vw_ruler(self):
        v3 = self.view3d
        v3.measure_mode = not v3.measure_mode
        v3.clear_measure()
        self.tabs.setCurrentWidget(v3)
        if not getattr(self, "_measure_connected", False):
            v3.measured.connect(lambda m: (self.statusBar().showMessage(m), self.log(m) if m.startswith("Distância") else None))
            self._measure_connected = True
        self.statusBar().showMessage("Régua 3D ligada: clique em dois pontos da superfície (⌘⇧R desliga)" if v3.measure_mode
                                     else "Régua 3D desligada")

    def vw_point_size(self):
        v = self.ask("Pontos", [("s", "Tamanho (px)", float(self.view3d.point_size), (1, 20))])
        if v:
            self.view3d.point_size = v["s"]; self.view3d.refresh()

    def vw_texture(self):
        objs = self.sel_objs()
        if not objs:
            return
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Textura", [("m", "Projeção", "topo", ["topo", "frente", "lado", "cilíndrica"])],
                     note=f"Imagem: {it.name}. Densifique a malha (Malha ▸ Densificar) para mais detalhe.")
        if v:
            for o in objs:
                o.vcolors = (id(o.mesh), texture_colors(o.mesh, it.data, v["m"]))
            self.view3d.refresh(); self.tabs.setCurrentWidget(self.view3d)

    def vw_show_all(self):
        for o in self.doc.objects:
            o.visible = True
        self.doc.changed()

    def vw_hide_sel(self):
        for o in self.sel_objs() or []:
            o.visible = False
        self.doc.changed()

    def vw_isolate(self):
        sel = {o.id for o in (self.sel_objs() or [])}
        if sel:
            for o in self.doc.objects:
                o.visible = o.id in sel
            self.doc.changed(); self.view3d.fit()

    def vw_theme(self):
        app = QApplication.instance()
        self._dark = not getattr(self, "_dark", True)
        if self._dark:
            app.setPalette(app.style().standardPalette()); app.setStyleSheet("")
        else:
            p = QPalette()
            for role, c in [(QPalette.Window, "#f4f4f2"), (QPalette.WindowText, "#1d1d1f"), (QPalette.Base, "#ffffff"),
                            (QPalette.AlternateBase, "#f0f0ee"), (QPalette.Text, "#1d1d1f"), (QPalette.Button, "#ececea"),
                            (QPalette.ButtonText, "#1d1d1f"), (QPalette.Highlight, "#2f6fd6"), (QPalette.HighlightedText, "#ffffff")]:
                p.setColor(role, QColor(c))
            app.setPalette(p)
        self.view3d.set_background(QColor(38, 41, 48) if self._dark else QColor(232, 234, 238))
        self.view3d.grid.setColor((255, 255, 255, 50) if self._dark else (0, 0, 0, 60))

    def vw_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def vw_window_opacity(self):
        v = self.ask("Transparência da janela", [("o", "Opacidade (%)", float(round(self.windowOpacity() * 100)), (20, 100))])
        if v:
            self.setWindowOpacity(v["o"] / 100)

    def vw_units(self):
        v = self.ask("Unidades", [("u", "Unidade", getattr(self.canvas, "unit", "mm"), ["mm", "cm", "dm", "m", "polegada"])],
                     note="O modelo continua em milímetros; muda apenas a forma como as medições da régua 2D são mostradas.")
        if v:
            self.canvas.unit = v["u"]
            self.canvas.unit_k = {"mm": 1, "cm": 0.1, "dm": 0.01, "m": 0.001, "polegada": 1 / 25.4}[v["u"]]
