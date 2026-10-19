"""Menu Aspeto — estilo de desenho: superfície / arestas / pontos (mostrar, escala de cor, tipo e
espessura de linha, marcador, tamanho), amostragem, transparência, barra de cores, legenda, iluminação, textura, nome,
cor de fundo, esquema de cores e propriedades dos eixos."""
import copy
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QFontDialog, QLabel
from ..core import colormaps as CM

DEFAULT_STYLE = {"surface": {"show": True, "scheme": "True color"},
                 "edge": {"show": False, "scheme": "True color", "dash": "Solid line", "width": 1.0},
                 "point": {"show": False, "scheme": "True color", "marker": "Dot", "size": 4.0},
                 "sampling": 1}


class StyleActions:
    # ------------------------------------------------------------------ auxiliares
    def _style_objs(self):
        objs = [o for o in self.doc.objects if o.id in self.view3d.selected_ids]
        return objs or [o for o in self.doc.objects if o.visible]

    def st_set(self, part, key, value=None, toggle=False):
        objs = self._style_objs()
        if not objs:
            return self.statusBar().showMessage("Não há objetos 3D.")
        for o in objs:
            st = copy.deepcopy(getattr(o, "style", None) or DEFAULT_STYLE)
            d = st if part is None else st.setdefault(part, {})
            d[key] = (not d.get(key, DEFAULT_STYLE.get(part, {}).get(key, False))) if toggle else value
            o.style = st
        self.view3d.refresh()
        if part == "surface" and key == "scheme":
            self.st_colorbar(show=True, quiet=True)

    def st_ask(self, part, key, title, label, default, rng):
        v = self.ask(title, [("x", label, default, rng)])
        if v:
            self.st_set(part, key, v["x"])

    # ------------------------------------------------------------------ menus
    def menus_style(self):
        def schemes(part):
            return [(CM.PT[s], (lambda checked=False, s=s: self.st_set(part, "scheme", s))) for s in CM.SCHEMES]
        return [("&Aspeto", [
            ("Superfície", [("Mostrar / ocultar", lambda: self.st_set("surface", "show", toggle=True)),
                            ("Escala de cor", schemes("surface"))]),
            ("Arestas", [("Mostrar / ocultar", lambda: self.st_set("edge", "show", toggle=True)),
                         ("Escala de cor", schemes("edge")),
                         ("Tipo de linha", [(CM.DASHES_PT[d], (lambda checked=False, d=d: self.st_set("edge", "dash", d))) for d in CM.DASHES]),
                         ("Espessura da linha…", lambda: self.st_ask("edge", "width", "Espessura da linha", "Espessura (píxeis)", 1.0, (0.5, 20)))]),
            ("Pontos", [("Mostrar / ocultar", lambda: self.st_set("point", "show", toggle=True)),
                        ("Escala de cor", schemes("point")),
                        ("Marcador", [(CM.MARKERS_PT[m], (lambda checked=False, m=m: self.st_set("point", "marker", m))) for m in CM.MARKERS]),
                        ("Tamanho…", lambda: self.st_ask("point", "size", "Tamanho dos pontos", "Tamanho", 4.0, (0.5, 100)))]),
            ("Amostragem…", self.st_sampling),
            ("Transparência (0–100)…", self.st_transparency),
            None,
            ("Barra de cores", [("Mostrar / ocultar", lambda: self.st_colorbar()), ("Tamanho…", self.st_colorbar_size)]),
            ("Legenda", [("Mostrar / ocultar", self.st_legend), ("Tipo de letra…", self.st_legend_font)]),
            ("Iluminação", [("Ligar / desligar", self.st_light_toggle), ("Luz ambiente (0–100)…", self.st_ambient)]),
            ("Textura", [("Mostrar / ocultar", self.st_texture_toggle), ("Abrir imagem (imagem atual)…", self.vw_texture),
                         ("Remover", self.st_texture_remove)]),
            ("Eixos", [("Eixos em caixa — ligar / desligar", lambda: self.view3d.set_axes_box(not self.view3d.axes_box)),
                       ("Ocultar / mostrar o triedro na origem", lambda: self.view3d.axis.setVisible(not self.view3d.axis.visible())),
                       ("Propriedades…", self.st_axis_props)]),
            None,
            ("Tamanho do texto 3D…", self.st_text3d_size),
            ("Largura das barras (gráficos)…", self.st_bar_width),
            ("Nome…", self.st_name),
            ("Cor de fundo…", self.st_background),
            ("Esquema de cores", [("Predefinido", lambda: self.st_scheme("default")), ("Cinzento", lambda: self.st_scheme("gray"))]),
            ("Repor o aspeto original", self.st_reset),
        ])]

    # ------------------------------------------------------------------ Ver
    def menus_view_extra(self):
        v3 = self.view3d
        return [("&Ver", [
            None,
            ("Zoom", [("1 : 1 (ajustar)", v3.fit), ("Aproximar", lambda: v3.zoom(0.8), "Ctrl+="), ("Afastar", lambda: v3.zoom(1.25), "Ctrl+-")]),
            ("Viewport (margens)", [(f"{pt} {s_}", (lambda checked=False, side=side, sg=sg: v3.viewport_shift(side, sg)))
                                    for side, pt in (("Left", "Esquerda"), ("Right", "Direita"), ("Top", "Cima"), ("Bottom", "Baixo"))
                                    for s_, sg in (("+", 1), ("−", -1))]),
            ("Ângulo de vista", [("Livre (atual)", lambda: None), ("Normal", lambda: v3.view("Isométrica")),
                                 ("X", lambda: (v3.setCameraPosition(elevation=0, azimuth=0), v3.fit())),
                                 ("Y", lambda: (v3.setCameraPosition(elevation=0, azimuth=90), v3.fit())),
                                 ("Z", lambda: (v3.setCameraPosition(elevation=90, azimuth=-90), v3.fit()))]),
            ("Modo do rato", [("Rodar a vista (Rotate)", lambda: self.vw_drag("Rotate")),
                              ("Deslocar a vista (Shift)", lambda: self.vw_drag("Shift")),
                              ("Arrastar objetos (Float)", lambda: self.vw_drag("Float")),
                              ("Mover a luz (Move light)", lambda: self.vw_drag("Move light"))]),
            ("Copiar a vista (imagem)", self.vw_copy_view),
            ("Colar imagem como nova imagem", self.vw_paste_image),
            ("Repor coordenadas da vista", lambda: (v3.setCameraPosition(elevation=30, azimuth=-45), v3.fit())),
            ("Saída do programa (registo)", lambda: (self.log_dock.show(), self.log_dock.raise_()) if hasattr(self, "log_dock") else None),
        ])]

    def vw_drag(self, mode):
        v3 = self.view3d
        v3.drag_mode = mode
        if not getattr(self, "_drag_connected", False):
            v3.object_dragged.connect(self._float_object)
            self._drag_connected = True
        self.statusBar().showMessage({"Rotate": "Arrastar roda a vista", "Shift": "Arrastar desloca a vista",
                                      "Float": "Arrastar move o objeto em que clicar",
                                      "Move light": "Arrastar muda a direção da luz"}[mode], 6000)

    def _float_object(self, oid, d):
        for o in self.doc.objects:
            if o.id == oid:
                if not getattr(self, "_float_snap", False):
                    self.doc.snapshot(); self._float_snap = True
                o.mesh = o.mesh.copy(); o.mesh.apply_translation(np.asarray(d, float))
                self.doc.changed()
                from PySide6.QtCore import QTimer
                QTimer.singleShot(600, lambda: setattr(self, "_float_snap", False))
                break

    def vw_copy_view(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setImage(self.view3d.grabFramebuffer())
        self.statusBar().showMessage("Vista 3D copiada para a área de transferência", 4000)

    def vw_paste_image(self):
        from PySide6.QtWidgets import QApplication
        img = QApplication.clipboard().image()
        if img.isNull():
            return self.error("A área de transferência não tem nenhuma imagem.")
        img = img.convertToFormat(img.Format.Format_RGB888)
        a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 3]
        self.add_image("colada", a.reshape(img.height(), img.width(), 3).copy())

    # ------------------------------------------------------------------ ações
    def st_sampling(self):
        v = self.ask("Amostragem", [("x", "Intervalo de amostragem (1 – 1000)", 1, (1, 1000))],
                     note="Mostra um ponto em cada N (útil em nuvens de pontos grandes).")
        if v:
            self.st_set(None, "sampling", int(v["x"]))

    def st_transparency(self):
        objs = self._style_objs()
        if not objs:
            return
        cur = int(round((1 - objs[0].alpha) * 100))
        v = self.ask("Transparência", [("t", "Transparência (0 – 100)", cur, (0, 95))])
        if v:
            for o in objs:
                o.alpha = max(0.05, 1 - v["t"] / 100)
            self.doc.changed(); self.view3d.refresh()

    def st_colorbar(self, show=None, quiet=False):
        v3 = self.view3d
        objs = self._style_objs()
        o = objs[0] if objs else None
        on = (not v3.legend.isVisible()) if show is None else show
        if not on or o is None:
            v3.hide_legend(); return
        sch = (getattr(o, "style", None) or DEFAULT_STYLE)["surface"].get("scheme", "True color")
        if sch in ("True color", "Solid color"):
            if not quiet:
                self.statusBar().showMessage("Escolha primeiro uma escala de cor em Aspeto ▸ Superfície ▸ Escala de cor.")
            v3.hide_legend(); return
        z = o.mesh.vertices[:, 2]
        v3.legend.solid = o.color
        v3.show_legend(sch, float(z.min()), float(z.max()))

    def st_colorbar_size(self):
        v3 = self.view3d
        cur = getattr(v3, "legend_size", None) or (500, 26)
        v = self.ask("Tamanho da barra de cores", [("s", "Largura e altura (píxeis)", f"{cur[0]} {cur[1]}")],
                     note="Exemplo: 15 200 (largura 15, altura 200 píxeis — fica vertical).")
        if v:
            try:
                w, h = [int(float(x)) for x in str(v["s"]).split()[:2]]
            except Exception:
                return self.error("Escreva a largura e a altura, por exemplo: 15 200")
            v3.legend_size = (w, h); v3._place_legend()

    def st_legend(self):
        v3 = self.view3d
        lab = getattr(v3, "_names_legend", None)
        if lab is None:
            lab = QLabel(v3); lab.setStyleSheet("color: #ddd; background: rgba(0,0,0,90); padding: 4px")
            v3._names_legend = lab
        if lab.isVisible():
            lab.hide(); return
        rows = []
        for o in self.doc.objects:
            if o.visible:
                c = QColor.fromRgbF(*o.color[:3]).name()
                rows.append(f"<span style='color:{c}'>■</span> {o.name}")
        lab.setText("<br>".join(rows) or "(sem objetos)")
        lab.adjustSize(); lab.move(v3.width() - lab.width() - 12, 12); lab.show(); lab.raise_()

    def st_legend_font(self):
        v3 = self.view3d
        lab = getattr(v3, "_names_legend", None)
        ok, f = QFontDialog.getFont(lab.font() if lab else self.font(), self)
        if ok and f is not None:
            if lab is None:
                self.st_legend(); lab = v3._names_legend
            lab.setFont(f); lab.adjustSize()

    def st_light_toggle(self):
        from .view3d import LIT
        cur = list(LIT.uniformData.get("u_light", [0.3, 0.55, 0.18, 0.4, 0.7, 0.6]))
        if getattr(self, "_light_off", None) is None:
            self._light_off = cur
            LIT["u_light"] = [1.0, 0.0, 0.0] + cur[3:]
            self.statusBar().showMessage("Iluminação desligada (cores planas)")
        else:
            LIT["u_light"] = self._light_off; self._light_off = None
            self.statusBar().showMessage("Iluminação ligada")
        self.view3d.update()

    def st_ambient(self):
        from .view3d import LIT
        cur = list(LIT.uniformData.get("u_light", [0.3, 0.55, 0.18, 0.4, 0.7, 0.6]))
        v = self.ask("Luz ambiente", [("x", "Intensidade da luz ambiente (0 – 100)", int(round(cur[0] * 100)), (0, 100))])
        if v:
            cur[0] = v["x"] / 100
            LIT["u_light"] = cur; self.view3d.update()

    def st_texture_toggle(self):
        for o in self._style_objs():
            if o.vcolors is not None:
                o._tex_hidden, o.vcolors = o.vcolors, None
            elif getattr(o, "_tex_hidden", None) is not None:
                o.vcolors, o._tex_hidden = o._tex_hidden, None
        self.view3d.refresh()

    def st_texture_remove(self):
        for o in self._style_objs():
            o.vcolors = None; o._tex_hidden = None
        self.view3d.refresh()

    def st_text3d_size(self):
        """Tamanho do texto 3D (marcas e títulos dos eixos e etiquetas na vista 3D)."""
        v3 = self.view3d
        v = self.ask("Tamanho do texto 3D", [("s", "Tamanho (pontos)", int(getattr(v3, "text3d_size", 11)), (4, 72))])
        if v:
            v3.text3d_size = int(v["s"])
            self._draw_axes()

    def st_bar_width(self):
        """Largura das barras para os gráficos de barras."""
        cur = float(self.chart.opts.get("bar_width", 0.8))
        v = self.ask("Largura das barras", [("w", "Largura (fração do intervalo)", cur, (0.05, 1.0))])
        if v:
            self.chart.opts["bar_width"] = v["w"]
            self.chart.draw()

    def st_name(self):
        objs = [o for o in self.doc.objects if o.id in self.view3d.selected_ids]
        if len(objs) != 1:
            return self.error("Selecione um objeto.")
        v = self.ask("Nome", [("n", "Nome do objeto", objs[0].name)])
        if v and v["n"].strip():
            objs[0].name = v["n"].strip(); self.doc.changed()

    def st_background(self):
        c = QColorDialog.getColor(self.view3d.bg if hasattr(self.view3d, "bg") else QColor(40, 40, 40), self, "Cor de fundo")
        if c.isValid():
            self.view3d.set_background(c)

    def st_scheme(self, which):
        if which == "gray":
            self.view3d.set_background(QColor(128, 128, 128))
            for o in self.doc.objects:
                o.style = copy.deepcopy(getattr(o, "style", None) or DEFAULT_STYLE); o.style["surface"]["scheme"] = "Gray"
        else:
            self.view3d.set_background(QColor(0, 0, 0))
            for o in self.doc.objects:
                o.style = None
        self.view3d.refresh()

    def st_reset(self):
        for o in self._style_objs():
            o.style = None
        self.view3d.hide_legend(); self.view3d.refresh()

    # ------------------------------------------------------------------ eixos
    def st_axis_props(self):
        """Grelha, título, marcas automáticas ou escritas (separadas por ;), limites automáticos ou mín/máx, por eixo."""
        v3 = self.view3d
        props = getattr(v3, "axis_props", None) or {}
        lo, hi = self._scene_bounds()
        rows = [("ax", "Eixo", "X", ["X", "Y", "Z"])]
        v = self.ask("Propriedades dos eixos", rows + [
            ("grid", "Grelha", True), ("title", "Título", ""),
            ("auto_tick", "Marcas automáticas", True), ("ticks", "Marcas (separadas por ;)", ";0;10;20;30;40;50;"),
            ("auto_lim", "Limites automáticos", True), ("mn", "Mínimo", float(lo[0])), ("mx", "Máximo", float(hi[0]))],
            note="Para escrever as marcas, desligue “Marcas automáticas”. As marcas ficam igualmente espaçadas ao longo do eixo; "
                 "cada marca pode ter qualquer texto e um espaço é uma marca sem texto.")
        if not v:
            return
        props[v["ax"]] = v
        v3.axis_props = props
        self._draw_axes()

    def _font3d(self, k=1.0):
        from PySide6.QtGui import QFont
        return QFont("Helvetica", int(round(getattr(self.view3d, "text3d_size", 11) * k)))

    def _scene_bounds(self):
        objs = [o for o in self.doc.objects if o.visible and len(o.mesh.vertices)]
        if not objs:
            return np.zeros(3), np.full(3, 100.0)
        return (np.min([o.mesh.bounds[0] for o in objs], 0), np.max([o.mesh.bounds[1] for o in objs], 0))

    def _draw_axes(self):
        """Desenha os eixos com marcas e títulos (texto 3D) segundo as propriedades escolhidas."""
        import pyqtgraph.opengl as gl
        v3 = self.view3d
        for it in getattr(v3, "_axis_items", []):
            if it in v3.items:
                v3.removeItem(it)
        v3._axis_items = []
        v3._extra_labels = []
        props = getattr(v3, "axis_props", {}) or {}
        lo, hi = self._scene_bounds()
        for k, axn in enumerate("XYZ"):
            p = props.get(axn)
            if not p:
                continue
            a0 = lo[k] if p["auto_lim"] else p["mn"]
            a1 = hi[k] if p["auto_lim"] else p["mx"]
            if p["auto_tick"]:
                ticks = [f"{x:g}" for x in np.linspace(a0, a1, 6)]
            else:
                ticks = str(p["ticks"]).split(";")
                if ticks and ticks[0] == "":
                    ticks = ticks[1:]
                if ticks and ticks[-1] == "":
                    ticks = ticks[:-1]
            n = max(len(ticks), 2)
            base = np.array([lo[0], lo[1], lo[2]], float)
            P = []
            for i, txt in enumerate(ticks):
                pos = base.copy(); pos[k] = a0 + (a1 - a0) * i / (n - 1)
                if txt.strip():
                    v3._extra_labels.append((pos.copy(), txt, (220, 220, 220, 255), self._font3d()))
                q = pos.copy(); q[(k + 1) % 3] -= (hi - lo).max() * 0.02
                P += [pos, q]
                if p["grid"]:
                    g2 = pos.copy(); g2[(k + 1) % 3] = hi[(k + 1) % 3]
                    P += [pos, g2]
            line = np.array([base, base]); line[1][k] = a1; line[0][k] = a0
            P += list(line)
            li = gl.GLLinePlotItem(pos=np.asarray(P), color=(0.8, 0.8, 0.8, 0.6), width=1, mode="lines")
            v3.addItem(li); v3._axis_items.append(li)
            if p["title"]:
                pos = base.copy(); pos[k] = (a0 + a1) / 2; pos[(k + 2) % 3] -= (hi - lo).max() * 0.06
                v3._extra_labels.append((pos.copy(), p["title"], (255, 230, 120, 255), self._font3d(1.3)))
        v3.update()
