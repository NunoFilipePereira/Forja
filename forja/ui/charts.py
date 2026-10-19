"""Separador de gráficos de dados (matplotlib embebido) e respetivo menu."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFileDialog
from .widgets import TextDialog

KINDS = ["linha", "pontos", "linha e pontos", "barras", "área", "degraus", "histograma"]
CMAPS = ["viridis", "jet", "turbo", "gray", "hot", "plasma", "inferno", "rainbow", "coolwarm", "hsv", "cividis"]


class ChartTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 5), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas)
        self.series = []        # {"x","y","label","kind","color"}
        self.image = None       # mapa 2D {"data","cmap","extent"}
        self.opts = {"title": "", "xlabel": "x", "ylabel": "y", "legend": True, "grid": True, "logx": False, "logy": False,
                     "polar": False, "xlim": None, "ylim": None, "bar_width": 0.8}
        self.draw()

    def clear(self):
        self.series = []; self.image = None; self.draw()

    def draw(self):
        self.fig.clear()
        o = self.opts
        ax = self.fig.add_subplot(111, projection="polar" if o["polar"] else None)
        if self.image is not None:
            im = ax.imshow(self.image["data"], cmap=self.image["cmap"], aspect="auto", origin="upper")
            self.fig.colorbar(im, ax=ax, label=self.image.get("label", ""))
        for s in self.series:
            x, y, k = np.asarray(s["x"]), np.asarray(s["y"]), s["kind"]
            if k != "histograma" and len(x) != len(y):
                x = np.arange(len(y))
            kw = {"label": s["label"]}
            if s.get("color"):
                kw["color"] = s["color"]
            if k == "linha":
                ax.plot(x, y, **kw)
            elif k == "pontos":
                ax.scatter(x, y, s=12, **kw)
            elif k == "linha e pontos":
                ax.plot(x, y, marker="o", markersize=3, **kw)
            elif k == "barras":
                ax.bar(x, y, width=o["bar_width"] * (np.min(np.diff(x)) if len(x) > 1 else 1), **kw)
            elif k == "área":
                ln, = ax.plot(x, y, **kw)
                ax.fill_between(x, y, alpha=0.35, color=ln.get_color())
            elif k == "degraus":
                ax.step(x, y, where="mid", **kw)
            elif k == "histograma":
                ax.hist(y[np.isfinite(y)], bins=s.get("bins", 50), alpha=0.7, **kw)
        if not o["polar"]:
            ax.set_xlabel(o["xlabel"]); ax.set_ylabel(o["ylabel"])
            if o["logx"]: ax.set_xscale("log")
            if o["logy"]: ax.set_yscale("log")
            if o["xlim"]: ax.set_xlim(*o["xlim"])
            if o["ylim"]: ax.set_ylim(*o["ylim"])
        if o["title"]:
            ax.set_title(o["title"])
        if o["grid"]:
            ax.grid(alpha=0.3)
        if o["legend"] and self.series:
            ax.legend()
        self.canvas.draw_idle()

    def to_json(self):
        return json.dumps({"opts": self.opts, "series": [{**s, "x": np.asarray(s["x"], float).tolist(),
                                                          "y": np.asarray(s["y"], float).tolist()} for s in self.series]})

    def from_json(self, text):
        d = json.loads(text)
        self.opts.update(d.get("opts", {})); self.series = d.get("series", []); self.draw()


class ChartActions:
    def menus_chart(self):
        return [("Gr&áficos", [
            ("Importar dados ASCII (colunas)…", self.ch_import),
            ("Série a partir de valores…", self.ch_manual),
            ("Perfil da imagem atual (linha/coluna)…", self.ch_profile),
            ("Imagem A (X) contra imagem B (Y)", self.ch_scatter_images),
            ("Histograma da imagem atual", self.ch_hist_image),
            ("Imagem atual como mapa de cores…", self.ch_image_map),
            ("Área das secções de um objeto 3D ao longo de Z", self.ch_section_areas),
            None,
            ("Tipo das séries…", self.ch_kind),
            ("Títulos e eixos…", self.ch_axes),
            ("Limites dos eixos…", self.ch_limits),
            ("Legenda / grelha / log / polar…", self.ch_toggles),
            ("Ver dados", self.ch_show_data),
            ("Limpar gráfico", lambda: (self.chart.clear(), self.tabs.setCurrentWidget(self.chart))),
            None,
            ("Guardar figura (PNG/SVG/PDF)…", self.ch_save_image),
            ("Guardar dados da figura (.fgs)…", self.ch_save_fgs),
            ("Abrir dados de figura (.fgs)…", self.ch_open_fgs),
        ])]

    def _show_chart(self):
        self.chart.draw(); self.tabs.setCurrentWidget(self.chart)

    def ch_import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Dados", self.last_dir, "Dados (*.txt *.csv *.dat *.asc);;Todos (*)")
        if not p:
            return
        from ..core.imageops2 import load_ascii
        try:
            A = load_ascii(p)
        except Exception as e:
            return self.error(e)
        v = self.ask("Dados", [("dir", "Séries em", "colunas", ["colunas", "linhas"]), ("x", "Primeira coluna/linha é o X", True),
                               ("k", "Tipo", "linha", KINDS)], note=f"{A.shape[0]} linhas × {A.shape[1]} colunas")
        if not v:
            return
        M = A if v["dir"] == "colunas" else A.T
        if v["x"] and M.shape[1] > 1:
            x, Ys = M[:, 0], M[:, 1:]
        else:
            x, Ys = np.arange(M.shape[0]), M
        base = os.path.basename(p)
        for j in range(Ys.shape[1]):
            self.chart.series.append({"x": x, "y": Ys[:, j], "label": f"{base} [{j + 1}]", "kind": v["k"]})
        self._show_chart()

    def ch_manual(self):
        v = self.ask("Série", [("x", "X (separados por espaço/vírgula; vazio = 0,1,2…)", ""), ("y", "Y", "1 4 9 16 25"),
                               ("l", "Nome", "série"), ("k", "Tipo", "linha e pontos", KINDS)])
        if not v:
            return
        try:
            y = np.array([float(t) for t in v["y"].replace(",", " ").split()])
            x = np.array([float(t) for t in v["x"].replace(",", " ").split()]) if v["x"].strip() else np.arange(len(y))
            assert len(x) == len(y)
        except Exception:
            return self.error("Valores inválidos (X e Y têm de ter o mesmo número de valores).")
        self.chart.series.append({"x": x, "y": y, "label": v["l"], "kind": v["k"]})
        self._show_chart()

    def ch_profile(self):
        it = self.cur_image()
        if not it:
            return
        a = np.asarray(it.data, float)
        if a.ndim == 3:
            a = a.mean(2)
        v = self.ask("Perfil", [("d", "Direção", "linha", ["linha", "coluna"]), ("i", "Índice", a.shape[0] // 2, (0, max(a.shape) - 1))])
        if v:
            y = a[min(v["i"], a.shape[0] - 1), :] if v["d"] == "linha" else a[:, min(v["i"], a.shape[1] - 1)]
            self.chart.series.append({"x": np.arange(len(y)), "y": y, "label": f"{it.name} {v['d']} {v['i']}", "kind": "linha"})
            self.chart.opts.update(xlabel="pixel", ylabel="valor")
            self._show_chart()

    def ch_scatter_images(self):
        it = self.sel_images(2)
        if not it:
            return
        a, b = np.asarray(it[0].data, float), np.asarray(it[1].data, float)
        if a.shape != b.shape:
            return self.error("As duas imagens têm de ter o mesmo tamanho.")
        a, b = a.ravel(), b.ravel()
        if a.size > 50000:
            k = np.random.default_rng(0).choice(a.size, 50000, replace=False); a, b = a[k], b[k]
        self.chart.series.append({"x": a, "y": b, "label": f"{it[0].name} × {it[1].name}", "kind": "pontos"})
        self.chart.opts.update(xlabel=it[0].name, ylabel=it[1].name)
        c = np.corrcoef(a[np.isfinite(a) & np.isfinite(b)], b[np.isfinite(a) & np.isfinite(b)])[0, 1]
        self.log(f"Correlação entre {it[0].name} e {it[1].name}: {c:.4f}")
        self._show_chart()

    def ch_hist_image(self):
        it = self.cur_image()
        if it:
            self.chart.series.append({"x": [0], "y": np.asarray(it.data, float).ravel(), "label": it.name, "kind": "histograma", "bins": 100})
            self.chart.opts.update(xlabel="valor", ylabel="pixels")
            self._show_chart()

    def ch_image_map(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Mapa de cores", [("c", "Mapa", "viridis", CMAPS), ("l", "Legenda da barra de cores", "")])
        if v:
            a = np.asarray(it.data, float)
            self.chart.image = {"data": a.mean(2) if a.ndim == 3 else a, "cmap": v["c"], "label": v["l"]}
            self._show_chart()

    def ch_section_areas(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        from ..core.mesh2 import section_areas
        pos, A = section_areas(objs[0].mesh, 2, 150)
        self.chart.series.append({"x": pos, "y": A, "label": f"secção de {objs[0].name}", "kind": "linha"})
        self.chart.opts.update(xlabel="Z (mm)", ylabel="área da secção (mm²)")
        self._show_chart()

    def ch_kind(self):
        if not self.chart.series:
            return self.error("O gráfico não tem séries.")
        v = self.ask("Tipo", [("k", "Tipo de todas as séries", self.chart.series[-1]["kind"], KINDS),
                              ("w", "Largura das barras (fração)", float(self.chart.opts["bar_width"]), (0.05, 1))])
        if v:
            for s in self.chart.series:
                if s["kind"] != "histograma" or v["k"] == "histograma":
                    s["kind"] = v["k"]
            self.chart.opts["bar_width"] = v["w"]; self._show_chart()

    def ch_axes(self):
        o = self.chart.opts
        v = self.ask("Títulos", [("t", "Título", o["title"]), ("x", "Eixo X", o["xlabel"]), ("y", "Eixo Y", o["ylabel"])])
        if v:
            o.update(title=v["t"], xlabel=v["x"], ylabel=v["y"]); self._show_chart()

    def ch_limits(self):
        o = self.chart.opts
        v = self.ask("Limites", [("auto", "Automático", o["xlim"] is None), ("x0", "X mín", float((o["xlim"] or (0, 1))[0])),
                                 ("x1", "X máx", float((o["xlim"] or (0, 1))[1])), ("y0", "Y mín", float((o["ylim"] or (0, 1))[0])),
                                 ("y1", "Y máx", float((o["ylim"] or (0, 1))[1]))])
        if v:
            o["xlim"] = None if v["auto"] else (v["x0"], v["x1"]); o["ylim"] = None if v["auto"] else (v["y0"], v["y1"])
            self._show_chart()

    def ch_toggles(self):
        o = self.chart.opts
        v = self.ask("Opções", [("l", "Legenda", o["legend"]), ("g", "Grelha", o["grid"]), ("lx", "X logarítmico", o["logx"]),
                                ("ly", "Y logarítmico", o["logy"]), ("p", "Polar (X em radianos)", o["polar"])])
        if v:
            o.update(legend=v["l"], grid=v["g"], logx=v["lx"], logy=v["ly"], polar=v["p"]); self._show_chart()

    def ch_show_data(self):
        lines = []
        for s in self.chart.series:
            lines.append(f"# {s['label']} ({s['kind']})")
            x, y = np.asarray(s["x"]), np.asarray(s["y"])
            n = min(len(x), len(y), 2000) if s["kind"] != "histograma" else 0
            lines += [f"{x[i]:.6g}\t{y[i]:.6g}" for i in range(n)]
        TextDialog.show_text(self, "Dados do gráfico", "\n".join(lines) or "(vazio)")

    def ch_save_image(self):
        p, _ = QFileDialog.getSaveFileName(self, "Guardar figura", os.path.join(self.last_dir, "grafico.png"),
                                           "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if p:
            self.chart.fig.savefig(p, dpi=200); self.log(f"Figura guardada: {p}")

    def ch_save_fgs(self):
        p, _ = QFileDialog.getSaveFileName(self, "Dados da figura", os.path.join(self.last_dir, "figura.fgs"), "Figura (*.fgs)")
        if p:
            open(p, "w").write(self.chart.to_json())

    def ch_open_fgs(self):
        p, _ = QFileDialog.getOpenFileName(self, "Dados da figura", self.last_dir, "Figura (*.fgs);;Todos (*)")
        if p:
            try:
                self.chart.from_json(open(p).read()); self.tabs.setCurrentWidget(self.chart)
            except Exception as e:
                self.error(f"Ficheiro de figura inválido: {e}")
