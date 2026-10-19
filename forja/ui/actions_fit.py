"""Menu Ajuste: superfícies, formas 2D, objetos 3D, curvatura e isolinhas."""
import numpy as np
from ..core import fitting as ft, shapes2d as s2
from .widgets import TextDialog


class FitActions:
    def menus_fit(self):
        return [("Aj&uste", [
            ("Imagem / mapa de altura", [
                ("Plano (Ax + By + Cz = D)…", self.act_fit_plane_img),
                ("Polinómio (ordem 0–5)…", self.act_fit_poly_img),
                ("Plano com peso (imagem de pesos)…", lambda: self.act_fit_poly_img(weighted=True, order=1)),
                ("Curvatura…", self.act_curv_img),
                ("Isolinhas → formas 2D…", self.act_isolines),
            ]),
            ("Formas 2D selecionadas", [
                ("Círculo de melhor ajuste", lambda: self.act_fit2d("círculo")),
                ("Elipse de melhor ajuste", lambda: self.act_fit2d("elipse")),
                ("Reta de melhor ajuste", lambda: self.act_fit2d("reta")),
                None,
                ("Caixa envolvente mínima", lambda: self.act_fit2d("caixa mínima")),
                ("Círculo envolvente mínimo", lambda: self.act_fit2d("círculo mínimo")),
                ("Invólucro convexo", lambda: self.act_fit2d("invólucro convexo")),
                ("Triângulo envolvente mínimo", lambda: self.act_fit2d("triângulo mínimo")),
            ]),
            ("Objeto 3D", [
                ("Plano de melhor ajuste", lambda: self.act_fit3d("plano")),
                ("Esfera de melhor ajuste", lambda: self.act_fit3d("esfera")),
                ("Cilindro vertical de melhor ajuste", lambda: self.act_fit3d("cilindro")),
                None,
                ("Mapa de curvatura (cores)…", self.act_curv_mesh),
                ("Limpar cores", self.act_clear_colors),
            ]),
        ])]

    # --- imagens ----------------------------------------------------------
    def act_fit_plane_img(self):
        it = self.cur_image()
        if not it:
            return
        Z = np.asarray(it.data, float)
        if Z.ndim == 3:
            Z = Z.mean(2)
        (A, B, C, D), S, rms = ft.plane_equation(Z, mask=it.meta.get("mask"))
        self.add_image(f"plano({it.name})", S, show=False)
        self.add_image(f"resíduo plano({it.name})", Z - S)
        self.log(f"Plano: {A:.6g}·x + {B:.6g}·y + {C:.6g}·z = {D:.6g}   (RMS do resíduo {rms:.4g})")

    def act_fit_poly_img(self, weighted=False, order=None):
        items = self.sel_images(2 if weighted else 1)
        if not items:
            return
        v = {"o": order} if order is not None else self.ask("Polinómio", [("o", "Ordem (0–5)", 2, (0, 5))])
        if not v:
            return
        Z = np.asarray(items[0].data, float)
        if Z.ndim == 3:
            Z = Z.mean(2)
        W = np.asarray(items[1].data, float) if weighted else None
        S, coef, rms = ft.fit_surface(Z, v["o"], W, items[0].meta.get("mask"))
        self.add_image(f"polinómio {v['o']}({items[0].name})", S, show=False)
        self.add_image(f"resíduo polinómio {v['o']}({items[0].name})", Z - S)
        self.log(f"Polinómio de ordem {v['o']}: {len(coef)} coeficientes, RMS do resíduo {rms:.4g}\n"
                 f"  coeficientes (x,y normalizados 0–1): {np.array2string(coef, precision=5)}")

    def act_curv_img(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Curvatura", [("px", "Tamanho do pixel (mm)", 1.0, (1e-6, 1e6)), ("s", "Suavização (px)", 1.5, (0, 50)),
                                   ("k", "Tipos", "todos", ["todos", "Gauss", "média", "mínima", "máxima", "em X", "em Y"])])
        if not v:
            return
        C = ft.curvature_image(np.asarray(it.data, float) if it.data.ndim == 2 else np.asarray(it.data, float).mean(2), v["px"], v["s"])
        keys = list(C) if v["k"] == "todos" else [v["k"]]
        for i, k in enumerate(keys):
            self.add_image(f"curvatura {k}({it.name})", C[k], show=i == len(keys) - 1)

    def act_isolines(self):
        it = self.cur_image()
        if not it:
            return
        Z = np.asarray(it.data, float)
        if Z.ndim == 3:
            Z = Z.mean(2)
        lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
        v = self.ask("Isolinhas", [("n", "Número de níveis", 10, (1, 500)), ("lo", "Primeiro nível", lo + (hi - lo) / 11),
                                   ("hi", "Último nível", hi - (hi - lo) / 11), ("px", "Tamanho do pixel (mm)", 0.5, (1e-6, 1e6))])
        if not v:
            return
        levels = np.linspace(v["lo"], v["hi"], v["n"])
        shapes = []
        for lv, pts in ft.iso_contours(Z, levels, v["px"]):
            closed = np.allclose(pts[0], pts[-1]) and len(pts) > 3
            s = s2.Shape("pol" if closed else "pl", {"pts": [tuple(p) for p in (pts[:-1] if closed else pts)]},
                         name=f"nível {lv:.4g}")
            shapes.append(s)
        self.canvas.add_shapes(shapes)
        self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()
        self.log(f"{len(shapes)} isolinhas em {v['n']} níveis")

    # --- 2D ---------------------------------------------------------------
    def act_fit2d(self, kind):
        sel = self.canvas.selected_shapes()
        if not sel:
            return self.error("Selecione formas ou pontos no separador 2D.")
        P = np.vstack([ft.geom_points(s.geom) for s in sel if len(ft.geom_points(s.geom))])
        try:
            if kind == "círculo":
                (cx, cy), r, rms = ft.fit_circle(P)
                new = s2.Shape("cir", dict(cx=cx, cy=cy, r=r), name="círculo ajustado")
                msg = f"Círculo: centro ({cx:.4f}, {cy:.4f}), raio {r:.4f}, diâmetro {2 * r:.4f}, RMS {rms:.4g}"
            elif kind == "elipse":
                (cx, cy), (a, b), ang = ft.fit_ellipse(P)
                new = s2.Shape("ell", dict(cx=cx, cy=cy, rx=a, ry=b, angle=ang), name="elipse ajustada")
                msg = f"Elipse: centro ({cx:.4f}, {cy:.4f}), semi-eixos {a:.4f} × {b:.4f}, ângulo {ang:.3f}°"
            elif kind == "reta":
                c, d, (p0, p1), rms = ft.fit_line(P)
                new = s2.Shape("ln", dict(x1=p0[0], y1=p0[1], x2=p1[0], y2=p1[1]), name="reta ajustada")
                msg = f"Reta: ponto ({c[0]:.4f}, {c[1]:.4f}), direção ({d[0]:.5f}, {d[1]:.5f}), ângulo {np.degrees(np.arctan2(d[1], d[0])):.3f}°, RMS {rms:.4g}"
            else:
                r = ft.min_bounding(P, kind)
                if kind == "caixa mínima":
                    new = s2.Shape("rcr", dict(cx=r["cx"], cy=r["cy"], w=r["w"], h=r["h"], angle=r["angle"]), name=kind)
                    msg = f"Caixa mínima: {r['w']:.4f} × {r['h']:.4f}, centro ({r['cx']:.4f}, {r['cy']:.4f}), ângulo {r['angle']:.3f}°"
                elif kind == "círculo mínimo":
                    new = s2.Shape("cir", dict(cx=r["cx"], cy=r["cy"], r=r["r"]), name=kind)
                    msg = f"Círculo mínimo: centro ({r['cx']:.4f}, {r['cy']:.4f}), raio {r['r']:.4f}"
                else:
                    new = s2.Shape("pol", {"pts": [tuple(p) for p in r["pts"]]}, name=kind)
                    msg = f"{kind}: {len(r['pts'])} vértices, área {new.geom.area:.4f}"
        except Exception as e:
            return self.error(e)
        self.canvas.add_shapes([new], select=False)
        self.log(msg); self.statusBar().showMessage(msg)

    # --- 3D ---------------------------------------------------------------
    def act_fit3d(self, kind):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        import trimesh
        P = objs[0].mesh.vertices
        if kind == "plano":
            n, c, rms = ft.fit_plane_points(P)
            d = float(n @ c)
            msg = f"Plano: {n[0]:.6f}x + {n[1]:.6f}y + {n[2]:.6f}z = {d:.6f}  (ponto {np.round(c, 4)}, RMS {rms:.4g} mm)"
            size = objs[0].mesh.scale * 0.8
            pl = trimesh.creation.box(extents=(size, size, 0.2))
            pl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], n)); pl.apply_translation(c)
            o = self.add_mesh(pl, "plano ajustado", place=False); o.alpha = 0.5
        elif kind == "esfera":
            c, r, rms = ft.fit_sphere(P)
            msg = f"Esfera: centro {np.round(c, 4)}, raio {r:.4f}, diâmetro {2 * r:.4f}, RMS {rms:.4g} mm"
            s = trimesh.creation.icosphere(4, r); s.apply_translation(c)
            o = self.add_mesh(s, "esfera ajustada", place=False); o.alpha = 0.4
        else:
            (cx, cy), r, rms = ft.fit_cylinder_z(P)
            z0, z1 = objs[0].mesh.bounds[:, 2]
            msg = f"Cilindro: eixo em ({cx:.4f}, {cy:.4f}), raio {r:.4f}, diâmetro {2 * r:.4f}, RMS {rms:.4g} mm"
            s = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=96); s.apply_translation((cx, cy, (z0 + z1) / 2))
            o = self.add_mesh(s, "cilindro ajustado", place=False); o.alpha = 0.4
        self.view3d.refresh()
        self.log(msg); self.statusBar().showMessage(msg)

    def act_curv_mesh(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        o = objs[0]
        v = self.ask("Curvatura da malha", [("k", "Tipo", "média", ["média", "Gauss"]),
                                            ("r", "Raio de cálculo (mm)", float(round(o.mesh.scale / 50, 3)), (1e-4, 1e5))])
        if not v:
            return
        mesh = o.mesh

        def done(vals):
            cols, (lo, hi) = ft.values_to_colors(vals)
            o.vcolors = (id(mesh), cols)
            self.view3d.refresh()
            self.view3d.show_legend("coolwarm", lo, hi)
            self.log(f"Curvatura {v['k']} de {o.name}: escala de cores {lo:.4g} (azul) a {hi:.4g} (vermelho) 1/mm")
        self.task("A calcular curvatura…", lambda: ft.mesh_curvature(mesh, v["k"], v["r"]), done)

    def act_clear_colors(self):
        for o in self.sel_objs() or []:
            o.vcolors = None
        self.view3d.hide_legend()
        self.view3d.refresh()
