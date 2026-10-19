"""Janela principal da Forja: espaço 3D, tela 2D, imagens/metrologia e todos os menus."""
import io
import json
import os
import zipfile
import numpy as np
import trimesh
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QColor, QActionGroup
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QDockWidget, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QFileDialog, QMessageBox, QColorDialog, QAbstractItemView,
                               QToolBar, QWidget, QVBoxLayout, QLabel, QInputDialog, QApplication)

from ..core.document import Document, mbounds
from ..i18n import tr
from .. import i18n
from ..core import meshops as mo, shapes2d as s2, nesting, toolpath as tp, fringe as fr, imageops as iop
from .view3d import View3D
from .view2d import Canvas2D, TOOLS
from .viewimage import ImageView, ImageItem
from .widgets import ParamDialog, TextDialog, run_task
from .actions_image import ImageActions
from .actions_fit import FitActions
from .actions_gen import GenActions
from .actions_sheet import SheetActions
from .actions_metro import MetroActions
from .actions_ct import CTActions
from .actions_view import ViewActions
from .actions_vision import VisionActions
from .actions_device import DeviceActions
from .actions_mesh2 import Mesh2Actions
from .actions_draw2 import Draw2Actions
from .charts import ChartTab, ChartActions
from .actions_image3 import Image3Actions
from .actions_metro3 import Metro3Actions
from .actions_import import ImportActions
from .actions_style import StyleActions
from .actions_param import ParamActions

MESH_FILTER = "Malhas 3D (*.stl *.obj *.ply *.off *.3mf *.glb *.gltf);;Todos (*)"
IMG_FILTER = ("Imagens e dados (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.s8 *.u8 *.s16 *.u16 *.s32 *.u32 *.f32 *.flt *.dbl "
              "*.raw *.asc *.csv *.dat *.dcm *.avi *.mp4 *.mov);;Todos (*)")
D2_FILTER = "Desenho 2D (*.dxf *.svg *.txt);;Todos (*)"
PROJ_FILTER = "Projeto Forja (*.forja)"


class MainWindow(ImportActions, StyleActions, ParamActions, ImageActions, FitActions, GenActions, SheetActions, MetroActions, CTActions, ViewActions, VisionActions, DeviceActions, Mesh2Actions, Draw2Actions, ChartActions, Image3Actions, Metro3Actions, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forja — prototipagem rápida")
        self.resize(1440, 900)
        self.settings = QSettings("Forja", "Forja")
        self.doc = Document()
        self.images: list[ImageItem] = []
        self.toolpaths = []
        self.last_dir = self.settings.value("last_dir", os.path.expanduser("~"))

        # --- centro: separadores -----------------------------------------
        self.tabs = QTabWidget()
        self.view3d = View3D(self.doc)
        self.canvas = Canvas2D()
        self.imgview = ImageView()
        self.tabs.addTab(self.view3d, tr("3D"))
        self.tabs.addTab(self.canvas, tr("2D / Corte"))
        self.tabs.addTab(self.imgview, tr("Imagem / Metrologia"))
        self.chart = ChartTab()
        self.tabs.addTab(self.chart, tr("Gráficos"))
        self.setCentralWidget(self.tabs)

        # --- painéis -------------------------------------------------------
        self.obj_list = QListWidget(); self.obj_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.obj_list.itemSelectionChanged.connect(self._obj_sel_changed)
        self.obj_list.itemChanged.connect(self._obj_item_changed)
        self.obj_list.itemDoubleClicked.connect(lambda it: self.act_properties())
        self.obj_dock = self._dock("Objetos 3D", self.obj_list, Qt.LeftDockWidgetArea)

        self.img_list = QListWidget(); self.img_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.img_list.currentRowChanged.connect(self._img_current)
        self.img_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.img_list.model().rowsMoved.connect(self._img_reordered)
        self.img_dock = self._dock("Imagens", self.img_list, Qt.LeftDockWidgetArea)

        self.logw = QPlainTextEdit(); self.logw.setReadOnly(True); self.logw.setMaximumBlockCount(3000)
        self.log_dock = self._dock("Registo", self.logw, Qt.BottomDockWidgetArea)
        self.setup_param_panel()
        self.setAcceptDrops(True)

        self.doc.listeners.append(self._refresh_obj_list)
        self.view3d.picked.connect(self._picked3d)
        self.canvas.message.connect(lambda m: self.statusBar().showMessage(m))
        self.canvas.changed.connect(lambda: None)

        self._build_menus()
        self._build_extra_menus()
        self._build_toolbar()
        self.statusBar().showMessage(tr("Pronto. Abra um modelo (⌘O) ou experimente Ficheiro ▸ Exemplos."))
        self.log("Forja iniciada.")

    # ======================================================================
    # utilitários
    # ======================================================================
    def _dock(self, title, w, area):
        d = QDockWidget(tr(title), self); d.setWidget(w); d.setObjectName(title)
        self.addDockWidget(area, d)
        return d

    def log(self, msg):
        self.logw.appendPlainText(str(msg))

    def error(self, msg):
        self.log("ERRO: " + str(msg).split("\n")[0])
        from .. import __main__ as entry
        import datetime
        with open(entry.LOG, "a") as f:
            f.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} (mensagem) ===\n{msg}\n")
        QMessageBox.warning(self, "Forja", str(msg).split("\n\n")[0])

    def ask(self, title, fields, note=None):
        return ParamDialog.ask(self, title, fields, note)

    def task(self, label, fn, done):
        run_task(self, label, fn, done, self.error)

    def sel_objs(self, n=1, exact=False):
        rows = sorted(i.row() for i in self.obj_list.selectedIndexes())
        objs = [self.doc.objects[r] for r in rows if r < len(self.doc.objects)]
        if not objs and len(self.doc.objects) == 1:
            objs = list(self.doc.objects)
        if len(objs) < n or (exact and len(objs) != n):
            self.error(f"Selecione {'exatamente' if exact else 'pelo menos'} {n} objeto(s): clique nele na vista 3D (Shift+clique para vários) ou na lista Objetos 3D.")
            return None
        return objs

    def sel_images(self, n=1):
        rows = sorted(i.row() for i in self.img_list.selectedIndexes())
        items = [self.images[r] for r in rows]
        if not items and self.img_list.currentRow() >= 0:
            items = [self.images[self.img_list.currentRow()]]
        if len(items) < n:
            self.error(f"Selecione pelo menos {n} imagem(ns) na lista Imagens.")
            return None
        return items

    def cur_image(self):
        r = self.img_list.currentRow()
        if r < 0:
            self.error("Nenhuma imagem selecionada. Abra uma imagem ou use Ficheiro ▸ Exemplos.")
            return None
        it = self.images[r]
        if it.meta.get("volume") and it is self.imgview.item:
            # num volume, as operações de imagem atuam sobre o corte visível
            return ImageItem(f"{it.name} corte {self.imgview.vol_slider.value()}", np.array(self.imgview.current_slice()))
        return it

    def add_image(self, name, data, meta=None, show=True):
        it = ImageItem(name, data, meta)
        self.images.append(it)
        self.img_list.addItem(name)
        if show:
            self.img_list.clearSelection()
            self.img_list.setCurrentRow(len(self.images) - 1)
            self.tabs.setCurrentWidget(self.imgview)
            if hasattr(self, "img_dock") and getattr(getattr(self, "tool_panel", None), "spec", None) is None:
                self.img_dock.raise_()
        return it

    def add_mesh(self, mesh, name, select=True, place=True):
        if place:
            self._place_new(mesh)
        o = self.doc.add(mesh, name)
        if select:
            self._select_objs([o])
        return o

    def _select_objs(self, objs):
        ids = {o.id for o in objs}
        self.obj_list.blockSignals(True)
        for i, o in enumerate(self.doc.objects):
            self.obj_list.item(i).setSelected(o.id in ids)
        self.obj_list.blockSignals(False)
        self._obj_sel_changed()

    def _picked3d(self, oid, add):
        objs = [o for o in self.doc.objects if o.id == oid]
        if add:
            cur = [o for o in self.doc.objects if o.id in self.view3d.selected_ids]
            objs = [o for o in cur if o.id != oid] + ([] if oid in {c.id for c in cur} else objs)
        self._select_objs(objs)
        if not objs:
            self.statusBar().showMessage("Nada selecionado. Clique num objeto (Shift+clique para vários).")

    def _place_new(self, mesh):
        """Coloca um objeto novo no centro da mesa, pousado no chão; se já houver objetos, ao lado deles."""
        if len(mesh.vertices):
            b0 = mbounds(mesh)
            mesh.apply_translation((-(b0[0][0] + b0[1][0]) / 2, -(b0[0][1] + b0[1][1]) / 2, -b0[0][2]))
        others = [mbounds(o.mesh) for o in self.doc.objects if o.visible and len(o.mesh.vertices)]
        if not others or not len(mesh.vertices):
            return mesh
        hi = max(m[1][0] for m in others)
        b = mbounds(mesh)
        if b[0][0] <= hi + 1e-6 and any(
                (b[0][0] < m[1][0] and b[1][0] > m[0][0] and b[0][1] < m[1][1] and b[1][1] > m[0][1]) for m in others):
            mesh.apply_translation((hi + 10 - b[0][0], -(b[0][1] + b[1][1]) / 2, 0))
        return mesh

    def _refresh_obj_list(self):
        sel = {self.doc.objects[i.row()].id for i in self.obj_list.selectedIndexes()
               if i.row() < len(self.doc.objects)} if False else self.view3d.selected_ids
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        for o in self.doc.objects:
            it = QListWidgetItem(f"{o.name}   ({len(o.mesh.faces)} faces)")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(Qt.Checked if o.visible else Qt.Unchecked)
            it.setData(Qt.UserRole, o.id)
            self.obj_list.addItem(it)
            it.setSelected(o.id in sel)
        self.obj_list.blockSignals(False)

    def _obj_sel_changed(self):
        ids = [self.doc.objects[i.row()].id for i in self.obj_list.selectedIndexes() if i.row() < len(self.doc.objects)]
        self.view3d.set_selection(ids)
        objs = [o for o in self.doc.objects if o.id in ids]
        if hasattr(self, "param_panel"):
            gens = [o for o in objs if getattr(o, "gen", None)]
            if len(objs) == 1 or (gens and len({o.gen["group"] for o in gens}) == 1 and len(gens) == len(objs)):
                tgt = gens[0] if gens else objs[0]
                if tgt is not self.param_panel.obj:
                    self.param_panel.show_obj(tgt)
            else:
                self.param_panel.show_obj(None)
        if len(objs) == 1 and getattr(objs[0], "tool", None) and hasattr(self, "reopen_tool"):
            # peça feita por uma ferramenta: o painel volta a mostrar as definições usadas, prontas a ajustar
            self.reopen_tool(objs[0])
        if len(objs) == 1:
            bb = mbounds(objs[0].mesh); e = bb[1] - bb[0]
            self.statusBar().showMessage(f"{objs[0].name}: {e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f} mm, "
                                         f"{len(objs[0].mesh.faces)} faces")

    def _obj_item_changed(self, it):
        oid = it.data(Qt.UserRole)
        for o in self.doc.objects:
            if o.id == oid:
                o.visible = it.checkState() == Qt.Checked
                name = it.text().split("   (")[0]
                o.name = name
        self.view3d.refresh()

    def _img_current(self, row):
        self.imgview.show_item(self.images[row] if 0 <= row < len(self.images) else None)

    def _img_reordered(self, *a):
        names = [self.img_list.item(i).text() for i in range(self.img_list.count())]
        pool = list(self.images)
        new = []
        for n in names:
            for it in pool:
                if it.name == n:
                    new.append(it); pool.remove(it); break
        self.images = new + pool

    def _dir(self, path):
        self.last_dir = os.path.dirname(path)
        self.settings.setValue("last_dir", self.last_dir)

    # ======================================================================
    # menus
    # ======================================================================
    def _menu(self, bar, title, entries):
        m = bar.addMenu(tr(title))
        m.setProperty("key", title.replace("&", ""))
        self._fill(m, entries)
        return m

    def _fill(self, m, entries):
        for e in entries:
            if e is None:
                m.addSeparator()
            elif isinstance(e[1], list):
                self._menu(m, e[0], e[1])
            else:
                a = QAction(tr(e[0]), self)
                a.triggered.connect(self._safe(tr(e[0]), e[1]))
                if len(e) > 2 and e[2]:
                    a.setShortcut(QKeySequence(e[2]))
                m.addAction(a)

    def _safe(self, label, fn):
        """Qualquer erro inesperado numa ação vira uma mensagem clara (e fica em erros.log), sem fechar o programa."""
        def run(*_):
            try:
                fn()
            except Exception as e:
                import traceback
                self.error(f"{label.replace('…', '')}: {e}\n\n{traceback.format_exc()}")
        return run

    def _extend_menu(self, title, entries):
        """Acrescenta entradas a um menu de topo existente (ou cria-o antes de Ajuda)."""
        key = title.replace("&", "")
        for act in self.menuBar().actions():
            if act.menu() and act.menu().property("key") == key:
                self._fill(act.menu(), entries)
                return
        help_act = [a for a in self.menuBar().actions() if a.menu() and a.menu().property("key") == "Ajuda"]
        from PySide6.QtWidgets import QMenu
        m = QMenu(tr(title), self)
        m.setProperty("key", key)
        self._fill(m, entries)
        if help_act:
            self.menuBar().insertMenu(help_act[0], m)
        else:
            self.menuBar().addMenu(m)

    def menus_lang(self):
        entries = [(i18n.NAMES.get(c, c), (lambda checked=False, c=c: self.set_language(c))) for c in i18n.available()]
        return [("A&juda", [None, ("Língua", entries)])]

    def set_language(self, code):
        self.settings.setValue("lang", code)
        QMessageBox.information(self, "Forja", "A língua muda quando voltar a abrir a Forja.\n"
                                "The language changes the next time you open Forja.")

    def _build_extra_menus(self):
        for provider in ("menus_image", "menus_fit", "menus_gen", "menus_sheet", "menus_metro", "menus_ct",
                         "menus_view", "menus_vision", "menus_device", "menus_mesh2", "menus_draw2", "menus_chart",
                         "menus_image3", "menus_metro3", "menus_batch", "menus_style", "menus_view_extra", "menus_import", "menus_lang"):
            fn = getattr(self, provider, None)
            if fn:
                for title, entries in fn():
                    self._extend_menu(title, entries)

    def _build_menus(self):
        mb = self.menuBar()
        self._menu(mb, "&Ficheiro", [
            ("&Novo projeto", self.act_new, "Ctrl+N"),
            ("&Abrir…", self.act_open, "Ctrl+O"),
            ("Abrir projeto…", self.act_open_project),
            ("&Guardar projeto…", self.act_save_project, "Ctrl+S"),
            None,
            ("Importar", [
                ("Malha 3D…", self.act_import_mesh),
                ("Imagem(ns)…", self.act_import_images),
                ("Imagem bruta (s8/u8/s16/u16/f32)…", self.act_import_raw),
                ("Desenho 2D (DXF/SVG/dados)…", self.act_import_2d),
            ]),
            ("Exportar", [
                ("Malha selecionada (STL/OBJ/PLY/3MF/GLB)…", self.act_export_mesh, "Ctrl+E"),
                ("Cada objeto em ficheiro separado…", self.act_export_each),
                ("Desenho 2D (DXF/SVG)…", self.act_export_2d),
                ("Dados das formas 2D (texto)…", self.act_export_data),
                ("Imagem atual…", self.act_export_image),
                ("Captura da vista 3D (PNG)…", self.act_snapshot),
            ]),
            None,
            ("Exemplos", [
                ("Modelo 3D de exemplo", self.demo_model),
                ("Franjas simuladas (luz estruturada)", self.demo_fringe),
                ("Pilha de foco simulada (luz branca)", self.demo_focus),
                ("Silhuetas simuladas (visual hull)", self.demo_silhouettes),
                ("Formas 2D para nesting", self.demo_shapes),
            ]),
            None,
            ("Sair", self.close, "Ctrl+Q"),
        ])
        self._menu(mb, "&Editar", [
            ("&Desfazer", self.act_undo, "Ctrl+Z"),
            None,
            ("Duplicar", self.act_duplicate, "Ctrl+D"),
            ("Apagar", self.act_delete, "Del"),
            ("Selecionar tudo", self.act_select_all, "Ctrl+A"),
            ("Mudar cor…", self.act_color),
            ("Renomear…", self.act_rename),
        ])
        modes = [(m, (lambda checked=False, m=m: self.set_mode(m))) for m in View3D.MODES]
        views = [(v, (lambda checked=False, v=v: self.view3d.view(v))) for v in
                 ["Isométrica", "Cima", "Frente", "Direita", "Esquerda", "Trás", "Baixo"]]
        self._menu(mb, "&Ver", [
            ("Ajustar à janela", self.act_fit, "Ctrl+0"),
            ("Ângulo de visão", views),
            ("Modo de exibição", modes),
            ("Transparência…", self.act_transparency),
            ("Cor de fundo…", self.act_background),
            ("Mostrar saliências (precisam de suporte)", self.act_overhang),
            ("Tamanho da mesa / grelha…", self.act_plate),
            None,
            ("Mapa de cores das imagens…", self.act_cmap),
        ])
        self._menu(mb, "&Criar", [
            ("Caixa…", lambda: self.create_param("caixa")),
            ("Esfera…", lambda: self.create_param("esfera")),
            ("Cilindro…", lambda: self.create_param("cilindro")),
            ("Cone…", lambda: self.create_param("cone")),
            ("Toro…", lambda: self.create_param("toro")),
            ("Cápsula…", lambda: self.create_param("capsula")),
            None,
            ("Texto 3D…", lambda: self.create_param("texto3d")),
            ("Relevo a partir de imagem…", lambda: self.create_param("relevo")),
            ("Litofania…", lambda: self.create_param("litofania")),
            ("Mapa de profundidade → superfície 3D…", self.act_depth_to_3d),
            ("Extrudir formas 2D selecionadas → 3D…", lambda: self.create_param("extrusao")),
            None,
            ("Litofania (18 formas)…", lambda: self.act_create("LITHO"), "Ctrl+Shift+T"),
            ("Photo magic (25 métodos)…", lambda: self.act_create("PHOTO_MAGIC"), "Ctrl+Shift+P"),
            ("Projeção de sombras…", lambda: self.act_create("PROJECTION")),
            ("Garrafa…", lambda: self.act_create("BOTTLE")),
            ("Hélice de bambu…", lambda: self.act_create("BAMBOO")),
            ("Dado…", lambda: self.act_create("DICE")),
            ("Óculos…", lambda: self.act_create("EYEGLASSES")),
            ("Planador…", lambda: self.act_create("GLIDER")),
        ])
        self._menu(mb, "&Transformar", [
            ("Mover…", self.act_move, "Ctrl+T"),
            ("Rodar…", self.act_rotate, "Ctrl+R"),
            ("Escalar…", self.act_scale),
            ("Escalar para tamanho…", self.act_scale_to),
            ("Espelhar…", self.act_mirror),
            ("Transformar (rodar, deslocar, escala, espelhar)…", self.act_transform_panel, "Ctrl+Alt+T"),
            ("Alinhar objetos…", self.act_align_objs),
            None,
            ("Centrar na mesa", self.act_center),
            ("Pousar no chão", self.act_floor),
            ("Assentar na face estável", self.act_layflat),
            ("Organizar objetos na mesa", self.act_arrange),
        ])
        self._menu(mb, "&Malha", [
            ("Cortar…", self.act_cut, "Ctrl+K"),
            ("Converter o modelo…", self.act_convert, "Ctrl+Shift+L"),
            ("Fatiador (G-code para impressora 3D)…", self.act_slicer),
            ("Nesting 3D (arrumar peças em caixas)…", self.act_nesting3d),
            ("Mapa de profundidade (direção e resolução)…", self.act_mesh_depth),
            ("Preencher buracos", self.act_hole_fill),
            ("Suporte mínimo (melhor orientação)", self.act_min_support),
            ("Zero XYZ (canto mínimo na origem)", self.act_zero_xyz),
            ("Estatísticas da malha", self.act_mesh_stats),
            ("Lote: a mesma operação numa pasta de modelos…", self.act_batch),
            ("Partição em grelha (com encaixes)…", self.act_split_grid),
            ("Partição por planos…", self.act_split_planes),
            ("Oco (esvaziar)…", self.act_hollow),
            ("Inflar / encolher…", self.act_inflate),
            None,
            ("Reparar", self.act_repair),
            ("Simplificar…", self.act_simplify),
            ("Densificar…", self.act_densify),
            ("Suavizar…", self.act_smooth),
            ("Inverter normais", self.act_reverse),
            None,
            ("Juntar (união)", lambda: self.act_boolean("union"), "Ctrl+M"),
            ("Subtrair (1º − restantes)", lambda: self.act_boolean("difference")),
            ("Interseção", lambda: self.act_boolean("intersection")),
            ("Combinar sem booleana", self.act_concat),
            ("Separar em componentes", self.act_separate),
            None,
            ("Texto em relevo / gravado…", self.act_imprint),
            ("Fatiar em camadas → 2D…", self.act_slice_layers),
            ("Secção por plano → 2D…", self.act_section),
        ])
        tools = [(TOOLS[k], (lambda checked=False, k=k: self.set_tool(k))) for k in TOOLS]
        self._menu(mb, "&Desenho 2D", [
            ("Ferramenta", tools),
            ("Dados das formas (ver/editar)…", self.act_shape_data, "Ctrl+I"),
            ("Informação da seleção", self.act_shape_info),
            None,
            ("Mover…", self.act2_move), ("Rodar…", self.act2_rotate), ("Escalar…", self.act2_scale),
            ("Copiar para…", self.act2_copy_to), ("Reescalar X / Y…", self.act2_rescale_xy), ("Propriedades da forma…", self.act2_props),
            ("Espelhar", self.act2_mirror), ("Deslocamento (offset)…", self.act2_offset),
            ("Alinhar", [(n, (lambda checked=False, m=m: self.act2_align(m))) for n, m in
                         [("Esquerda", "left"), ("Direita", "right"), ("Cima", "top"), ("Baixo", "bottom"),
                          ("Centro horizontal", "hcenter"), ("Centro vertical", "vcenter")]]),
            ("Booleana", [("União", lambda: self.act2_bool("union")),
                          ("Diferença", lambda: self.act2_bool("difference")),
                          ("Interseção", lambda: self.act2_bool("intersection")),
                          ("Ou exclusivo", lambda: self.act2_bool("xor"))]),
            ("Agrupar", self.act2_group, "Ctrl+G"), ("Desagrupar", self.act2_ungroup, "Ctrl+Shift+G"),
            ("Matriz (copiar em grelha)…", self.act2_array),
            ("Ajuste à grelha (snap)", self.act2_snap),
            None,
            ("Nesting (encaixar em chapas)…", self.act_nesting, "Ctrl+Shift+N"),
            ("Percurso de ferramenta…", self.act_toolpath),
            ("Exportar G-code…", self.act_gcode),
            ("Limpar sobreposições", self.act2_clear_overlay),
        ])
        self._menu(mb, "&Imagem", [
            ("Limiar…", self.act_threshold),
            ("Morfologia…", self.act_morph),
            ("Preencher buracos", lambda: self.img_op("preenchido", iop.fill_holes)),
            ("Esqueleto", lambda: self.img_op("esqueleto", iop.skeleton)),
            ("Filtro…", self.act_filter),
            ("Equalizar histograma", lambda: self.img_op("equalizada", iop.equalize)),
            ("Inverter", lambda: self.img_op("invertida", lambda a: 255 - iop.to_u8(a))),
            ("Converter para cinzento", lambda: self.img_op("cinzento", iop.to_gray)),
            ("Arestas…", self.act_edges),
            None,
            ("Contornos (medir objetos)…", self.act_contours),
            ("Círculos (Hough)…", self.act_circles),
            ("Cantos / pontos-chave…", self.act_corners),
            ("Contornos → formas 2D (vetorizar)…", self.act_vectorize),
            None,
            ("Aritmética entre duas imagens…", self.act_arith),
            ("Rodar…", self.act_img_rotate), ("Redimensionar…", self.act_img_resize),
            ("Recortar…", self.act_img_crop),
            ("Estatísticas", self.act_img_stats),
            ("Remover da lista", self.act_img_remove),
        ])
        self._menu(mb, "Metro&logia", [
            ("Gerar padrões", [("Franjas sinusoidais (desfasamento)…", self.act_gen_fringes),
                               ("Código de Gray…", self.act_gen_gray)]),
            ("Fase embrulhada (N passos)…", self.act_wrapped_phase),
            ("Desembrulhar fase", [
                ("Esquerda → direita", lambda: self.act_unwrap("lr")),
                ("Guiado por qualidade", lambda: self.act_unwrap("quality")),
                ("Multi-frequência (temporal)…", self.act_unwrap_multi),
                ("Com código de Gray…", self.act_unwrap_gray),
            ]),
            ("Fase → altura…", self.act_phase_height),
            ("Remover plano (inclinação)", lambda: self.img_op("sem plano", lambda a: fr.remove_plane(np.asarray(a, float)))),
            ("Profundidade → nuvem de pontos (PLY)…", self.act_depth_ply),
            None,
            ("Deflectometria (fases X e Y → superfície)…", self.act_deflecto),
            ("Varrimento vertical: profundidade por foco…", self.act_focus),
            ("Varrimento vertical: coerência (luz branca)…", self.act_coherence),
            ("Speckle", [("Mapa de contraste…", self.act_speckle_contrast),
                         ("Redução de speckle…", self.act_speckle_reduce)]),
            ("Visual hull (silhuetas em mesa rotativa)…", self.act_visual_hull),
            None,
            ("Calibração de câmara (tabuleiro)…", self.act_calibrate),
            ("Adquirir da câmara…", self.act_capture),
        ])
        self._menu(mb, "Medi&r", [
            ("Propriedades do objeto", self.act_properties, "Ctrl+P"),
            ("Relatório de todos os objetos…", self.act_report),
            ("Distância entre centros (2 objetos)", self.act_distance),
            ("Régua 2D", lambda: self.set_tool("ruler")),
            ("Transferidor 2D", lambda: self.set_tool("protractor")),
        ])
        self._menu(mb, "A&juda", [
            ("Formato dos dados de formas", lambda: TextDialog.show_text(self, "Formato", s2.DATA_HELP)),
            ("Atalhos e navegação", self.act_help),
            ("Sobre a Forja", self.act_about),
        ])

    def _build_toolbar(self):
        tb = QToolBar("Ferramentas"); tb.setObjectName("tb"); self.addToolBar(tb)
        for label, fn in [("Abrir", self.act_open), ("Ajustar", self.act_fit), ("Propriedades", self.act_properties),
                          ("Exportar", self.act_export_mesh)]:
            a = tb.addAction(tr(label)); a.triggered.connect(fn)
        tb.addSeparator()
        # ferramentas (abrem o painel à esquerda)
        for label, fn in [("Cortar", self.act_cut), ("Converter", self.act_convert), ("Fatiador", self.act_slicer),
                          ("Nesting 3D", self.act_nesting3d), ("Transformar", self.act_transform_panel),
                          ("Litofania", lambda: self.act_create("LITHO")), ("Photo magic", lambda: self.act_create("PHOTO_MAGIC")),
                          ("Oco", self.act_hollow), ("Reparar", self.act_repair)]:
            a = tb.addAction(tr(label)); a.triggered.connect(fn)
        tb.addSeparator()
        grp = QActionGroup(self)
        for k in ["select", "ln", "rc", "cir", "ell", "pol", "spl", "txt", "ruler"]:
            a = tb.addAction(tr(TOOLS[k])); a.setCheckable(True); grp.addAction(a)
            a.triggered.connect(lambda checked=False, k=k: self.set_tool(k))
            if k == "select":
                a.setChecked(True)
        tb.addSeparator()
        a = tb.addAction(tr("Nesting 2D")); a.triggered.connect(self.act_nesting)

    # ======================================================================
    # Ficheiro
    # ======================================================================
    def act_new(self):
        if QMessageBox.question(self, "Novo", "Limpar tudo (objetos, desenho e imagens)?") != QMessageBox.Yes:
            return
        self.doc.clear(); self.canvas.shapes = []; self.canvas.redraw(); self.canvas.clear_overlay()
        self.images = []; self.img_list.clear(); self.toolpaths = []

    # arrastar ficheiros (STL, OBJ, imagens, DXF…) do Finder para a janela
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.open_path(u.toLocalFile())
        e.acceptProposedAction()

    def act_open(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Abrir", self.last_dir,
                                                "Todos os suportados (*.stl *.obj *.ply *.off *.3mf *.glb *.gltf "
                                                "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dxf *.svg *.forja *.asc *.csv *.dcm *.avi *.mp4 *.mov "
                                                "*.s8 *.u8 *.s16 *.u16 *.s32 *.u32 *.f32 *.dbl *.raw);;" + MESH_FILTER)
        for p in paths:
            self.open_path(p)

    def open_path(self, p):
        ext = p.lower().rsplit(".", 1)[-1]
        self._dir(p)
        try:
            if ext in ("stl", "obj", "ply", "off", "3mf", "glb", "gltf"):
                self.load_mesh(p)
            elif ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
                self.add_image(os.path.basename(p), iop.load(p))
            elif ext in ("dxf", "svg", "txt"):
                self.load_2d(p)
            elif ext == "forja":
                self.load_project(p)
            elif ext in ("s8", "u8", "s16", "u16", "s32", "u32", "f32", "flt", "dbl", "raw"):
                self._import_raw_path(p)
            elif ext in ("asc", "csv", "dat"):
                from ..core import imageops2 as io2
                self.add_image(os.path.basename(p), io2.load_ascii(p))
            elif ext in ("dcm", "dicom"):
                from ..core import imageops2 as io2
                self.add_image(os.path.basename(p), io2.load_dicom(p))
            elif ext in ("avi", "mp4", "mov", "mkv"):
                from ..core import imageops2 as io2
                fr_ = io2.load_video(p)
                for i, a in enumerate(fr_):
                    self.add_image(f"{os.path.basename(p)} {i + 1:04d}", a, show=i == len(fr_) - 1)
            else:
                self.error(f"Formato não suportado: {ext}")
        except Exception as e:
            self.error(f"Falha ao abrir {os.path.basename(p)}: {e}")

    def load_mesh(self, p):
        def work():
            try:
                m = trimesh.load(p, force="mesh")
            except Exception:
                raise ValueError(f"{os.path.basename(p)} não é uma malha 3D num formato suportado (STL, OBJ, PLY, OFF, 3MF, GLB).")
            if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
                raise ValueError("Ficheiro sem malha triangular.")
            return m

        def done(m):
            self.add_mesh(m, os.path.splitext(os.path.basename(p))[0])
            self.log(f"Aberto {p}: {len(m.faces)} faces, {'fechada' if m.is_watertight else 'NÃO fechada'}")
            self.tabs.setCurrentWidget(self.view3d); self.view3d.fit()
        self.task(f"A abrir {os.path.basename(p)}…", work, done)

    def load_2d(self, p):
        ext = p.lower().rsplit(".", 1)[-1]
        try:
            shapes = s2.import_dxf(p) if ext == "dxf" else s2.import_svg(p) if ext == "svg" else s2.from_data(open(p, encoding="utf-8").read())
        except UnicodeDecodeError:
            return self.error(f"{os.path.basename(p)} não é um ficheiro de texto/DXF/SVG válido.")
        self.canvas.add_shapes(shapes)
        self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()
        self.log(f"Importadas {len(shapes)} formas de {os.path.basename(p)}")

    def act_import_mesh(self):
        for p in QFileDialog.getOpenFileNames(self, "Importar malha", self.last_dir, MESH_FILTER)[0]:
            self._dir(p); self.load_mesh(p)

    def act_import_images(self):
        for p in QFileDialog.getOpenFileNames(self, "Importar imagens", self.last_dir, IMG_FILTER)[0]:
            self.open_path(p)

    def act_import_2d(self):
        for p in QFileDialog.getOpenFileNames(self, "Importar desenho", self.last_dir, D2_FILTER)[0]:
            self._dir(p); self.load_2d(p)

    def act_import_raw(self):
        p, _ = QFileDialog.getOpenFileName(self, "Imagem bruta", self.last_dir, IMG_FILTER)
        if p:
            self._import_raw_path(p)

    def _import_raw_path(self, p):
        from ..core import imageops2 as io2
        ext = p.lower().rsplit(".", 1)[-1]
        dt = {"s8": "int8", "u8": "uint8", "s16": "int16", "u16": "uint16", "s32": "int32", "u32": "uint32",
              "f32": "float32", "flt": "float32", "dbl": "float64"}.get(ext, "uint8")
        v = self.ask("Dados brutos", [("w", "Largura (px)", 640, (1, 100000)), ("h", "Altura (px)", 480, (1, 100000)),
                                      ("d", "Profundidade (fatias, 3D)", 1, (1, 100000)),
                                      ("dt", "Tipo", dt, list(io2.RAW_TYPES)), ("off", "Desvio inicial (bytes)", 0, (0, 2 ** 31 - 1)),
                                      ("le", "Ordem de bytes", "little-endian (Intel/C)", ["little-endian (Intel/C)", "big-endian"])],
                     note=f"Tamanho do ficheiro: {os.path.getsize(p)} bytes")
        if not v:
            return
        try:
            a = io2.load_raw(p, v["w"], v["h"], v["dt"], v["off"], v["le"].startswith("little"), v["d"])
        except Exception as e:
            return self.error(f"Dimensões não coincidem com o ficheiro: {e}")
        if a.ndim == 3:
            if hasattr(self, "add_volume"):
                self.add_volume(os.path.basename(p), a)
            else:
                for k in range(a.shape[0]):
                    self.add_image(f"{os.path.basename(p)} z{k}", a[k], show=k == a.shape[0] - 1)
        else:
            self.add_image(os.path.basename(p), a)

    def act_export_mesh(self):
        objs = self.sel_objs()
        if not objs:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Exportar malha", os.path.join(self.last_dir, objs[0].name + ".stl"),
                                           MESH_FILTER)
        if not p:
            return
        self._dir(p)
        m = trimesh.util.concatenate([o.mesh for o in objs]) if len(objs) > 1 else objs[0].mesh
        m.export(p)
        self.log(f"Exportado {p} ({len(m.faces)} faces)")

    def act_export_each(self):
        objs = self.sel_objs()
        if not objs:
            return
        d = QFileDialog.getExistingDirectory(self, "Pasta de destino", self.last_dir)
        if not d:
            return
        v = self.ask("Formato", [("fmt", "Formato", "stl", ["stl", "obj", "ply", "3mf", "glb"])])
        if not v:
            return
        for o in objs:
            safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in o.name)
            o.mesh.export(os.path.join(d, f"{safe}.{v['fmt']}"))
        self.log(f"Exportados {len(objs)} ficheiros para {d}")

    def act_export_2d(self):
        shapes = self.canvas.selected_shapes() or self.canvas.shapes
        if not shapes:
            return self.error("Não há formas 2D.")
        p, _ = QFileDialog.getSaveFileName(self, "Exportar desenho", os.path.join(self.last_dir, "desenho.dxf"),
                                           "DXF (*.dxf);;SVG (*.svg)")
        if not p:
            return
        self._dir(p)
        paths = [q for q in self.toolpaths]
        (s2.export_svg if p.lower().endswith(".svg") else s2.export_dxf)(shapes, p, extra_paths=paths)
        self.log(f"Exportado {p} ({len(shapes)} formas)")

    def act_export_data(self):
        shapes = self.canvas.selected_shapes() or self.canvas.shapes
        p, _ = QFileDialog.getSaveFileName(self, "Dados das formas", os.path.join(self.last_dir, "formas.txt"), "Texto (*.txt)")
        if p:
            open(p, "w").write(s2.to_data(shapes)); self._dir(p)

    def act_export_image(self):
        it = self.cur_image()
        if not it:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Exportar imagem", os.path.join(self.last_dir, it.name.split(".")[0] + ".png"),
                                           "PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg);;Bruto float32 (*.f32);;Bruto int8 (*.s8)")
        if not p:
            return
        self._dir(p)
        d = it.data
        if p.endswith(".f32"):
            d = np.asarray(d, np.float32)
        elif p.endswith(".s8"):
            d = np.asarray(d).astype(np.int8)
        elif p.endswith(".png") and d.dtype != np.uint8:
            d, _ = iop.colorize(d, self.imgview.cmap)
        iop.save(p, d)
        self.log(f"Imagem guardada: {p}")

    def act_snapshot(self):
        p, _ = QFileDialog.getSaveFileName(self, "Captura 3D", os.path.join(self.last_dir, "vista.png"), "PNG (*.png)")
        if p:
            self.view3d.snapshot(p); self.log(f"Captura guardada: {p}")

    # --- projeto --------------------------------------------------------
    def act_save_project(self, password=None):
        p, _ = QFileDialog.getSaveFileName(self, "Guardar projeto", os.path.join(self.last_dir, "projeto.forja"), PROJ_FILTER)
        if not p:
            return
        self._dir(p)
        if password is None and getattr(self, "project_password", None):
            password = self.project_password
        if password:
            from ..core import projcrypt
            buf = io.BytesIO(); self._write_project(buf)
            open(p, "wb").write(projcrypt.encrypt(buf.getvalue(), password))
            self.log(f"Projeto guardado (protegido com palavra-passe): {p}")
            return
        self._write_project(p)
        self.log(f"Projeto guardado: {p}")

    def act_set_password(self):
        """Palavra-passe do projeto: os projetos passam a ser gravados cifrados."""
        v = self.ask("Palavra-passe do projeto", [("p", "Palavra-passe (vazio = sem palavra-passe)", ""),
                                                  ("q", "Repetir a palavra-passe", "")],
                     note="O projeto é cifrado (AES) ao gravar e só abre com esta palavra-passe. Se a esquecer, não é possível recuperá-lo.")
        if v is None:
            return
        if v["p"] != v["q"]:
            return self.error("As palavras-passe não coincidem.")
        self.project_password = v["p"] or None
        self.log("Palavra-passe do projeto " + ("definida." if v["p"] else "removida."))

    def _write_project(self, p):
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            meta = {"objects": [], "images": []}
            for i, o in enumerate(self.doc.objects):
                fn = f"mesh_{i}.ply"
                z.writestr(fn, o.mesh.export(file_type="ply"))
                entry = {"file": fn, "name": o.name, "color": o.color, "visible": o.visible, "alpha": o.alpha,
                         "style": getattr(o, "style", None), "id": o.id}
                t = getattr(o, "tool", None)
                if t:
                    tt = {k: v for k, v in t.items() if k != "source_mesh"}
                    if t.get("source_mesh") is not None:
                        sfn = f"tool_src_{t['group']}.ply"
                        if sfn not in z.namelist():
                            z.writestr(sfn, t["source_mesh"].export(file_type="ply"))
                        tt["source_file"] = sfn
                    entry["tool"] = tt
                if getattr(o, "gen", None):
                    g = o.gen; src = g.get("src") or {}
                    entry["gen"] = {"id": g["id"], "values": g["values"], "group": g["group"], "part": g.get("part", 0),
                                    "geom": src.get("geom"), "image_name": src.get("image_name")}
                    if src.get("image") is not None:
                        ib = io.BytesIO(); np.save(ib, np.asarray(src["image"])); z.writestr(f"gen_img_{i}.npy", ib.getvalue())
                        entry["gen"]["image_file"] = f"gen_img_{i}.npy"
                meta["objects"].append(entry)
            for i, im in enumerate(self.images):
                fn = f"img_{i}.npy"
                buf = io.BytesIO(); np.save(buf, np.asarray(im.data)); z.writestr(fn, buf.getvalue())
                meta["images"].append({"file": fn, "name": im.name, "volume": bool(im.meta.get("volume"))})
            z.writestr("shapes.txt", s2.to_data(self.canvas.shapes))
            z.writestr("shapes.json", s2.to_json(self.canvas.shapes))
            z.writestr("project.json", json.dumps(meta, ensure_ascii=False, indent=1))

    def act_open_project(self):
        p, _ = QFileDialog.getOpenFileName(self, "Abrir projeto", self.last_dir, PROJ_FILTER)
        if p:
            self._dir(p); self.load_project(p)

    def load_project(self, p):
        from ..core import projcrypt
        name = os.path.basename(p)
        if projcrypt.is_encrypted(p):
            v = self.ask("Projeto protegido", [("p", "Palavra-passe", "")], note=f"{name} está protegido com palavra-passe.")
            if not v:
                return
            try:
                p = io.BytesIO(projcrypt.decrypt(open(p, "rb").read(), v["p"]))
            except Exception:
                return self.error("Palavra-passe errada (ou ficheiro danificado).")
            self.project_password = v["p"]
        if not zipfile.is_zipfile(p):
            return self.error(f"{name} não é um projeto Forja válido.")
        _srcs, _idmap = {}, {}
        with zipfile.ZipFile(p) as z:
            meta = json.loads(z.read("project.json"))
            for o in meta["objects"]:
                m = trimesh.load(io.BytesIO(z.read(o["file"])), file_type="ply", force="mesh")
                obj = self.doc.add(m, o["name"], tuple(o["color"]))
                obj.visible = o.get("visible", True)
                obj.alpha = o.get("alpha", 1.0)
                if o.get("style"):
                    obj.style = o["style"]
                if o.get("tool"):
                    t = dict(o["tool"])
                    if t.get("source_file"):
                        key = t["source_file"]
                        if key not in _srcs:
                            _srcs[key] = trimesh.load(io.BytesIO(z.read(key)), file_type="ply", force="mesh")
                        t["source_mesh"] = _srcs[key]
                    obj.tool = t
                _idmap[o.get("id")] = obj.id
                g = o.get("gen")
                if g:
                    src = {k: g[k] for k in ("geom", "image_name") if g.get(k)}
                    if g.get("image_file"):
                        src["image"] = np.load(io.BytesIO(z.read(g["image_file"])))
                    obj.gen = {"id": g["id"], "values": g["values"], "src": src, "group": g["group"], "part": g.get("part", 0)}
            for im in meta["images"]:
                self.add_image(im["name"], np.load(io.BytesIO(z.read(im["file"]))), {"volume": True} if im.get("volume") else None,
                               show=False)
            names = z.namelist()
            shapes = (s2.from_json(z.read("shapes.json").decode()) if "shapes.json" in names
                      else s2.from_data(z.read("shapes.txt").decode()))
            self.canvas.add_shapes(shapes, select=False)
        for _o in self.doc.objects:  # os ids mudam ao abrir: atualiza a ligação ao objeto original
            _t = getattr(_o, "tool", None)
            if _t and _t.get("source_id") in _idmap:
                _t["source_id"] = _idmap[_t["source_id"]]
        from .cutpanel import _GROUPS  # noqa: F401  (os grupos novos continuam a numeração)
        self.doc.changed(); self.view3d.fit()
        self.log(f"Projeto aberto: {p}")

    # --- exemplos -------------------------------------------------------
    def demo_model(self):
        body = mo.primitive("box", x=60, y=40, z=30)
        head = mo.primitive("sphere", r=22); head.apply_translation((0, 0, 22))
        hole = mo.primitive("cylinder", r=8, h=80); hole.apply_translation((0, 0, -10))
        m = mo.boolean(mo.boolean(body, head, "union"), hole, "difference")
        self.add_mesh(mo.center_on_plate(m), "exemplo")
        self.tabs.setCurrentWidget(self.view3d); self.view3d.fit()

    def demo_fringe(self):
        Z = fr.demo_object()
        for i, im in enumerate(fr.simulate_capture(Z)):
            self.add_image(f"objeto_franja_{i + 1}", im, show=False)
        for i, im in enumerate(fr.simulate_capture(np.zeros_like(Z))):
            self.add_image(f"referência_franja_{i + 1}", im, show=False)
        self.add_image("altura_real_mm", Z)
        self.log("Exemplo: selecione as 4 imagens 'objeto_franja' → Metrologia ▸ Fase embrulhada; repita para a "
                 "referência; desembrulhe ambas; depois Fase → altura (fator 2,857 mm/rad).")

    def demo_focus(self):
        Z = fr.demo_object(200, 150)
        zs = np.linspace(-2, 28, 16)
        for z, im in zip(zs, fr.simulate_focus_stack(Z, zs)):
            self.add_image(f"foco_z={z:.1f}", im, show=False)
        self.add_image("altura_real_mm", Z)
        self.log("Exemplo: selecione as 16 imagens 'foco_' → Metrologia ▸ Profundidade por foco (z de -2 a 28).")

    def demo_silhouettes(self):
        sil, angs = iop.demo_silhouettes()
        for s, a in zip(sil, angs):
            self.add_image(f"silhueta_{a:.0f}°", s, show=False)
        self.img_list.setCurrentRow(len(self.images) - 1)
        self.tabs.setCurrentWidget(self.imgview)
        self.log("Exemplo: selecione as 12 silhuetas → Metrologia ▸ Visual hull (0 a 330°, passo 30°).")

    def demo_shapes(self):
        shapes = s2.from_data(("rc 0 0 50 30\ncir 0 0 20\npol 0 0 60 0 0 40\nrcr 0 0 70 20 30\n"
                               "ell 0 0 30 10\npol 0 0 40 0 40 40 30 40 30 10 0 10\n") * 3)
        for i, s in enumerate(shapes):
            shapes[i] = s2.move(s, (i % 6) * 80, (i // 6) * 60 + 250)
        self.canvas.add_shapes(shapes)
        self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()

    # ======================================================================
    # Editar / Ver
    # ======================================================================
    def in_2d(self):
        return self.tabs.currentWidget() is self.canvas

    def act_undo(self):
        if self.in_2d():
            self.canvas.undo()
        elif not self.doc.undo():
            self.statusBar().showMessage("Nada para desfazer.")

    def act_duplicate(self):
        if self.in_2d():
            sel = self.canvas.selected_shapes()
            self.canvas.add_shapes([s2.move(s, 10, -10) for s in sel])
            return
        objs = self.sel_objs()
        if objs:
            new = []
            for o in objs:
                m = o.mesh.copy(); m.apply_translation((o.mesh.extents[0] + 5, 0, 0))
                new.append(self.doc.add(m, o.name + " cópia"))
            self._select_objs(new)

    def act_delete(self):
        if self.in_2d():
            return self.canvas.delete_selected()
        objs = self.sel_objs()
        if objs:
            self.doc.snapshot(); self.doc.remove(objs)

    def act_select_all(self):
        if self.in_2d():
            return self.canvas.select_all()
        self.obj_list.selectAll()

    def act_color(self):
        objs = self.sel_objs()
        if objs:
            c = QColorDialog.getColor(QColor.fromRgbF(*objs[0].color), self)
            if c.isValid():
                for o in objs:
                    o.color = (c.redF(), c.greenF(), c.blueF())
                self.view3d.refresh()

    def act_rename(self):
        objs = self.sel_objs(1, True)
        if objs:
            t, ok = QInputDialog.getText(self, "Renomear", "Nome:", text=objs[0].name)
            if ok and t:
                objs[0].name = t; self.doc.changed()

    def act_fit(self):
        (self.canvas.fit if self.in_2d() else self.imgview.fit if self.tabs.currentWidget() is self.imgview
         else self.view3d.fit)()

    def set_mode(self, m):
        self.view3d.mode = m; self.view3d.refresh()

    def act_transparency(self):
        objs = self.sel_objs()
        if objs:
            v = self.ask("Transparência", [("a", "Opacidade (0–1)", float(objs[0].alpha), (0.05, 1.0))])
            if v:
                for o in objs:
                    o.alpha = v["a"]
                self.view3d.refresh()

    def act_background(self):
        c = QColorDialog.getColor(self.view3d.bg, self)
        if c.isValid():
            self.view3d.set_background(c)

    def act_overhang(self):
        self.view3d.show_overhang = not self.view3d.show_overhang
        self.view3d.refresh()
        self.statusBar().showMessage("Saliências a vermelho (faces > 45° para baixo)" if self.view3d.show_overhang else "")

    def act_plate(self):
        v = self.ask("Mesa", [("s", "Tamanho da grelha (mm)", 220.0, (10, 5000))])
        if v:
            self.view3d.set_grid(v["s"])

    def act_cmap(self):
        v = self.ask("Mapa de cores", [("c", "Mapa", self.imgview.cmap,
                                        ["viridis", "jet", "turbo", "gray", "plasma", "inferno", "magma", "coolwarm", "hsv", "cividis"])])
        if v:
            self.imgview.cmap = v["c"]; self._img_current(self.img_list.currentRow())

    # ======================================================================
    # Criar
    # ======================================================================
    def act_primitive(self, kind):
        F = {"box": [("x", "X (mm)", 20.0), ("y", "Y (mm)", 20.0), ("z", "Z (mm)", 20.0)],
             "sphere": [("r", "Raio (mm)", 10.0)],
             "cylinder": [("r", "Raio (mm)", 10.0), ("h", "Altura (mm)", 20.0), ("seg", "Segmentos", 64, (3, 512))],
             "cone": [("r", "Raio (mm)", 10.0), ("h", "Altura (mm)", 20.0), ("seg", "Segmentos", 64, (3, 512))],
             "torus": [("R", "Raio maior (mm)", 20.0), ("r", "Raio menor (mm)", 5.0)],
             "capsule": [("r", "Raio (mm)", 5.0), ("h", "Altura (mm)", 20.0)]}
        names = {"box": "caixa", "sphere": "esfera", "cylinder": "cilindro", "cone": "cone", "torus": "toro", "capsule": "cápsula"}
        v = self.ask(names[kind].capitalize(), F[kind])
        if v:
            self.add_mesh(mo.primitive(kind, **v), names[kind]); self.view3d.fit()

    def act_text3d(self):
        from ..core import text3d
        v = self.ask("Texto 3D", [("t", "Texto", "Forja"), ("size", "Altura das letras (mm)", 20.0),
                                  ("depth", "Espessura (mm)", 5.0), ("font", "Tipo de letra", "Helvetica"),
                                  ("bold", "Negrito", True), ("base", "Base por baixo (mm, 0 = sem)", 0.0)])
        if not v:
            return
        try:
            m = text3d.text_mesh(v["t"], v["size"], v["depth"], v["font"], v["bold"])
            if v["base"] > 0:
                ext = m.extents
                b = mo._box_mesh([-3, -3, -v["base"]], [ext[0] + 3, ext[1] + 3, 0.01])
                m = mo.boolean(b, m, "union") if m.is_watertight else trimesh.util.concatenate([b, m])
            self.add_mesh(mo.center_on_plate(m), f"texto '{v['t']}'"); self.view3d.fit()
        except Exception as e:
            self.error(e)

    def _image_for_3d(self):
        if self.images and self.img_list.currentRow() >= 0:
            return self.images[self.img_list.currentRow()]
        p, _ = QFileDialog.getOpenFileName(self, "Imagem", self.last_dir, IMG_FILTER)
        if not p:
            return None
        return self.add_image(os.path.basename(p), iop.load(p), show=False)

    def act_relief(self):
        it = self._image_for_3d()
        if not it:
            return
        v = self.ask("Relevo", [("w", "Largura (mm)", 100.0), ("h", "Altura do relevo (mm)", 5.0),
                                ("base", "Espessura da base (mm)", 2.0), ("inv", "Claro = alto", True),
                                ("blur", "Suavização (px)", 1.0), ("res", "Resolução máx. (px)", 300, (20, 2000))])
        if not v:
            return
        g = iop.to_gray(it.data) if it.data.dtype == np.uint8 else iop.to_u8(it.data)

        def work():
            H = mo.image_to_heightmap(g, v["h"], invert=v["inv"], blur=v["blur"], max_px=v["res"])
            r, c = H.shape
            return mo.heightmap_to_solid(H, v["w"], v["w"] * r / c, base=v["base"])
        self.task("A criar relevo…", work, lambda m: (self.add_mesh(m, f"relevo {it.name}"),
                                                      self.tabs.setCurrentWidget(self.view3d), self.view3d.fit()))

    def act_lithophane(self):
        it = self._image_for_3d()
        if not it:
            return
        v = self.ask("Litofania", [("w", "Largura (mm)", 100.0), ("tmax", "Espessura máx. (mm)", 3.0),
                                   ("tmin", "Espessura mín. (mm)", 0.6), ("res", "Resolução máx. (px)", 400, (20, 2000))])
        if not v:
            return
        g = iop.to_gray(it.data) if it.data.dtype == np.uint8 else iop.to_u8(it.data)

        def work():
            H = mo.image_to_heightmap(g, v["tmax"] - v["tmin"], max_px=v["res"])
            r, c = H.shape
            m = mo.heightmap_to_solid(H, v["w"], v["w"] * r / c, base=v["tmin"])
            return mo.rotate(m, 90, 0, 0)
        self.task("A criar litofania…", work, lambda m: (self.add_mesh(mo.drop_to_floor(m), f"litofania {it.name}"),
                                                         self.tabs.setCurrentWidget(self.view3d), self.view3d.fit()))

    def act_depth_to_3d(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Mapa de profundidade → 3D", [("px", "Tamanho do pixel (mm)", 0.2), ("zs", "Escala de Z", 1.0),
                                                   ("solid", "Sólido fechado (para imprimir)", False),
                                                   ("base", "Base (mm, se sólido)", 2.0), ("step", "Saltar pixels (1 = todos)", 1, (1, 50))])
        if not v:
            return
        Z = np.asarray(it.data, float)
        if Z.ndim == 3:
            Z = Z.mean(2)
        Z = Z[::v["step"], ::v["step"]] * v["zs"]
        px = v["px"] * v["step"]

        def work():
            r, c = Z.shape
            if v["solid"]:
                Zf = np.nan_to_num(Z - np.nanmin(Z), nan=0)
                return mo.heightmap_to_solid(Zf, c * px, r * px, base=v["base"])
            return mo.heightmap_to_surface(Z, c * px, r * px)
        self.task("A criar superfície…", work, lambda m: (self.add_mesh(m, f"superfície {it.name}"),
                                                          self.tabs.setCurrentWidget(self.view3d), self.view3d.fit()))

    def act_extrude2d(self):
        from ..core.text3d import extrude
        sel = [s for s in (self.canvas.selected_shapes() or self.canvas.shapes) if s.closed]
        if not sel:
            return self.error("Selecione formas 2D fechadas no separador 2D.")
        v = self.ask("Extrudir", [("h", "Altura (mm)", 5.0), ("merge", "Unir num só objeto", True)])
        if not v:
            return
        from shapely.ops import unary_union
        if v["merge"]:
            self.add_mesh(extrude(unary_union([s.geom for s in sel]), v["h"]), "extrusão")
        else:
            for s in sel:
                self.add_mesh(extrude(s.geom, v["h"]), f"extrusão {s.name}", select=False)
        self.tabs.setCurrentWidget(self.view3d); self.view3d.fit()

    # ======================================================================
    # Transformar
    # ======================================================================
    def _apply(self, fn, label=None):
        objs = self.sel_objs()
        if not objs:
            return
        self.doc.snapshot()
        try:
            for o in objs:
                o.mesh = fn(o.mesh.copy())
        except Exception as e:
            self.doc.undo(); return self.error(e)
        self.doc.changed()
        if label:
            self.log(f"{label}: {', '.join(o.name for o in objs)}")

    def act_move(self):
        v = self.ask("Mover", [("x", "ΔX (mm)", 0.0), ("y", "ΔY (mm)", 0.0), ("z", "ΔZ (mm)", 0.0)])
        if v:
            self._apply(lambda m: mo.translate(m, v["x"], v["y"], v["z"]), "Movido")

    def act_rotate(self):
        v = self.ask("Rodar", [("x", "Em torno de X (°)", 0.0), ("y", "Em torno de Y (°)", 0.0), ("z", "Em torno de Z (°)", 90.0)])
        if v:
            self._apply(lambda m: mo.rotate(m, v["x"], v["y"], v["z"]), "Rodado")

    def act_scale(self):
        v = self.ask("Escalar", [("u", "Uniforme (%)", 100.0), ("x", "X (%) se não uniforme", 100.0),
                                 ("y", "Y (%)", 100.0), ("z", "Z (%)", 100.0), ("uni", "Uniforme", True)])
        if v:
            k = [v["u"] / 100] * 3 if v["uni"] else [v["x"] / 100, v["y"] / 100, v["z"] / 100]
            self._apply(lambda m: mo.scale(m, *k), "Escalado")

    def act_scale_to(self):
        v = self.ask("Escalar para tamanho", [("s", "Tamanho (mm)", 100.0), ("ax", "Eixo", "maior", ["maior", "X", "Y", "Z"])])
        if v:
            ax = {"maior": None, "X": 0, "Y": 1, "Z": 2}[v["ax"]]
            self._apply(lambda m: mo.scale_to_size(m, v["s"], ax), "Escalado")

    def act_mirror(self):
        v = self.ask("Espelhar", [("ax", "Eixo", "X", ["X", "Y", "Z"])])
        if v:
            self._apply(lambda m: mo.mirror(m, "XYZ".index(v["ax"])), "Espelhado")

    def act_center(self): self._apply(mo.center_on_plate, "Centrado")
    def act_floor(self): self._apply(mo.drop_to_floor, "Pousado")
    def act_layflat(self): self._apply(mo.lay_flat, "Assentado")

    def act_arrange(self):
        objs = [o for o in self.doc.objects if o.visible]
        if not objs:
            return
        v = self.ask("Organizar", [("w", "Largura da mesa (mm)", 220.0), ("h", "Profundidade da mesa (mm)", 220.0),
                                   ("gap", "Espaço entre peças (mm)", 5.0)])
        if not v:
            return
        from shapely.geometry import MultiPoint
        items = []
        for o in objs:
            hull = MultiPoint(o.mesh.vertices[:, :2]).convex_hull
            items.append((o.id, hull))
        plates, un = nesting.nest(items, v["w"], v["h"], v["gap"], 0, rotations=(0,))
        self.doc.snapshot()
        idmap = {o.id: o for o in objs}
        for pi, pl in enumerate(plates):
            for iid, g, a in pl["placements"]:
                o = idmap[iid]
                b = o.mesh.bounds
                hb = MultiPoint(o.mesh.vertices[:, :2]).convex_hull.bounds
                dx = g.bounds[0] - hb[0] - v["w"] / 2 + pi * (v["w"] + 20)
                dy = g.bounds[1] - hb[1] - v["h"] / 2
                o.mesh.apply_translation((dx, dy, -b[0][2]))
        self.doc.changed(); self.view3d.fit()
        self.log(nesting.report(plates, un, v["w"], v["h"]))

    # ======================================================================
    # Malha
    # ======================================================================
    def _replace_with(self, o, meshes, suffix):
        self.doc.objects.remove(o)
        new = [self.doc.add(m, f"{o.name} {suffix}{i + 1}" if len(meshes) > 1 else o.name) for i, m in enumerate(meshes)]
        self._select_objs(new)
        return new

    def act_cut_simple(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        o = objs[0]
        c = o.mesh.bounds.mean(axis=0)
        v = self.ask("Cortar por plano", [("ax", "Normal do plano", "Z", ["X", "Y", "Z", "personalizada"]),
                                          ("pos", "Posição ao longo do eixo (mm)", float(round(c[2], 2))),
                                          ("nx", "Normal X (personalizada)", 0.0), ("ny", "Normal Y", 0.0), ("nz", "Normal Z", 1.0),
                                          ("gap", "Afastar as partes (mm)", 5.0), ("keep", "Manter", "ambas", ["ambas", "acima", "abaixo"])],
                     note=f"Caixa do objeto: {np.round(o.mesh.bounds[0], 2)} a {np.round(o.mesh.bounds[1], 2)}")
        if not v:
            return
        if v["ax"] == "personalizada":
            n = np.array([v["nx"], v["ny"], v["nz"]], float); origin = c + n / np.linalg.norm(n) * 0
            origin = c.copy(); origin[2] = v["pos"] if abs(n[2]) > 0.5 else c[2]
        else:
            n = np.zeros(3); n["XYZ".index(v["ax"])] = 1
            origin = c.copy(); origin["XYZ".index(v["ax"])] = v["pos"]
        mesh = o.mesh.copy()

        def work():
            return mo.cut_plane(mesh, origin, n)

        def done(r):
            pos, neg = r
            self.doc.snapshot()
            nn = n / np.linalg.norm(n)
            pos.apply_translation(nn * v["gap"] / 2); neg.apply_translation(-nn * v["gap"] / 2)
            keep = {"ambas": [neg, pos], "acima": [pos], "abaixo": [neg]}[v["keep"]]
            keep = [k for k in keep if len(k.faces)]
            self._replace_with(o, keep, "parte ")
            self.log(f"Corte: {len(keep)} parte(s)")
        self.task("A cortar…", work, done)

    def act_split_grid(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        o = objs[0]
        e = o.mesh.extents
        v = self.ask("Partição em grelha", [
            ("nx", "Partes em X", 2, (1, 20)), ("ny", "Partes em Y", 1, (1, 20)), ("nz", "Partes em Z", 1, (1, 20)),
            ("con", "Pinos de encaixe", True), ("r", "Raio do pino (mm)", 3.0), ("l", "Comprimento do pino (mm)", 10.0),
            ("cl", "Folga (mm)", 0.2), ("n", "Pinos por face", 1, (1, 8)), ("gap", "Afastar peças (mm)", 8.0)],
            note=f"Tamanho: {e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm. Use para dividir modelos maiores que a impressora.")
        if not v:
            return
        mesh = o.mesh.copy()

        def work():
            return mo.split_grid(mesh, v["nx"], v["ny"], v["nz"], v["con"], v["r"], v["l"], v["cl"], v["n"])

        def done(parts):
            self.doc.snapshot()
            meshes = []
            for (i, j, k), m in parts:
                m.apply_translation((i * v["gap"], j * v["gap"], k * v["gap"]))
                meshes.append(m)
            self._replace_with(o, meshes, "peça ")
            self.log(f"Partição: {len(meshes)} peças")
        self.task("A dividir o modelo…", work, done)

    def act_split_planes(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        o = objs[0]
        b = o.mesh.bounds
        v = self.ask("Partição por planos", [("ax", "Eixo", "Z", ["X", "Y", "Z"]),
                                             ("pos", "Posições (mm, separadas por vírgulas)", f"{(b[0][2] + b[1][2]) / 2:.1f}"),
                                             ("gap", "Afastar (mm)", 5.0)])
        if not v:
            return
        ax = "XYZ".index(v["ax"])
        try:
            pos = [float(x) for x in v["pos"].replace(";", ",").split(",") if x.strip()]
        except ValueError:
            return self.error("Posições inválidas.")
        mesh = o.mesh.copy()

        def done(pieces):
            self.doc.snapshot()
            for i, m in enumerate(pieces):
                d = np.zeros(3); d[ax] = i * v["gap"]; m.apply_translation(d)
            self._replace_with(o, pieces, "parte ")
        self.task("A dividir…", lambda: mo.split_by_planes(mesh, ax, pos), done)

    def act_hollow(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        o = objs[0]
        v = self.ask("Oco", [("t", "Distância entre a superfície original e a interior (mm)", 10.0), ("res", "Resolução (voxels no maior lado)", 150, (30, 500)),
                             ("drain", "Raio do furo de escoamento (mm, 0 = sem)", 0.0)],
                     note="Poupa material na impressão. Resoluções altas são mais lentas.")
        if not v:
            return
        mesh = o.mesh.copy()

        def done(m):
            self.doc.snapshot(); o.mesh = m; self.doc.changed()
            self.log(f"Oco: volume {abs(mesh.volume) / 1000:.2f} → {abs(m.volume) / 1000:.2f} cm³")
        self.task("A esvaziar…", lambda: mo.hollow(mesh, v["t"], v["res"], v["drain"]), done)

    def act_inflate(self):
        v = self.ask("Inflar / encolher", [("d", "Distância (mm, negativo encolhe)", 10.0)],
                     note="Exemplo 1: 3.2 (inflar 3,2 mm). Exemplo 2: −5.7 (encolher 5,7 mm).")
        if v:
            self._apply(lambda m: mo.shell_offset(m, v["d"]), "Inflado")

    def act_repair(self):
        objs = self.sel_objs()
        if not objs:
            return
        self.doc.snapshot()
        for o in objs:
            o.mesh, rep = mo.repair(o.mesh.copy())
            self.log(f"Reparar {o.name}: " + "; ".join(rep))
        self.doc.changed()

    def act_simplify(self):
        objs = self.sel_objs()
        if not objs:
            return
        v = self.ask("Simplificar", [("r", "Fração de triângulos a reduzir", 0.3, (0.0, 0.999))],
                     note=f"Exemplo: 0.3 (reduz 30% dos triângulos). Faces atuais: {', '.join(str(len(o.mesh.faces)) for o in objs)}")
        if v:
            self._apply(lambda m: mo.simplify(m, len(m.faces) * (1 - v["r"])), "Simplificado")

    def act_densify(self):
        v = self.ask("Densificar", [("e", "Comprimento da aresta (mm)", 10.0, (0.01, 1000))])
        if v:
            self._apply(lambda m: mo.densify(m, v["e"]), "Densificado")

    def act_smooth(self):
        v = self.ask("Suavizar", [("n", "Iterações", 10, (1, 500))])
        if v:
            self._apply(lambda m: mo.smooth(m, v["n"]), "Suavizado")

    def act_reverse(self):
        self._apply(lambda m: (m.invert(), m)[1], "Normais invertidas")

    def act_boolean(self, op):
        objs = self.sel_objs(2)
        if not objs:
            return
        meshes = [o.mesh.copy() for o in objs]

        def work():
            if op == "union":
                return mo.union_all(meshes)
            acc = meshes[0]
            for m in meshes[1:]:
                acc = mo.boolean(acc, m, op)
            return acc

        def done(m):
            self.doc.snapshot()
            name = objs[0].name
            self.doc.remove(objs)
            self.add_mesh(m, {"union": "união", "difference": "diferença", "intersection": "interseção"}[op] + f" {name}", place=False)
        self.task("Operação booleana…", work, done)

    def act_concat(self):
        objs = self.sel_objs(2)
        if objs:
            self.doc.snapshot()
            m = trimesh.util.concatenate([o.mesh for o in objs])
            self.doc.remove(objs); self.add_mesh(m, "combinado", place=False)

    def act_separate(self):
        objs = self.sel_objs(1, True)
        if objs:
            parts = mo.separate(objs[0].mesh)
            if len(parts) < 2:
                return self.statusBar().showMessage("O objeto tem só um componente.")
            self.doc.snapshot(); self._replace_with(objs[0], parts, "comp ")
            self.log(f"Separado em {len(parts)} componentes")

    def act_imprint(self):
        from ..core import text3d
        objs = self.sel_objs(1, True)
        if not objs:
            return
        o = objs[0]
        b = o.mesh.bounds
        v = self.ask("Texto em relevo / gravado", [("t", "Texto", "Forja"), ("size", "Altura das letras (mm)", 8.0),
                                                   ("d", "Profundidade (mm)", 1.0), ("mode", "Modo", "gravar", ["gravar", "relevo"]),
                                                   ("face", "Face", "maior face plana", ["maior face plana", "topo", "frente (−Y)", "fundo"]),
                                                   ("dx", "Deslocar X (mm)", 0.0), ("dy", "Deslocar Y/Z (mm)", 0.0)])
        if not v:
            return
        mesh = o.mesh.copy()

        def work():
            t = text3d.text_mesh(v["t"], v["size"], v["d"] * 2 if v["mode"] == "gravar" else v["d"] + 0.5)
            ext = t.extents
            t.apply_translation((-ext[0] / 2, -ext[1] / 2, 0))
            c = b.mean(axis=0)
            if v["face"] == "maior face plana":
                # o texto vai para a maior face plana do modelo
                fa = mesh.facets_area if len(mesh.facets) else np.array([])
                if len(fa):
                    k = int(np.argmax(fa)); fidx = mesh.facets[k]; nrm = mesh.facets_normal[k]
                    cen = (mesh.triangles_center[fidx] * mesh.area_faces[fidx][:, None]).sum(0) / mesh.area_faces[fidx].sum()
                else:
                    i = int(np.argmax(mesh.area_faces)); nrm = mesh.face_normals[i]; cen = mesh.triangles_center[i]
                import trimesh as _tm
                Rm = _tm.geometry.align_vectors([0, 0, 1], nrm)
                t.apply_translation((v["dx"], v["dy"], -(v["d"] if v["mode"] == "gravar" else 0.5)))
                t.apply_transform(Rm)
                t.apply_translation(cen)
                return mo.boolean(mesh, t, "difference" if v["mode"] == "gravar" else "union")
            if v["face"] == "topo":
                z = b[1][2] - (v["d"] if v["mode"] == "gravar" else 0.5)
                t.apply_translation((c[0] + v["dx"], c[1] + v["dy"], z))
            elif v["face"] == "fundo":
                t = mo.mirror(t, 0); t = mo.rotate(t, 180, 0, 0, about_center=False)
                z = b[0][2] + (v["d"] if v["mode"] == "gravar" else 0.5)
                t.apply_translation((c[0] + v["dx"], c[1] + v["dy"], z - t.bounds[0][2] - t.extents[2]))
            else:
                t = mo.rotate(t, 90, 0, 0, about_center=False)
                y = b[0][1] + (v["d"] if v["mode"] == "gravar" else 0.5)
                t.apply_translation((c[0] + v["dx"], y - t.bounds[1][1], c[2] + v["dy"] - t.bounds.mean(axis=0)[2]))
            return mo.boolean(mesh, t, "difference" if v["mode"] == "gravar" else "union")

        def done(m):
            self.doc.snapshot(); o.mesh = m; self.doc.changed()
        self.task("A aplicar texto…", work, done)

    def act_slice_layers(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        v = self.ask("Fatiar em camadas", [("h", "Espessura de cada camada/chapa (mm)", 3.0, (0.05, 1000)),
                                           ("gap", "Espaço entre camadas no desenho (mm)", 5.0)],
                     note="Cria uma forma 2D por camada (ex.: para cortar em chapas de MDF e empilhar).")
        if not v:
            return
        mesh = objs[0].mesh.copy()

        def done(layers):
            shapes = []
            x = 0.0
            for i, (z, g) in enumerate(layers):
                if g.is_empty:
                    continue
                from shapely import affinity
                g = affinity.translate(g, x - g.bounds[0], -g.bounds[1])
                s = s2.Shape("geo", {}, g, f"camada {i + 1} (z={z:.1f})")
                shapes.append(s)
                x += g.bounds[2] - g.bounds[0] + v["gap"]
            self.canvas.add_shapes(shapes)
            self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()
            self.log(f"{len(shapes)} camadas criadas no separador 2D")
        self.task("A fatiar…", lambda: mo.slice_layers(mesh, v["h"]), done)

    def act_section(self):
        objs = self.sel_objs(1, True)
        if not objs:
            return
        c = objs[0].mesh.bounds.mean(axis=0)
        v = self.ask("Secção", [("ax", "Normal", "Z", ["X", "Y", "Z"]), ("pos", "Posição (mm)", float(round(c[2], 2)))])
        if not v:
            return
        n = np.zeros(3); ax = "XYZ".index(v["ax"]); n[ax] = 1
        o = c.copy(); o[ax] = v["pos"]
        g = mo.section_polygons(objs[0].mesh, o, n)
        if g is None or g.is_empty:
            return self.error("O plano não interseta o objeto.")
        self.canvas.add_shapes([s2.Shape("geo", {}, g, f"secção {v['ax']}={v['pos']}")])
        self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()
        self.log(f"Secção: área {g.area:.2f} mm², perímetro {g.length:.2f} mm")

    # ======================================================================
    # Desenho 2D
    # ======================================================================
    def set_tool(self, k):
        self.tabs.setCurrentWidget(self.canvas)
        self.canvas.set_tool(k)

    def _sel2(self, n=1):
        sel = self.canvas.selected_shapes()
        if len(sel) < n:
            self.error(f"Selecione pelo menos {n} forma(s) no separador 2D.")
            return None
        return sel

    def act_shape_data(self):
        sel = self.canvas.selected_shapes() or self.canvas.shapes
        txt = TextDialog.edit(self, "Dados das formas", "# " + s2.DATA_HELP.replace("\n", "\n# ") + "\n" + s2.to_data(sel))
        if txt is None:
            return
        try:
            shapes = s2.from_data(txt)
        except Exception as e:
            return self.error(f"Erro nos dados: {e}")
        self.canvas.selected = {s.id for s in sel}
        self.canvas.replace_selected(shapes)
        self.tabs.setCurrentWidget(self.canvas)

    def act_shape_info(self):
        sel = self._sel2()
        if sel:
            lines = []
            for s in sel:
                lines.append(f"[{s.id}] {s.name}" + (f"  (grupo {s.group})" if s.group else ""))
                lines += [f"    {k}: {v}" for k, v in s2.shape_info(s).items()]
            closed = [s for s in sel if s.closed]
            if len(closed) > 1:
                lines.append(f"\nÁrea total: {sum(s.geom.area for s in closed):.3f}")
            TextDialog.show_text(self, "Informação", "\n".join(lines))

    def _map2(self, fn):
        sel = self._sel2()
        if sel:
            self.canvas.replace_selected([fn(s) for s in sel])

    def act2_move(self):
        v = self.ask("Mover", [("x", "ΔX", 10.0), ("y", "ΔY", 0.0)])
        if v: self._map2(lambda s: s2.move(s, v["x"], v["y"]))

    def act2_rotate(self):
        v = self.ask("Rodar", [("a", "Ângulo (°, anti-horário)", 90.0)])
        if v: self._map2(lambda s: s2.rotate(s, v["a"]))

    def act2_scale(self):
        v = self.ask("Escalar", [("k", "Fator", 2.0)])
        if v: self._map2(lambda s: s2.scale(s, v["k"]))

    def act2_copy_to(self):
        """Copiar para: cópias da seleção com o canto inferior esquerdo
        numa posição, repetidas N vezes com o espaçamento indicado."""
        sel = self._sel2()
        if not sel:
            return
        from shapely.ops import unary_union
        b = unary_union([x.geom for x in sel]).bounds
        v = self.ask("Copiar para", [("x", "X do canto (mm)", float(b[2] + 10)), ("y", "Y do canto (mm)", float(b[1])),
                                     ("n", "Número de cópias", 1, (1, 1000)), ("dx", "Espaçamento X entre cópias (mm)", float(b[2] - b[0] + 10)),
                                     ("dy", "Espaçamento Y entre cópias (mm)", 0.0)])
        if not v:
            return
        new = []
        for k in range(int(v["n"])):
            ox, oy = v["x"] - b[0] + k * v["dx"], v["y"] - b[1] + k * v["dy"]
            for x in sel:
                c = s2.move(x.copy(), ox, oy); c.name = f"{x.name} (cópia {k + 1})"; new.append(c)
        self.canvas.add_shapes(new)
        self.log(f"Copiar para: {len(new)} forma(s)")

    def act2_rescale_xy(self):
        """Reescalar com fatores diferentes em X e Y, a partir do centro."""
        v = self.ask("Reescalar", [("kx", "Fator X", 1.0, (1e-4, 1e4)), ("ky", "Fator Y", 1.0, (1e-4, 1e4))])
        if v:
            from shapely import affinity
            self._map2(lambda x: s2.transformed(x, lambda g: affinity.scale(g, v["kx"], v["ky"], origin="centroid")))

    def act2_props(self):
        """Propriedades da forma: nome, posição, tamanho, área, perímetro e
        estilo — editáveis."""
        sel = self._sel2()
        if not sel:
            return
        x = sel[0]
        g = x.geom
        b = g.bounds
        st = x.style
        v = self.ask(f"Propriedades — {x.name}", [
            ("name", "Nome", x.name), ("x0", "X mínimo (mm)", float(b[0])), ("y0", "Y mínimo (mm)", float(b[1])),
            ("w", "Largura (mm)", float(b[2] - b[0])), ("h", "Altura (mm)", float(b[3] - b[1])),
            ("lw", "Espessura da linha (px, 0 = predefinida)", float(st.get("width") or 0), (0, 50)),
            ("dash", "Tipo de linha", st.get("dash", "contínua"), ["contínua", "tracejada", "pontilhada", "traço-ponto"]),
            ("hidden", "Oculta", bool(x.hidden))],
            note=f"Tipo: {g.geom_type} · área {g.area:.3f} mm² · perímetro {g.length:.3f} mm · {len(sel)} forma(s) selecionada(s)")
        if not v:
            return
        from shapely import affinity
        kx = v["w"] / (b[2] - b[0]) if b[2] - b[0] > 1e-9 else 1.0
        ky = v["h"] / (b[3] - b[1]) if b[3] - b[1] > 1e-9 else 1.0

        def fn(geom):
            gg = affinity.scale(geom, kx, ky, origin=(b[0], b[1]))
            return affinity.translate(gg, v["x0"] - b[0], v["y0"] - b[1])
        y = s2.transformed(x, fn)
        y.name = v["name"]; y.hidden = v["hidden"]
        y.style = dict(st); y.style["width"] = v["lw"] or None; y.style["dash"] = v["dash"]
        self.canvas.replace_selected([y] + [s for s in sel[1:]])

    def act2_mirror(self):
        self._map2(s2.mirror_x)

    def act2_offset(self):
        v = self.ask("Offset", [("d", "Distância (mm, negativo = para dentro)", 2.0)])
        if v: self._map2(lambda s: s2.offset(s, v["d"]))

    def act2_align(self, mode):
        sel = self._sel2(2)
        if sel: self.canvas.replace_selected(s2.align(sel, mode))

    def act2_bool(self, op):
        sel = self._sel2(2)
        if sel:
            try:
                self.canvas.replace_selected([s2.boolean(sel, op)])
            except Exception as e:
                self.error(e)

    def act2_group(self):
        sel = self._sel2(2)
        if sel:
            gid = f"g{sel[0].id}"
            for s in sel:
                s.group = gid
            self.log(f"Agrupadas {len(sel)} formas ({gid}); serão encaixadas juntas no nesting.")

    def act2_ungroup(self):
        sel = self._sel2()
        if sel:
            for s in sel:
                s.group = None

    def act2_array(self):
        sel = self._sel2()
        if not sel:
            return
        v = self.ask("Matriz", [("nx", "Colunas", 3, (1, 200)), ("ny", "Linhas", 2, (1, 200)),
                                ("dx", "Passo X (mm)", 30.0), ("dy", "Passo Y (mm)", 30.0)])
        if v:
            new = [s2.move(s, i * v["dx"], j * v["dy"]) for s in sel for i in range(v["nx"]) for j in range(v["ny"])
                   if i or j]
            self.canvas.add_shapes(new, select=False)

    def act2_snap(self):
        self.canvas.snap = not self.canvas.snap
        self.statusBar().showMessage("Ajuste à grelha " + ("ligado" if self.canvas.snap else "desligado") + " (Alt desliga temporariamente)")

    def act2_clear_overlay(self):
        self.canvas.clear_overlay(); self.toolpaths = []

    def act_nesting(self):
        from shapely.ops import unary_union
        shapes = [s for s in (self.canvas.selected_shapes() or self.canvas.shapes) if s.closed]
        if not shapes:
            return self.error("Não há formas fechadas para encaixar (desenhe ou importe, ou use Ficheiro ▸ Exemplos ▸ Formas 2D).")
        v = self.ask("Nesting", [("w", "Largura da chapa (mm)", 600.0), ("h", "Altura da chapa (mm)", 400.0),
                                 ("sp", "Espaço entre peças (mm)", 3.0), ("mg", "Margem (mm)", 5.0),
                                 ("rot", "Rotações", "0, 90, 180, 270", ["0", "0, 180", "0, 90, 180, 270", "a cada 45°", "a cada 15°"]),
                                 ("copies", "Cópias de cada peça", 1, (1, 500))])
        if not v:
            return
        rots = {"0": (0,), "0, 180": (0, 180), "0, 90, 180, 270": (0, 90, 180, 270),
                "a cada 45°": tuple(range(0, 360, 45)), "a cada 15°": tuple(range(0, 360, 15))}[v["rot"]]
        groups = {}
        for s in shapes:
            groups.setdefault(s.group or f"s{s.id}", []).append(s)
        items = []
        members = {}
        n = 0
        for key, ss in groups.items():
            g = unary_union([s.geom for s in ss])
            for c in range(v["copies"]):
                n += 1
                items.append((n, g))
                members[n] = (key, ss)

        def done(res):
            plates, un = res
            from shapely import affinity
            self.canvas.push_undo()
            removed = {s.id for s in shapes}
            self.canvas.shapes = [s for s in self.canvas.shapes if s.id not in removed]
            self.canvas.clear_overlay()
            new = []
            for pi, pl in enumerate(plates):
                ox = pi * (v["w"] + 30)
                self.canvas.add_overlay_rect(ox, 0, v["w"], v["h"])
                for iid, g, a in pl["placements"]:
                    key, ss = members[iid]
                    src = unary_union([s.geom for s in ss])
                    # transforma cada membro tal como o conjunto: rodar em torno do centróide e alinhar
                    rot = affinity.rotate(src, a, origin=src.centroid)
                    dx, dy = g.bounds[0] - rot.bounds[0] + ox, g.bounds[1] - rot.bounds[1]
                    for s in ss:
                        gg = affinity.translate(affinity.rotate(s.geom, a, origin=src.centroid), dx, dy)
                        ns = s2.Shape("geo", {}, gg, s.name); ns.group = s.group
                        new.append(ns)
            self.canvas.shapes += new
            self.canvas.selected = {s.id for s in new}
            self.canvas.redraw(); self.canvas.fit()
            rep = nesting.report(plates, un, v["w"], v["h"])
            self.log(rep)
            TextDialog.show_text(self, "Resultado do nesting", rep)
        self.task(f"A encaixar {len(items)} peças…",
                  lambda: nesting.nest(items, v["w"], v["h"], v["sp"], v["mg"], rots), done)

    def act_toolpath(self):
        sel = self.canvas.selected_shapes() or self.canvas.shapes
        if not sel:
            return self.error("Não há formas 2D.")
        v = self.ask("Percurso de ferramenta", [
            ("kind", "Tipo", "contorno exterior", ["contorno exterior", "contorno interior", "sobre a linha", "bolsa", "tramado (laser)"]),
            ("d", "Diâmetro da ferramenta / feixe (mm)", 3.0), ("so", "Sobreposição bolsa (fração do diâmetro)", 0.5),
            ("hs", "Espaçamento do tramado (mm)", 0.3), ("ha", "Ângulo do tramado (°)", 0.0)])
        if not v:
            return
        paths = []
        for s in sel:
            g = s.geom
            k = v["kind"]
            if k == "sobre a linha" or not s.closed:
                paths += tp.follow(g)
            elif k == "contorno exterior":
                paths += tp.contour(g, v["d"], "outside")
            elif k == "contorno interior":
                paths += tp.contour(g, v["d"], "inside")
            elif k == "bolsa":
                paths += tp.pocket(g, v["d"], v["so"])
            else:
                paths += tp.hatch(g, v["hs"], v["ha"])
        self.toolpaths = tp.order_paths(paths)
        self.canvas.clear_overlay()
        self.canvas.add_overlay_paths(self.toolpaths)
        self.log(f"Percurso: {len(paths)} trajetos, {tp.path_length(paths):.1f} mm de corte")

    def act_gcode(self):
        if not self.toolpaths:
            return self.error("Crie primeiro um percurso (Desenho 2D ▸ Percurso de ferramenta).")
        v = self.ask("G-code", [("mode", "Máquina", "laser", ["laser", "cnc"]), ("feed", "Velocidade de corte (mm/min)", 1000.0),
                                ("plunge", "Velocidade de descida (mm/min, CNC)", 300.0), ("depth", "Profundidade total (mm, CNC)", 3.0),
                                ("passes", "Passagens", 1, (1, 100)), ("safe", "Z seguro (mm, CNC)", 5.0),
                                ("power", "Potência laser (S)", 1000.0), ("rpm", "Rotação fresa (rpm)", 12000.0)])
        if not v:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Guardar G-code", os.path.join(self.last_dir, "corte.gcode"),
                                           "G-code (*.gcode *.nc *.ngc)")
        if not p:
            return
        self._dir(p)
        code = tp.gcode(self.toolpaths, v["mode"], v["feed"], v["plunge"], v["safe"], v["depth"], v["passes"],
                        v["power"], v["rpm"])
        open(p, "w").write(code)
        t = tp.estimate_time(self.toolpaths, v["feed"], v["passes"])
        self.log(f"G-code guardado: {p} ({code.count(chr(10))} linhas, tempo estimado {t / 60:.1f} min)")

    # ======================================================================
    # Imagem
    # ======================================================================
    def img_op(self, label, fn, item=None):
        """Aplica fn à imagem atual, ou a todas as selecionadas (processamento em lote)."""
        if item is not None:
            items = [item]
        else:
            rows = sorted({i.row() for i in self.img_list.selectedIndexes()})
            items = [self.images[r] for r in rows if not self.images[r].meta.get("volume")] if len(rows) > 1 else []
            if not items:
                it = self.cur_image()
                if not it:
                    return
                items = [it]
        for k, it in enumerate(items):
            try:
                out = fn(it.data)
            except Exception as e:
                return self.error(f"{it.name}: {e}")
            self.add_image(f"{label}({it.name})", out, show=k == len(items) - 1)
        if len(items) > 1:
            self.log(f"{label}: aplicado a {len(items)} imagens")

    def act_threshold(self):
        v = self.ask("Limiar", [("mode", "Método", "otsu", ["otsu", "fixo", "adaptativo"]), ("val", "Valor (fixo)", 128, (0, 255)),
                                ("block", "Bloco (adaptativo)", 31, (3, 501)), ("c", "Constante C (adaptativo)", 5, (-100, 100)),
                                ("inv", "Inverter", False)])
        if v:
            self.img_op("limiar", lambda a: iop.threshold(a, v["mode"], v["val"], v["block"], v["c"], v["inv"]))

    def act_morph(self):
        v = self.ask("Morfologia", [("op", "Operação", "abrir", ["erodir", "dilatar", "abrir", "fechar", "gradiente", "tophat", "blackhat"]),
                                    ("size", "Tamanho do elemento (px)", 5, (1, 201)), ("shape", "Forma", "elipse", ["elipse", "rect", "cruz"]),
                                    ("it", "Iterações", 1, (1, 50))])
        if v:
            self.img_op(v["op"], lambda a: iop.morphology(a, v["op"], v["size"], v["shape"], v["it"]))

    def act_filter(self):
        v = self.ask("Filtro", [("k", "Tipo", "gaussiano", ["gaussiano", "mediana", "bilateral", "média", "nitidez"]),
                                ("size", "Tamanho (px, ímpar)", 5, (1, 101)), ("sigma", "Sigma", 1.5)])
        if v:
            self.img_op(v["k"], lambda a: iop.filter_(a, v["k"], v["size"], v["sigma"]))

    def act_edges(self):
        v = self.ask("Arestas", [("k", "Método", "canny", ["canny", "sobel", "laplaciano"]), ("lo", "Limiar baixo (Canny)", 50, (0, 1000)),
                                 ("hi", "Limiar alto (Canny)", 150, (0, 1000))])
        if v:
            self.img_op(v["k"], lambda a: iop.edges(a, v["k"], v["lo"], v["hi"]))

    def act_contours(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Contornos", [("ma", "Área mínima (px²)", 20.0), ("px", "Tamanho do pixel (mm, 0 = px)", 0.0),
                                   ("auto", "Limiarizar automaticamente (Otsu) se não for binária", True)])
        if not v:
            return
        a = np.asarray(it.data)
        binary = a if (a.dtype == np.uint8 and set(np.unique(a)) <= {0, 255}) or not v["auto"] else iop.threshold(a, "otsu")
        cs = iop.contours(binary, v["ma"])
        k = v["px"] or 1.0
        u = "mm" if v["px"] else "px"
        lines = [f"{len(cs)} objetos"]
        for c in cs:
            rx, ry, rw, rh, ang = c["rect_rodado"]
            lines.append(f"#{c['id']}: área={c['área'] * k * k:.4g} {u}²  perímetro={c['perímetro'] * k:.4g} {u}  "
                         f"centro=({c['centro'][0]:.1f}, {c['centro'][1]:.1f})  caixa rodada={rw * k:.4g}×{rh * k:.4g} {u} @ {ang:.1f}°")
        self.add_image(f"contornos({it.name})", iop.draw_overlay(a, cs))
        self.log("\n".join(lines))
        TextDialog.show_text(self, "Contornos", "\n".join(lines))

    def act_circles(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Círculos", [("md", "Distância mínima (px)", 20.0), ("p2", "Sensibilidade (menor = mais círculos)", 30.0),
                                  ("rmin", "Raio mínimo (px)", 5, (0, 5000)), ("rmax", "Raio máximo (px, 0 = sem)", 0, (0, 5000))])
        if v:
            cs = iop.circles(it.data, 1.2, v["md"], 100, v["p2"], v["rmin"], v["rmax"])
            self.add_image(f"círculos({it.name})", iop.draw_overlay(it.data, circs=cs))
            self.log(f"{len(cs)} círculos: " + "; ".join(f"({x:.1f},{y:.1f}) r={r:.1f}" for x, y, r in cs[:50]))

    def act_corners(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Cantos", [("n", "Máximo", 200, (1, 10000)), ("q", "Qualidade", 0.01), ("d", "Distância mínima (px)", 10.0)])
        if v:
            pts = iop.corners(it.data, v["n"], v["q"], v["d"])
            self.add_image(f"cantos({it.name})", iop.draw_overlay(it.data, pts=pts))
            self.log(f"{len(pts)} cantos")

    def act_vectorize(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Vetorizar", [("px", "Tamanho do pixel (mm)", 0.25), ("ma", "Área mínima (px²)", 30.0),
                                   ("simp", "Simplificação (px)", 0.8), ("holes", "Incluir furos", True)])
        if not v:
            return
        import cv2
        from shapely.geometry import Polygon
        a = np.asarray(it.data)
        binary = a if (a.dtype == np.uint8 and set(np.unique(a)) <= {0, 255}) else iop.threshold(a, "otsu")
        cs, hier = cv2.findContours((binary > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        h = a.shape[0]
        shapes = []
        if hier is not None:
            hier = hier[0]
            for i, c in enumerate(cs):
                if hier[i][3] != -1 or cv2.contourArea(c) < v["ma"] or len(c) < 3:
                    continue
                ext = [(x * v["px"], (h - y) * v["px"]) for x, y in c[:, 0, :]]
                holes = []
                if v["holes"]:
                    j = hier[i][2]
                    while j != -1:
                        if cv2.contourArea(cs[j]) >= v["ma"] and len(cs[j]) >= 3:
                            holes.append([(x * v["px"], (h - y) * v["px"]) for x, y in cs[j][:, 0, :]])
                        j = hier[j][0]
                g = Polygon(ext, holes).buffer(0).simplify(v["simp"] * v["px"])
                if not g.is_empty:
                    shapes.append(s2.Shape("geo", {}, g, f"contorno {len(shapes) + 1}"))
        self.canvas.add_shapes(shapes)
        self.tabs.setCurrentWidget(self.canvas); self.canvas.fit()
        self.log(f"Vetorizado: {len(shapes)} formas")

    def act_arith(self):
        items = self.sel_images(2)
        if not items:
            return
        v = self.ask("Aritmética", [("op", "Operação", "subtrair", ["somar", "subtrair", "diferença abs.", "multiplicar", "dividir",
                                                                     "média", "máximo", "mínimo"])],
                     note=f"A = {items[0].name}\nB = {items[1].name}")
        if v:
            self.add_image(f"{v['op']}({items[0].name}, {items[1].name})", iop.arithmetic(items[0].data, items[1].data, v["op"]))

    def act_img_rotate(self):
        v = self.ask("Rodar imagem", [("a", "Ângulo (°)", 90.0)])
        if v: self.img_op("rodada", lambda a: iop.rotate(a, v["a"]))

    def act_img_resize(self):
        v = self.ask("Redimensionar", [("k", "Fator", 0.5, (0.01, 20))])
        if v: self.img_op("redim", lambda a: iop.resize(a, v["k"]))

    def act_img_crop(self):
        it = self.cur_image()
        if not it:
            return
        h, w = it.data.shape[:2]
        v = self.ask("Recortar", [("x", "X", 0, (0, w)), ("y", "Y", 0, (0, h)), ("w", "Largura", w, (1, w)), ("h", "Altura", h, (1, h))])
        if v: self.img_op("recorte", lambda a: iop.crop(a, v["x"], v["y"], v["w"], v["h"]))

    def act_img_stats(self):
        it = self.cur_image()
        if it:
            a = np.asarray(it.data, float)
            f = np.isfinite(a)
            TextDialog.show_text(self, "Estatísticas", f"{it.name}\nDimensões: {a.shape}\nTipo: {it.data.dtype}\n"
                                 f"Mín: {np.nanmin(a):.6g}\nMáx: {np.nanmax(a):.6g}\nMédia: {np.nanmean(a):.6g}\n"
                                 f"Desvio-padrão: {np.nanstd(a):.6g}\nPixels válidos: {f.sum()} / {a.size}")

    def act_img_remove(self):
        rows = sorted({i.row() for i in self.img_list.selectedIndexes()}, reverse=True)
        for r in rows:
            self.img_list.takeItem(r); del self.images[r]

    # ======================================================================
    # Metrologia
    # ======================================================================
    def act_gen_fringes(self):
        v = self.ask("Franjas", [("w", "Largura (px)", 1280, (2, 10000)), ("h", "Altura (px)", 800, (2, 10000)),
                                 ("p", "Passo / período (px)", 32.0, (2, 10000)), ("n", "Número de passos", 4, (1, 360)),
                                 ("dir", "Direção", "vertical", ["vertical", "horizontal"]), ("ph", "Desfasamento inicial (°)", 0.0)],
                     note="Projete estas imagens com um projetor e fotografe cada uma.")
        if not v:
            return
        try:
            imgs = fr.fringe_patterns(v["w"], v["h"], v["p"], v["n"], v["dir"], v["ph"])
        except ValueError as e:
            return self.error(e)
        for i, im in enumerate(imgs):
            self.add_image(f"padrão p{v['p']:g} {i + 1}/{v['n']}", im, show=i == len(imgs) - 1)

    def act_gen_gray(self):
        v = self.ask("Código de Gray", [("w", "Largura (px)", 1280, (2, 10000)), ("h", "Altura (px)", 800, (2, 10000)),
                                        ("b", "Bits (0 = automático)", 0, (0, 16))])
        if v:
            for i, im in enumerate(fr.gray_code_patterns(v["w"], v["h"], v["b"] or None)):
                self.add_image(f"gray bit {i + 1}", im, show=False)
            self.img_list.setCurrentRow(len(self.images) - 1)

    def act_wrapped_phase(self):
        items = self.sel_images(3)
        if not items:
            return
        v = self.ask("Fase embrulhada", [("shifts", "Desfasamentos (°, vazio = 360/N uniformes)", ""),
                                         ("thr", "Limiar de modulação (fração do máximo)", 0.1, (0, 1))],
                     note="Imagens por ordem da lista:\n" + "\n".join(i.name for i in items))
        if not v:
            return
        shifts = [float(x) for x in v["shifts"].split(",")] if v["shifts"].strip() else None

        def done(r):
            phi, mod, bg = r
            mask = fr.modulation_mask(mod, v["thr"])
            base = items[0].name
            self.add_image(f"modulação({base})", mod, show=False)
            self.add_image(f"fundo({base})", bg, show=False)
            self.add_image(f"fase({base})", phi, {"mask": mask})
            self.log(f"Fase embrulhada de {len(items)} imagens; {mask.mean() * 100:.1f}% pixels válidos")
        self.task("A calcular fase…", lambda: fr.wrapped_phase([i.data for i in items], shifts), done)

    def act_unwrap(self, method):
        it = self.cur_image()
        if not it:
            return
        mask = it.meta.get("mask")
        fn = fr.unwrap_quality_guided if method == "quality" else fr.unwrap_left_to_right
        phi = np.asarray(it.data, float)

        def done(U):
            self.add_image(f"desembrulhada({it.name})", U, {"mask": mask})
        self.task("A desembrulhar…", lambda: fn(phi, mask), done)

    def act_unwrap_multi(self):
        items = self.sel_images(2)
        if not items:
            return
        v = self.ask("Multi-frequência", [("p", "Passos (px) de cada fase, pela ordem da lista", ", ".join("?" for _ in items))],
                     note="Selecione as fases embrulhadas de cada frequência. A de maior passo deve cobrir o campo com ≤ 1 período.\n"
                          + "\n".join(i.name for i in items))
        if not v:
            return
        try:
            pitches = [float(x) for x in v["p"].split(",")]
            assert len(pitches) == len(items)
        except Exception:
            return self.error("Indique um passo por fase, separados por vírgulas.")
        self.add_image("desembrulhada multi-freq", fr.unwrap_multifrequency([np.asarray(i.data, float) for i in items], pitches))

    def act_unwrap_gray(self):
        items = self.sel_images(2)
        if not items:
            return
        self.add_image(f"desembrulhada gray({items[0].name})",
                       fr.unwrap_gray_code(np.asarray(items[0].data, float), [i.data for i in items[1:]]))
        self.log("Usada a primeira imagem selecionada como fase e as restantes como bits de Gray (do mais significativo).")

    def act_phase_height(self):
        items = self.sel_images(1)
        if not items:
            return
        v = self.ask("Fase → altura", [("k", "Fator (mm por radiano)", 1.0), ("tilt", "Remover plano de inclinação", True)],
                     note="Fase do objeto: " + items[0].name + ("\nFase de referência: " + items[1].name if len(items) > 1 else
                                                                 "\n(Selecione também a fase de referência para a subtrair.)"))
        if not v:
            return
        Phi = np.asarray(items[0].data, float)
        ref = np.asarray(items[1].data, float) if len(items) > 1 else None
        mask = items[0].meta.get("mask")
        H = fr.phase_to_height(Phi, ref, v["k"], v["tilt"], mask)
        if mask is not None:
            H = np.where(mask, H, np.nan)
        self.add_image(f"altura_mm({items[0].name})", H)

    def act_depth_ply(self):
        it = self.cur_image()
        if not it:
            return
        v = self.ask("Nuvem de pontos", [("px", "Tamanho do pixel (mm)", 0.2), ("step", "Saltar pixels", 1, (1, 50))])
        if not v:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Guardar PLY", os.path.join(self.last_dir, "nuvem.ply"), "PLY (*.ply)")
        if not p:
            return
        Z = np.asarray(it.data, float)[::v["step"], ::v["step"]]
        pts, _ = fr.depth_to_points(Z, v["px"] * v["step"])
        fr.save_ply(p, pts)
        self.log(f"Nuvem com {len(pts)} pontos guardada em {p}")

    def act_deflecto(self):
        items = self.sel_images(2)
        if not items:
            return
        v = self.ask("Deflectometria", [("g", "Ganho (declive por radiano)", 1.0)],
                     note=f"Fase X: {items[0].name}\nFase Y: {items[1].name}\n(fases já desembrulhadas)")
        if v:
            self.add_image("superfície deflectometria", fr.deflectometry(np.asarray(items[0].data, float),
                                                                          np.asarray(items[1].data, float), v["g"]))

    def act_focus(self):
        items = self.sel_images(3)
        if not items:
            return
        v = self.ask("Profundidade por foco", [("z0", "Z inicial (mm)", -2.0), ("z1", "Z final (mm)", 28.0),
                                               ("win", "Janela de foco (px)", 7, (3, 51))],
                     note=f"{len(items)} imagens, por ordem da lista, igualmente espaçadas em Z.")
        if not v:
            return
        zs = np.linspace(v["z0"], v["z1"], len(items))

        def done(r):
            Z, conf = r
            self.add_image("confiança foco", conf, show=False)
            self.add_image("altura por foco (mm)", Z)
        self.task("A analisar foco…", lambda: fr.depth_from_focus([i.data for i in items], zs, v["win"]), done)

    def act_coherence(self):
        items = self.sel_images(3)
        if not items:
            return
        v = self.ask("Coerência (luz branca)", [("z0", "Z inicial (µm)", 0.0), ("z1", "Z final (µm)", 10.0)])
        if v:
            Z, env = fr.coherence_scan([iop.to_gray(i.data) for i in items], np.linspace(v["z0"], v["z1"], len(items)))
            self.add_image("altura coerência (µm)", Z)

    def act_speckle_contrast(self):
        v = self.ask("Contraste de speckle", [("w", "Janela (px)", 7, (3, 101))])
        if v: self.img_op("contraste speckle", lambda a: fr.speckle_contrast(iop.to_gray(a), v["w"]))

    def act_speckle_reduce(self):
        v = self.ask("Redução de speckle", [("m", "Método", "median", ["median", "lee", "nlmeans"]), ("s", "Tamanho", 5, (1, 51))])
        if v: self.img_op("sem speckle", lambda a: fr.speckle_reduce(iop.to_gray(a), v["m"], v["s"]))

    def act_visual_hull(self):
        items = self.sel_images(2)
        if not items:
            return
        n = len(items)
        v = self.ask("Visual hull", [("a0", "Ângulo inicial (°)", 0.0), ("step", "Passo angular (°)", 360.0 / n),
                                     ("px", "Tamanho do pixel (mm)", 0.5), ("res", "Resolução (voxels)", 96, (16, 400)),
                                     ("auto", "Limiarizar (Otsu) se não for binária", True)],
                     note=f"{n} silhuetas, por ordem da lista, tiradas com o objeto a rodar numa mesa (câmara de lado).")
        if not v:
            return
        sil = []
        for i in items:
            a = np.asarray(i.data)
            sil.append(a if not v["auto"] or (a.dtype == np.uint8 and set(np.unique(a)) <= {0, 255}) else iop.threshold(a, "otsu"))
        angs = [v["a0"] + k * v["step"] for k in range(n)]
        self.task("A esculpir voxels…", lambda: iop.visual_hull(sil, angs, v["px"], v["res"]),
                  lambda m: (self.add_mesh(m, "visual hull"), self.tabs.setCurrentWidget(self.view3d), self.view3d.fit()))

    def act_calibrate(self):
        items = self.sel_images(3)
        if not items:
            return
        v = self.ask("Calibração", [("c", "Cantos internos (colunas)", 9, (2, 50)), ("r", "Cantos internos (linhas)", 6, (2, 50)),
                                    ("s", "Tamanho do quadrado (mm)", 10.0)])
        if not v:
            return

        def done(r):
            np.set_printoptions(precision=4, suppress=True)
            txt = (f"Imagens usadas: {r['imagens_usadas']}\nErro RMS: {r['erro_rms_px']:.4f} px\n\n"
                   f"Matriz da câmara K:\n{r['K']}\n\nDistorção (k1 k2 p1 p2 k3):\n{r['distorção']}")
            self.calib = r
            self.log(txt); TextDialog.show_text(self, "Calibração", txt)
        self.task("A calibrar…", lambda: iop.calibrate_chessboard([i.data for i in items], v["c"], v["r"], v["s"]), done)

    def act_capture(self):
        v = self.ask("Câmara", [("cam", "Índice da câmara", 0, (0, 10)), ("n", "Imagens", 1, (1, 100))])
        if not v:
            return
        # no macOS a autorização da câmara tem de ser pedida no fio principal
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            imgs = iop.capture(v["cam"], v["n"])
        except Exception as e:
            return self.error(e)
        finally:
            QApplication.restoreOverrideCursor()
        for i, im in enumerate(imgs):
            self.add_image(f"câmara {i + 1}", im, show=i == len(imgs) - 1)

    # ======================================================================
    # Medir / Ajuda
    # ======================================================================
    def act_properties(self):
        objs = self.sel_objs()
        if objs:
            lines = []
            for o in objs:
                lines.append(f"== {o.name} ==")
                lines += [f"  {k}: {v}" for k, v in mo.properties(o.mesh).items()]
                b = o.mesh.bounds
                lines.append(f"  Caixa: ({b[0][0]:.2f}, {b[0][1]:.2f}, {b[0][2]:.2f}) → ({b[1][0]:.2f}, {b[1][1]:.2f}, {b[1][2]:.2f})")
                lines.append(f"  Faces com saliência > 45°: {int(mo.overhang_faces(o.mesh).sum())}")
            TextDialog.show_text(self, "Propriedades", "\n".join(lines))

    def act_report(self):
        if not self.doc.objects:
            return self.error("Não há objetos.")
        lines = ["RELATÓRIO FORJA", "=" * 40]
        total = 0.0
        for o in self.doc.objects:
            pr = mo.properties(o.mesh)
            lines.append(f"\n{o.name}")
            lines += [f"  {k}: {v}" for k, v in pr.items()]
            if o.mesh.is_watertight:
                total += abs(o.mesh.volume)
        lines.append(f"\nVolume total (objetos fechados): {total / 1000:.3f} cm³  ≈ {total / 1000 * 1.24:.1f} g de PLA")
        txt = "\n".join(lines)
        TextDialog.show_text(self, "Relatório", txt)
        p, _ = QFileDialog.getSaveFileName(self, "Guardar relatório (opcional)", os.path.join(self.last_dir, "relatorio.txt"), "Texto (*.txt)")
        if p:
            open(p, "w").write(txt)

    def act_distance(self):
        objs = self.sel_objs(2, True)
        if objs:
            a, b = (o.mesh.bounds.mean(axis=0) for o in objs)
            d = b - a
            self.log(f"Distância entre centros: {np.linalg.norm(d):.3f} mm  (Δ = {d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})")
            self.statusBar().showMessage(f"Distância: {np.linalg.norm(d):.3f} mm")

    def act_help(self):
        TextDialog.show_text(self, "Ajuda", """NAVEGAÇÃO 3D
  Arrastar com botão esquerdo: rodar a vista
  Arrastar com botão do meio / Ctrl+arrastar: deslocar
  Roda do rato / dois dedos: zoom
  ⌘0: ajustar à janela
  Selecione objetos na lista "Objetos 3D" (Shift/⌘ para vários). A caixa liga/desliga a visibilidade.
  Duplo clique no nome: propriedades.

TELA 2D
  Roda: zoom. Botão direito/meio ou ⌘+arrastar: deslocar.
  Ferramentas na barra ou em Desenho 2D ▸ Ferramenta.
  Polígono/polilinha/spline: clique para cada ponto, duplo clique ou Enter para terminar.
  Selecionar: clique, Shift+clique para juntar, arrastar no vazio para caixa de seleção, arrastar a forma para mover.
  Alt durante o desenho desliga o ajuste à grelha. Delete apaga.

IMAGENS
  A ordem na lista "Imagens" (arrastável) é a ordem usada nas operações com várias imagens.
  Selecione várias com Shift/⌘. Passe o rato sobre a imagem para ver valores.

FLUXOS TÍPICOS
  Modelo grande para imprimir: Abrir STL → Malha ▸ Partição em grelha (com pinos) → Exportar cada objeto.
  Poupar material: Malha ▸ Oco.
  Corte laser em camadas: Malha ▸ Fatiar em camadas → Desenho 2D ▸ Nesting → Percurso → G-code / DXF.
  Digitalização 3D: Metrologia ▸ Gerar padrões → fotografar → Fase embrulhada → Desembrulhar → Fase→altura → Criar ▸ Mapa de profundidade → 3D.
  Com projetor e câmara: Dispositivos ▸ Aquisição automática de franjas faz a projeção e as fotos sozinho.
  Caixa em MDF/acrílico: Chapa ▸ Caixa com juntas em dente → Desenho 2D ▸ Percurso → G-code/DXF.
  Escultura em fatias: Chapa ▸ Escultura em grelha de fatias (ou radial) → Nesting → corte.
  Candeeiro: Gerar ▸ Fotografia ▸ Litofania cilíndrica + Base de candeeiro.
  Ensaio de deformação: Metrologia ▸ DIC (antes/depois) → Deformações.
  Tomografia: CT / Volume ▸ Projeções → volume → Isosuperfície → objeto 3D para imprimir.

MENUS
  Ficheiro, Editar, Ver (vistas, iluminação, régua 3D com R, projeção F1, texturas), Criar, Transformar, Malha,
  Desenho 2D, Imagem (filtros, frequência, análise, cor), Lote (pastas, vídeo, estatísticas), Metrologia,
  Medir, Ajuste (planos, polinómios, círculos, curvatura), Gerar (caixas, vasos, molas, cortadores, moldes,
  puzzles, suportes…), Chapa (laser/CNC), CT / Volume, Visão (rostos, ArUco, QR, foco total, movimento),
  Dispositivos (câmara, projetor, porta série, mesas).

Erros inesperados aparecem numa mensagem e ficam no ficheiro erros.log, na pasta da Forja (na versão instalada, em ~/.forja).
""")

    def act_about(self):
        from .. import __version__
        QMessageBox.about(self, "Sobre a Forja",
                          f"<b>Forja {__version__}</b> — ferramenta de prototipagem rápida.<br><br>"
                          "Malhas 3D, partição, oco, booleanas, texto e relevos; desenho 2D, nesting, percursos e G-code; "
                          "processamento de imagem e metrologia ótica (luz estruturada, foco, deflectometria, visual hull).<br><br>"
                          "Implementação própria em Python (PySide6, trimesh, manifold3d, shapely, OpenCV, scikit-image).")
