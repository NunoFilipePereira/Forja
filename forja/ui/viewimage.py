"""Visualizador de imagens/mapas (dados numéricos) com leitura de pixel e barra de cores."""
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QLinearGradient, QColor
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QVBoxLayout, QLabel,
                               QHBoxLayout, QSlider, QComboBox)
from ..core import imageops


class ImageItem:
    def __init__(self, name, data, meta=None):
        self.name = name
        self.data = data
        self.meta = meta or {}


def array_to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim == 2:
        h, w = rgb.shape
        return QImage(rgb.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class _Graphics(QGraphicsView):
    hovered = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pix = QGraphicsPixmapItem(); self.scene().addItem(self.pix)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setBackgroundBrush(QColor(30, 30, 34))

    def wheelEvent(self, e):
        k = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(k, k)

    def mouseMoveEvent(self, e):
        p = self.mapToScene(e.position().toPoint())
        self.hovered.emit(int(p.x()), int(p.y()))
        super().mouseMoveEvent(e)


class ColorBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(10)
        self.lo = self.hi = None
        self.cmap = None

    def set(self, cmap, lo, hi):
        self.cmap, self.lo, self.hi = cmap, lo, hi
        self.setVisible(cmap is not None)
        self.update()

    def _rgb(self, t):
        from ..core import colormaps as CM
        if self.cmap in CM.SCHEMES:
            c = CM.colors(np.array([t]), self.cmap, getattr(self, "solid", (0.8, 0.8, 0.8)))[0]
            return c[0], c[1], c[2]
        import matplotlib
        r, gg, b, _ = matplotlib.colormaps[self.cmap](t)
        return r, gg, b

    def paintEvent(self, e):
        if self.cmap is None:
            return
        p = QPainter(self)
        w, h = self.width(), self.height()
        if h > w:  # barra vertical: mínimo em baixo
            g = QLinearGradient(0, h - 20, 0, 20)
            for t in np.linspace(0, 1, 16):
                g.setColorAt(t, QColor.fromRgbF(*self._rgb(t)))
            bw = max(6, w - 50)
            p.fillRect(QRectF(0, 20, bw, h - 40), g)
            p.setPen(QColor(220, 220, 220))
            p.drawText(QRectF(bw + 4, h - 30, 60, 20), Qt.AlignLeft | Qt.AlignVCenter, f"{self.lo:.4g}")
            p.drawText(QRectF(bw + 4, 10, 60, 20), Qt.AlignLeft | Qt.AlignVCenter, f"{self.hi:.4g}")
            return
        g = QLinearGradient(60, 0, w - 60, 0)
        for t in np.linspace(0, 1, 16):
            g.setColorAt(t, QColor.fromRgbF(*self._rgb(t)))
        p.fillRect(QRectF(60, 4, w - 120, h - 8), g)
        p.setPen(QColor(220, 220, 220))
        p.drawText(QRectF(0, 0, 56, h), Qt.AlignRight | Qt.AlignVCenter, f"{self.lo:.4g}")
        p.drawText(QRectF(w - 56, 0, 56, h), Qt.AlignLeft | Qt.AlignVCenter, f"{self.hi:.4g}")


class ImageView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.g = _Graphics(); lay.addWidget(self.g)
        # controlo de cortes para volumes 3D
        self.volbar = QWidget(); vl = QHBoxLayout(self.volbar); vl.setContentsMargins(4, 0, 4, 0)
        self.vol_axis = QComboBox(); self.vol_axis.addItems(["Corte Z (axial)", "Corte Y", "Corte X"])
        self.vol_slider = QSlider(Qt.Horizontal); self.vol_label = QLabel("")
        vl.addWidget(self.vol_axis); vl.addWidget(self.vol_slider, 1); vl.addWidget(self.vol_label)
        self.volbar.setVisible(False); lay.addWidget(self.volbar)
        self.vol_axis.currentIndexChanged.connect(self._vol_axis_changed)
        self.vol_slider.valueChanged.connect(lambda v: self._render())
        self.bar = ColorBar(); lay.addWidget(self.bar)
        self.info = QLabel(" "); lay.addWidget(self.info)
        self.item = None
        self.cmap = "viridis"
        self.g.hovered.connect(self._hover)
        self.mag = QLabel(self.g.viewport()); self.mag.setFixedSize(180, 180); self.mag.setVisible(False)
        self.mag.setStyleSheet("border: 2px solid #f0c040; background: black")
        self.mag_on = False

    def is_volume(self, item=None):
        it = item or self.item
        return it is not None and it.meta.get("volume") and np.ndim(it.data) == 3

    def current_slice(self):
        if not self.is_volume():
            return np.asarray(self.item.data)
        ax = self.vol_axis.currentIndex()
        return np.take(self.item.data, min(self.vol_slider.value(), self.item.data.shape[ax] - 1), axis=ax)

    def _vol_axis_changed(self, *_):
        if self.is_volume():
            n = self.item.data.shape[self.vol_axis.currentIndex()]
            self.vol_slider.blockSignals(True)
            self.vol_slider.setRange(0, n - 1); self.vol_slider.setValue(n // 2)
            self.vol_slider.blockSignals(False)
            self._render(refit=True)

    def show_item(self, item: ImageItem | None):
        self.item = item
        if item is None:
            self.g.pix.setPixmap(QPixmap()); self.bar.set(None, 0, 0); self.info.setText(" ")
            self.volbar.setVisible(False)
            return
        vol = self.is_volume(item)
        self.volbar.setVisible(bool(vol))
        if vol:
            self._vol_axis_changed()
        else:
            self._render()

    def _render(self, refit=False):
        item = self.item
        if item is None:
            return
        d = self.current_slice()
        if self.is_volume():
            ax = self.vol_axis.currentIndex()
            vmin, vmax = float(np.nanmin(item.data)), float(np.nanmax(item.data))
            rgb, _ = imageops.colorize(np.clip(d, vmin, vmax), self.cmap)
            self.bar.set(self.cmap, vmin, vmax)
            self.vol_label.setText(f"{self.vol_slider.value() + 1}/{item.data.shape[ax]}")
        elif d.dtype == np.uint8 and (d.ndim == 2 or d.shape[2] == 3):
            rgb = d; self.bar.set(None, 0, 0)
        else:
            rgb, (lo, hi) = imageops.colorize(d if d.ndim == 2 else d.mean(2), self.cmap)
            self.bar.set(self.cmap, lo, hi)
        first = self.g.pix.pixmap().isNull() or refit
        self.g.pix.setPixmap(QPixmap.fromImage(array_to_qimage(rgb)))
        self.g.scene().setSceneRect(0, 0, d.shape[1], d.shape[0])
        if first:
            self.fit()
        shape = "×".join(str(x) for x in np.shape(item.data)[::-1]) if self.is_volume() else f"{d.shape[1]}×{d.shape[0]}"
        self.info.setText(f"{item.name}   {shape}   tipo {np.asarray(item.data).dtype}" + ("   (volume)" if self.is_volume() else ""))

    def fit(self):
        self.g.fitInView(self.g.pix, Qt.KeepAspectRatio)

    def toggle_magnifier(self):
        self.mag_on = not self.mag_on
        self.mag.setVisible(False)
        return self.mag_on

    def _update_mag(self, x, y):
        pm = self.g.pix.pixmap()
        if pm.isNull():
            return
        r = 12
        crop = pm.copy(x - r, y - r, 2 * r, 2 * r).scaled(180, 180, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.mag.setPixmap(crop)
        vp = self.g.mapFromScene(x, y)
        px = vp.x() + 20 if vp.x() + 200 < self.g.viewport().width() else vp.x() - 200
        py = vp.y() + 20 if vp.y() + 200 < self.g.viewport().height() else vp.y() - 200
        self.mag.move(px, py); self.mag.setVisible(True); self.mag.raise_()

    def _hover(self, x, y):
        if self.item is None:
            return
        if self.mag_on:
            self._update_mag(x, y)
        d = self.current_slice()
        if 0 <= y < d.shape[0] and 0 <= x < d.shape[1]:
            v = d[y, x]
            vs = ", ".join(f"{a:.4g}" for a in np.ravel(v))
            self.info.setText(f"{self.item.name}   {d.shape[1]}×{d.shape[0]}   x={x} y={y}   valor={vs}")
