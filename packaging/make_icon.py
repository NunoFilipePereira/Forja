"""Gera o ícone da Forja 3D (packaging/forja.png, .ico e .icns) — uma bigorna estilizada."""
import os, sys
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor, QPainterPath, QLinearGradient, QFont
from PySide6.QtCore import Qt, QRectF, QPointF

app = QGuiApplication(sys.argv)
S = 1024
img = QImage(S, S, QImage.Format_ARGB32); img.fill(Qt.transparent)
p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
bg = QPainterPath(); bg.addRoundedRect(QRectF(40, 40, S - 80, S - 80), 200, 200)
g = QLinearGradient(0, 0, 0, S); g.setColorAt(0, QColor("#E0702E")); g.setColorAt(1, QColor("#8E3A12"))
p.fillPath(bg, g)
# bigorna
a = QPainterPath()
a.moveTo(200, 380); a.lineTo(820, 380); a.lineTo(820, 440); a.cubicTo(720, 450, 660, 480, 640, 560)
a.lineTo(640, 640); a.lineTo(720, 700); a.lineTo(720, 780); a.lineTo(300, 780); a.lineTo(300, 700); a.lineTo(380, 640)
a.lineTo(380, 560); a.cubicTo(360, 500, 290, 470, 200, 440); a.closeSubpath()
p.fillPath(a, QColor("#2B2B2E"))
p.fillRect(QRectF(200, 380, 620, 22), QColor("#5A5A60"))
# faísca
p.setPen(Qt.NoPen); p.setBrush(QColor("#FFD54A"))
for x, y, r in ((560, 250, 34), (650, 200, 20), (470, 205, 16), (720, 290, 14)):
    p.drawEllipse(QPointF(x, y), r, r)
# "3D"
f = QFont("Helvetica", 1); f.setPixelSize(170); f.setBold(True); p.setFont(f)
p.setPen(QColor("#FFF3E0")); p.drawText(QRectF(560, 800, 380, 180), Qt.AlignRight | Qt.AlignVCenter, "3D")
p.end()
out = os.path.join(os.path.dirname(__file__), "forja.png")
img.save(out)
from PIL import Image
im = Image.open(out)
im.save(out[:-4] + ".ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
im.save(out[:-4] + ".icns")
print("ok", out)
