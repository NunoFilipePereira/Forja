"""Menu Dispositivos: câmara em direto, projetor, aquisição automática, porta série, mesa rotativa."""
import time
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QPixmap, QGuiApplication
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QWidget,
                               QPlainTextEdit, QLineEdit, QComboBox, QApplication, QFormLayout)
from ..core import devices as dv
from .viewimage import array_to_qimage


class ProjectorWindow(QWidget):
    """Janela sem margens em ecrã inteiro num monitor escolhido (projetor)."""

    def __init__(self, screen_index=0):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet("background:black")
        self.lbl = QLabel(self); self.lbl.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.lbl)
        screens = QGuiApplication.screens()
        scr = screens[min(screen_index, len(screens) - 1)]
        self.setGeometry(scr.geometry())

    def show_image(self, a):
        from ..core.imageops import to_u8
        pm = QPixmap.fromImage(array_to_qimage(to_u8(np.asarray(a))))
        self.lbl.setPixmap(pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        QApplication.processEvents()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()


class CameraDialog(QDialog):
    def __init__(self, win, index=0):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Câmara")
        self.cam = dv.Camera(index)
        lay = QVBoxLayout(self)
        self.view = QLabel("…"); self.view.setMinimumSize(640, 400); self.view.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.view)
        row = QHBoxLayout(); lay.addLayout(row)
        self.exp = QDoubleSpinBox(); self.exp.setRange(-20, 10000); self.exp.setValue(-6); self.exp.setPrefix("exposição ")
        self.exp.valueChanged.connect(lambda v: self.cam.set_exposure(v))
        b1 = QPushButton("Foto"); b1.clicked.connect(self.single)
        self.nshots = QSpinBox(); self.nshots.setRange(2, 1000); self.nshots.setValue(5); self.nshots.setPrefix("× ")
        b2 = QPushButton("Várias fotos"); b2.clicked.connect(self.multi)
        self.rec = QPushButton("Gravar"); self.rec.setCheckable(True)
        for wdg in (self.exp, b1, self.nshots, b2, self.rec):
            row.addWidget(wdg)
        self.frames = []
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(60)
        self.last = None

    def tick(self):
        try:
            f = self.cam.grab(0)
        except Exception:
            return
        self.last = f
        if self.rec.isChecked():
            self.frames.append(f)
            self.rec.setText(f"A gravar ({len(self.frames)})")
        elif self.frames:
            for k, fr in enumerate(self.frames):
                self.win.add_image(f"gravação {k + 1:04d}", fr, show=k == len(self.frames) - 1)
            self.win.log(f"Gravação: {len(self.frames)} imagens")
            self.frames = []; self.rec.setText("Gravar")
        pm = QPixmap.fromImage(array_to_qimage(f))
        self.view.setPixmap(pm.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def single(self):
        if self.last is not None:
            self.win.add_image(f"foto {time.strftime('%H:%M:%S')}", self.cam.grab())

    def multi(self):
        for k in range(self.nshots.value()):
            self.win.add_image(f"foto {k + 1}", self.cam.grab(), show=k == self.nshots.value() - 1)

    def done(self, r):
        self.timer.stop(); self.cam.close()
        super().done(r)


class SerialDialog(QDialog):
    """Consola série com modelos de comando para mesas/controladores."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Porta série / controlador")
        self.resize(640, 460)
        self.dev = None
        st = QSettings("Forja", "Forja")
        lay = QVBoxLayout(self)
        form = QFormLayout(); lay.addLayout(form)
        self.port = QComboBox(); self.port.setEditable(True)
        try:
            for dev, desc in dv.list_ports():
                self.port.addItem(f"{dev}", dev)
        except Exception:
            pass
        self.port.addItem("loop://  (teste sem equipamento)", "loop://")
        self.baud = QComboBox(); self.baud.addItems(["9600", "19200", "38400", "57600", "115200"]); self.baud.setEditable(True)
        self.eol = QComboBox(); self.eol.addItems(["\\r\\n", "\\r", "\\n"])
        self.tpl_move = QLineEdit(st.value("serial/move", "MOVE {pos:.3f}"))
        self.tpl_rot = QLineEdit(st.value("serial/rot", "ROT {angle:.2f}"))
        form.addRow("Porta", self.port); form.addRow("Velocidade (baud)", self.baud); form.addRow("Fim de linha", self.eol)
        form.addRow("Modelo: mover mesa linear", self.tpl_move); form.addRow("Modelo: rodar mesa rotativa", self.tpl_rot)
        row = QHBoxLayout(); lay.addLayout(row)
        self.bopen = QPushButton("Ligar"); self.bopen.clicked.connect(self.toggle)
        row.addWidget(self.bopen)
        self.cmd = QLineEdit(); self.cmd.setPlaceholderText("Comando…"); self.cmd.returnPressed.connect(self.send)
        row.addWidget(self.cmd, 1)
        bs = QPushButton("Enviar"); bs.clicked.connect(self.send); row.addWidget(bs)
        self.out = QPlainTextEdit(); self.out.setReadOnly(True); lay.addWidget(self.out)

    def toggle(self):
        if self.dev:
            self.dev.close(); self.dev = None; self.bopen.setText("Ligar"); self.out.appendPlainText("— desligado —"); return
        try:
            self.dev = dv.SerialDevice(self.port.currentData() or self.port.currentText().split()[0], int(self.baud.currentText()),
                                       eol=self.eol.currentText())
            self.bopen.setText("Desligar"); self.out.appendPlainText(f"— ligado a {self.port.currentText()} —")
            self.win.serial_dev = self.dev
            st = QSettings("Forja", "Forja"); st.setValue("serial/move", self.tpl_move.text()); st.setValue("serial/rot", self.tpl_rot.text())
            self.win.serial_templates = {"move": self.tpl_move.text(), "rot": self.tpl_rot.text()}
        except Exception as e:
            self.out.appendPlainText(f"ERRO: {e}")

    def send(self):
        if not self.dev:
            self.out.appendPlainText("Ligue primeiro."); return
        c = self.cmd.text()
        try:
            r = self.dev.send(c)
        except Exception as e:
            r = f"ERRO: {e}"
        self.out.appendPlainText(f"> {c}\n< {r}")
        self.cmd.clear()


class DeviceActions:
    def menus_device(self):
        return [("&Dispositivos", [
            ("Câmara em direto…", self.dv_camera),
            ("Projetor: mostrar imagens selecionadas…", self.dv_projector),
            ("Aquisição automática de franjas (projetar + fotografar)…", self.dv_auto_fringe),
            None,
            ("Porta série / controlador…", self.dv_serial),
            ("Varrimento em mesa rotativa (rodar + fotografar)…", self.dv_turntable_scan),
            ("Varrimento em mesa linear (mover + fotografar)…", self.dv_linear_scan),
        ])]

    def dv_camera(self):
        v = self.ask("Câmara", [("i", "Índice da câmara", 0, (0, 10))])
        if not v:
            return
        try:
            CameraDialog(self, v["i"]).exec()
        except Exception as e:
            self.error(e)

    def _screens(self):
        return [f"{k}: {s.name()} {s.geometry().width()}×{s.geometry().height()}" for k, s in enumerate(QGuiApplication.screens())]

    def dv_projector(self):
        it = self.sel_images(1)
        if not it:
            return
        sc = self._screens()
        v = self.ask("Projetor", [("s", "Ecrã", sc[-1], sc), ("t", "Tempo por imagem (s)", 1.0, (0.05, 600))],
                     note="Esc fecha a janela do projetor.")
        if not v:
            return
        pw = ProjectorWindow(int(v["s"].split(":")[0])); pw.showFullScreen()
        for i in it:
            if not pw.isVisible():
                break
            pw.show_image(i.data)
            t0 = time.time()
            while time.time() - t0 < v["t"] and pw.isVisible():
                QApplication.processEvents(); time.sleep(0.01)
        pw.close()

    def dv_auto_fringe(self):
        it = self.sel_images(1)
        if not it:
            return self.error("Gere e selecione os padrões (Metrologia ▸ Gerar padrões) a projetar.")
        sc = self._screens()
        v = self.ask("Aquisição automática", [("s", "Ecrã do projetor", sc[-1], sc), ("cam", "Câmara", 0, (0, 10)),
                                              ("d", "Espera antes de cada foto (s)", 0.4, (0.0, 10))],
                     note=f"{len(it)} padrões serão projetados por ordem e fotografados um a um.")
        if not v:
            return
        try:
            cam = dv.Camera(v["cam"])
        except Exception as e:
            return self.error(e)
        pw = ProjectorWindow(int(v["s"].split(":")[0])); pw.showFullScreen()
        shots = []
        try:
            for i in it:
                pw.show_image(i.data)
                t0 = time.time()
                while time.time() - t0 < v["d"]:
                    QApplication.processEvents(); time.sleep(0.01)
                shots.append((i.name, cam.grab(flush=3)))
        finally:
            pw.close(); cam.close()
        for k, (n, a) in enumerate(shots):
            self.add_image(f"captura de {n}", a, show=k == len(shots) - 1)
        self.log(f"Aquisição: {len(shots)} imagens. Selecione-as e use Metrologia ▸ Fase embrulhada.")

    def dv_serial(self):
        SerialDialog(self).exec()

    def _serial_ready(self):
        if not getattr(self, "serial_dev", None):
            self.error("Ligue primeiro o controlador em Dispositivos ▸ Porta série.")
            return False
        return True

    def dv_turntable_scan(self):
        if not self._serial_ready():
            return
        v = self.ask("Mesa rotativa", [("n", "Número de vistas", 12, (2, 3600)), ("total", "Rotação total (°)", 360.0),
                                       ("cam", "Câmara", 0, (0, 10)), ("d", "Espera após rodar (s)", 1.0, (0, 60))])
        if not v:
            return
        self._scan("rot", [k * v["total"] / v["n"] for k in range(v["n"])], v["cam"], v["d"], "angle", "vista")

    def dv_linear_scan(self):
        if not self._serial_ready():
            return
        v = self.ask("Mesa linear", [("a", "Posição inicial (mm)", 0.0), ("b", "Posição final (mm)", 10.0), ("n", "Passos", 11, (2, 10000)),
                                     ("cam", "Câmara", 0, (0, 10)), ("d", "Espera após mover (s)", 0.5, (0, 60))])
        if not v:
            return
        self._scan("move", list(np.linspace(v["a"], v["b"], v["n"])), v["cam"], v["d"], "pos", "posição")

    def _scan(self, kind, values, cam_i, delay, key, label):
        tpl = getattr(self, "serial_templates", {}).get(kind, "MOVE {pos}" if kind == "move" else "ROT {angle}")
        try:
            cam = dv.Camera(cam_i)
        except Exception as e:
            return self.error(e)
        try:
            for k, val in enumerate(values):
                cmd = dv.fill(tpl, **{key: val})
                resp = self.serial_dev.send(cmd)
                self.log(f"> {cmd}  < {resp}")
                t0 = time.time()
                while time.time() - t0 < delay:
                    QApplication.processEvents(); time.sleep(0.01)
                self.add_image(f"{label} {val:.3f}", cam.grab(flush=3), show=k == len(values) - 1)
        finally:
            cam.close()
