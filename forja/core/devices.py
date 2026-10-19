"""Dispositivos: porta série genérica (mesas lineares/rotativas, lasers, controladores) e câmara."""
import time
import numpy as np


def list_ports():
    from serial.tools import list_ports as lp
    return [(p.device, p.description) for p in lp.comports()]


class SerialDevice:
    """Ligação série com comandos em texto. Aceita também URLs do pyserial (p.ex. 'loop://' para testes)."""

    def __init__(self, port, baud=9600, timeout=1.0, eol="\r\n"):
        import serial
        self.ser = serial.serial_for_url(port, baudrate=baud, timeout=timeout)
        self.eol = eol.encode().decode("unicode_escape")

    def send(self, cmd, read=True, wait=0.05):
        self.ser.reset_input_buffer()
        self.ser.write((cmd + self.eol).encode())
        self.ser.flush()
        if not read:
            return ""
        time.sleep(wait)
        data = b""
        t0 = time.time()
        while time.time() - t0 < self.ser.timeout:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if not chunk:
                break
            data += chunk
            if data.endswith(self.eol.encode()):
                break
        return data.decode(errors="replace").strip()

    def close(self):
        self.ser.close()


def fill(template, **kw):
    """Preenche um modelo de comando, p.ex. 'MOVE {pos:.3f}' ou '1PA{angle}'."""
    return template.format(**kw)


class Camera:
    def __init__(self, index=0):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível abrir a câmara (verifique Definições do Sistema ▸ Privacidade ▸ Câmara).")

    def set_exposure(self, value):
        return self.cap.set(self.cv2.CAP_PROP_EXPOSURE, float(value))

    def set_resolution(self, w, h):
        self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, w); self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, h)

    def grab(self, flush=2):
        for _ in range(flush):
            self.cap.grab()
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("A câmara não devolveu imagem.")
        return self.cv2.cvtColor(f, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()
