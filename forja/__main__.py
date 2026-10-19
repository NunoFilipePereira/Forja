import datetime
import os
import sys
import traceback

if getattr(sys, "frozen", False):  # aplicação instalada: a pasta do programa pode não aceitar escrita
    LOG = os.path.join(os.path.expanduser("~"), ".forja", "erros.log")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
else:
    LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "erros.log")


def _install_error_log():
    """Qualquer erro não tratado fica em erros.log e aparece numa mensagem, em vez de falhar em silêncio."""
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        with open(LOG, "a") as f:
            f.write(f"\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===\n{text}")
        sys.__stderr__.write(text)
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Forja — erro", f"{exc_type.__name__}: {exc}\n\n(registado em erros.log)")
        except Exception:
            pass
    sys.excepthook = hook


def _selftest(app, w):
    try:
        from . import i18n
        from .core import conv3d, vision
        import cv2
        w.demo_model()
        m = w.doc.objects[0].mesh
        assert m.is_watertight and len(w.doc.objects) == 1
        conv3d.block(m.copy())
        assert i18n.available() and os.path.exists(os.path.join(vision.DATA, "face_detection_yunet.onnx"))
        print("FORJA_SELFTEST_OK", cv2.__version__, flush=True)
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    app.exit(code)


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat(); fmt.setSamples(4); fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    app.setApplicationName("Forja")
    _install_error_log()
    from PySide6.QtCore import QSettings
    from . import i18n
    i18n.load(QSettings("Forja", "Forja").value("lang", "pt"))
    from .ui.mainwindow import MainWindow
    w = MainWindow()
    w.show()
    if os.environ.get("FORJA_SELFTEST"):  # usado na compilação: arranca, faz um modelo e sai
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, lambda: _selftest(app, w))
    for p in sys.argv[1:]:
        w.open_path(p)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
