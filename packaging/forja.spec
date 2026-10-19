# PyInstaller: compila a Forja numa aplicação para macOS (.app) ou Windows (pasta com Forja.exe).
# Uso (a partir da raiz do projeto):  pyinstaller packaging/forja.spec --noconfirm
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)
from forja import __version__ as VERSION  # noqa: E402

datas = [(os.path.join(ROOT, "forja", "lang"), "forja/lang"),
         (os.path.join(ROOT, "forja", "data"), "forja/data")]
for pkg in ("trimesh", "skimage", "pyqtgraph", "matplotlib", "ezdxf", "shapely"):
    datas += collect_data_files(pkg)

hidden = (collect_submodules("forja") + collect_submodules("skimage") + collect_submodules("trimesh")
          + ["manifold3d", "fast_simplification", "pydicom", "serial", "serial.tools.list_ports", "rtree", "networkx",
             "svgpathtools", "cryptography.fernet", "OpenGL.platform.darwin", "OpenGL.platform.win32",
             "OpenGL.arrays.numpymodule", "scipy.spatial", "scipy.ndimage", "scipy.optimize", "scipy.interpolate"])

excludes = ["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
            "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick3D", "PySide6.QtMultimedia", "PySide6.QtPdf",
            "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PySide6.QtLocation",
            "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors", "PySide6.QtSql", "PySide6.QtTest",
            "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets",
            "PyQt5", "PyQt6", "IPython", "pytest"]

ICON = os.path.join(SPECPATH, "forja.icns" if sys.platform == "darwin" else "forja.ico")

a = Analysis([os.path.join(SPECPATH, "run_forja.py")], pathex=[ROOT], datas=datas, hiddenimports=hidden,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Forja", console=False, icon=ICON,
          upx=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Forja", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(coll, name="Forja.app", icon=ICON, bundle_identifier="pt.forja.app",
                 info_plist={"CFBundleName": "Forja", "CFBundleDisplayName": "Forja",
                             "CFBundleShortVersionString": VERSION, "CFBundleVersion": VERSION,
                             "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "11.0",
                             "NSCameraUsageDescription": "A Forja usa a câmara para aquisição de imagens (metrologia).",
                             "CFBundleDocumentTypes": [{"CFBundleTypeName": "Modelo 3D", "CFBundleTypeRole": "Editor",
                                                        "CFBundleTypeExtensions": ["stl", "obj", "ply", "3mf", "forja"]}]})
