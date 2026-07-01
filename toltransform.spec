# -*- mode: python ; coding: utf-8 -*-
#
# toltransform.spec — PyInstaller build spec for TolTransform
#
# Mac:     pyinstaller toltransform.spec  →  dist/TolTransform.app
# Windows: pyinstaller toltransform.spec  →  dist/TolTransform/TolTransform.exe
#
# One-folder mode (onedir) is used instead of one-file so the app starts
# quickly on repeated launches (no temp-dir extraction overhead).

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

def _icon(path):
    return path if os.path.exists(path) else None

block_cipher = None

# ---------------------------------------------------------------------------
# Data files that are read from disk at runtime (not importable .py modules)
# ---------------------------------------------------------------------------
datas = []
datas += collect_data_files('pyqtgraph')      # GLSL shaders for GLViewWidget
datas += collect_data_files('matplotlib')     # mpl-data: fonts, styles, matplotlibrc
datas += collect_data_files('pytransform3d')  # any bundled rotation-convention data

# ---------------------------------------------------------------------------
# Hidden imports — modules that PyInstaller cannot detect via static analysis
# ---------------------------------------------------------------------------
hiddenimports = [
    # PySide6 OpenGL modules (used by pyqtgraph.opengl.GLViewWidget)
    'PySide6.QtOpenGL',
    'PySide6.QtOpenGLWidgets',
    # pyqtgraph OpenGL package (dynamic imports inside pyqtgraph)
    'pyqtgraph.opengl',
    # matplotlib Qt backend used by FigureCanvasQTAgg in results viewer / point-pair panel
    'matplotlib.backends.backend_qtagg',
    # pydantic v2 internals loaded via __init_subclass__ / plugin machinery
    'pydantic.deprecated.class_validators',
    'pydantic.deprecated.config',
    'pydantic.deprecated.decorator',
    # scipy internal — commonly missed
    'scipy._lib.messagestream',
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['packaging/rthook_opengl.py'],
    excludes=[
        'tkinter',   # not used; saves ~10 MB
        'PyQt5',     # exclude Qt5 if present in env — we use PySide6
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------------------------------------------------------------------------
# EXE (entry-point stub) — shared by both platforms
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TolTransform',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,            # no terminal window on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,         # None = native arch; set 'universal2' for fat Mac binary
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon('assets/icon.ico' if sys.platform == 'win32' else 'assets/icon.icns'),
)

# ---------------------------------------------------------------------------
# COLLECT — one-folder distribution bundle
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TolTransform',
)

# ---------------------------------------------------------------------------
# BUNDLE — Mac .app wrapper (ignored on Windows)
# ---------------------------------------------------------------------------
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='TolTransform.app',
        icon=_icon('assets/icon.icns'),
        bundle_identifier='com.joeylitjens.toltransform',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,  # allows dark mode
        },
    )
