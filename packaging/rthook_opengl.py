# PyInstaller runtime hook — runs before any Python imports (including PySide6).
# Forces native desktop OpenGL on Windows so Qt does not use ANGLE (its
# DirectX-based OpenGL wrapper), which breaks pyqtgraph's GLViewWidget.
import sys
import os

if sys.platform == "win32":
    os.environ["QT_OPENGL"] = "software"
