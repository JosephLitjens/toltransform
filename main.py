"""
main.py — Application entry point for TolTransform.
"""

import os
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat

from gui.main_window import MainWindow


def main() -> None:
    # On Windows: force native OpenGL (opengl32.dll) before QApplication is
    # created.  ANGLE (Qt's DirectX wrapper) breaks pyqtgraph's GLViewWidget.
    # The frozen-app runtime hook sets this even earlier (before PySide6 loads),
    # but setting it here too covers the non-frozen development case.
    if sys.platform == "win32":
        os.environ.setdefault("QT_OPENGL", "desktop")

    # Request a 24-bit depth buffer and a Compatibility profile — pyqtgraph uses
    # OpenGL features (immediate-mode calls, fixed-function pipeline) that are
    # absent from Core profile contexts.
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("TolTransform")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
