"""
main.py — Application entry point for TolTransform.
"""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat

from gui.main_window import MainWindow


def main() -> None:
    # ANGLE (Qt's DirectX-based OpenGL wrapper) breaks pyqtgraph's GLViewWidget
    # on Windows — force native opengl32.dll instead.
    if sys.platform == "win32":
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)

    # Ensure a 24-bit depth buffer for correct 3D depth sorting on all platforms.
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("TolTransform")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
