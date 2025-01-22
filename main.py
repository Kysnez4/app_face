import sys
from PyQt5.QtWidgets import QApplication
from src.gui import FaceRecognitionApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())