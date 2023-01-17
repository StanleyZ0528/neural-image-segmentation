# This is the main script to start the application for our Neural Image Segmentation project
import warnings
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PIL import Image, ImageQt
from UI.ut import Ui_MainWindow
from unet.unet_utils import gamma_correction


class ImgSeg(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.toolButton.clicked.connect(self.tool_button_callback)
        self.ui.pushButton.clicked.connect(self.process_button_callback)
        self.ui.clearButton.clicked.connect(self.clear_button_callback)
        self.file_name = ""
        self.show()

    def tool_button_callback(self):
        self.file_name = QtWidgets.QFileDialog.getOpenFileName(self)[0]
        self.ui.textBrowser.setText(self.file_name)
        if self.file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            pix = QtGui.QPixmap(self.file_name)
            item = QtWidgets.QGraphicsPixmapItem(pix)
            scene = QtWidgets.QGraphicsScene(self)
            scene.addItem(item)
            self.ui.graphicsView.setScene(scene)
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Please check the input file")
            msg.show()

    def process_button_callback(self):
        if self.file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            # conduct the process
            gamma_image = gamma_correction(self.file_name)
            result = gamma_image
            # more process ...

            # display the result
            img = Image.fromarray(result, mode='RGB')
            qt_img = ImageQt.ImageQt(img)
            pix = QtGui.QPixmap.fromImage(qt_img)
            item = QtWidgets.QGraphicsPixmapItem(pix)
            scene = QtWidgets.QGraphicsScene(self)
            scene.addItem(item)
            self.ui.graphicsView.setScene(scene)
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Please check the input file")
            msg.show()

    def clear_button_callback(self):
        self.ui.textBrowser.setText("")
        scene = QtWidgets.QGraphicsScene(self)
        self.ui.graphicsView.setScene(scene)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = ImgSeg()
    sys.exit(app.exec_())
