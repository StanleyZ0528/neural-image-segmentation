# This is the main script to start the application for our Neural Image Segmentation project
import warnings
import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PIL import Image, ImageQt
from UI.ut import Ui_MainWindow
from unet.unet_utils import gamma_correction, unet_predict
from postImgProc.utils import *
from postImgProc.alg import *
import pyqtgraph as pg

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImgSeg(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow(self)
        self.ui.toolButton.clicked.connect(self.tool_button_callback)
        self.ui.pushButton.clicked.connect(self.process_button_callback)
        self.ui.preButton.clicked.connect(self.pre_button_callback)
        self.ui.analyzeButton.clicked.connect(self.analyze_button_callback)
        self.ui.clearButton.clicked.connect(self.clear_button_callback)
        self.ui.saveButton.clicked.connect(self.save_button_callback)
        self.input_file_name = ""
        self.output_file_name = ""
        self.img = None
        self.gamma_image = None
        self.seg_image = None
        self.show()

    def save_button_callback(self):
        self.output_file_name = QtWidgets.QFileDialog.getSaveFileName(self)[0]
        if self.output_file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            # save the output image...
            pass
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Save Failed")
            msg.show()

    def display_img(self, image):
        img = Image.fromarray(image, mode='RGB')
        qt_img = ImageQt.ImageQt(img)
        pix = QtGui.QPixmap.fromImage(qt_img)
        item = QtWidgets.QGraphicsPixmapItem(pix)
        scene = QtWidgets.QGraphicsScene(self)
        scene.addItem(item)
        self.ui.graphicsView2.setPhotoByScnen(scene)

    def tool_button_callback(self):
        self.input_file_name = QtWidgets.QFileDialog.getOpenFileName(self)[0]
        self.ui.textBrowser.setText(self.input_file_name)
        if self.input_file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            self.img = readImg(self.input_file_name)
            pix = QtGui.QPixmap(self.input_file_name)
            self.ui.graphicsView1.setPhoto(pix)
            self.ui.graphicsView2.setPhoto(pix)
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Please check the input file")
            msg.show()

    def pre_button_callback(self):
        if self.img is not None:
            # conduct the process
            if self.gamma_image is None:
                self.gamma_image = gamma_correction(self.img)

            # display the result
            self.display_img(self.gamma_image)
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Please check the input file")
            msg.show()

    def process_button_callback(self):
        if self.gamma_image is not None:
            # conduct the process
            if self.seg_image is None:
                self.seg_image = unet_predict(self.gamma_image)

            # display the result
            self.display_img(self.seg_image.astype(np.uint8))
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Warning: Gamma Correction not applied!")
            msg.show()

    def clear_button_callback(self):
        self.ui.textBrowser.setText("")
        self.ui.graphicsView1.setPhotoByScnen(None)
        self.ui.graphicsView2.setPhotoByScnen(None)
        self.img = None
        self.gamma_image = None
        self.seg_image = None

    def analyze_button_callback(self):
        labeled_axon, labeled_cell, nr_cell = separate_axon_and_cell(self.img)
        axons_to_cells = get_touching_dict(labeled_axon, labeled_cell)
        fil = filter_axon(self.img)
        info_list = analyze_axons(fil, labeled_axon, axons_to_cells)
        segmented_axons = getSegmentedAxons(fil, info_list, nr_cell)
        img_annotated = self.img.copy()
        # print(len(img_annotated), len(img_annotated[0]))
        height = len(img_annotated)
        width = len(img_annotated[0])
        # print(img_annotated[0][0])
        for axon in segmented_axons:
            for pixel in axon:
                # print(pixel)
                img_annotated[max(pixel[0] - 3, 0): min(pixel[0] + 3, height - 1),
                max(pixel[1] - 3, 0): min(pixel[1] + 3, width - 1)] = [255, 0, 0]
        # print(img_annotated)
        # for item in info_list:
        #     touch_points = item["touch_points"]
        #     intersect_points = item["intersect_points"]
        #     end_points = item["end_points"]
        #     for coord in touch_points:
        #         img_annotated[coord[0]][coord[1]] = [255, 0, 0]
        #     for coord in intersect_points:
        #         if not type(coord) is tuple:
        #             continue
        #         img_annotated[coord[0]][coord[1]] = [0, 0, 255]
        #     for coord in end_points:
        #         if not type(coord) is tuple:
        #             continue
        #         img_annotated[coord[0]][coord[1]] = [0, 255, 0]
        # display the result
        im = Image.fromarray(img_annotated)
        im.save("result/result.png", format="png")
        pix = QtGui.QPixmap("result/result.png")
        self.ui.graphicsView2.setPhoto(pix)

        _translate = QtCore.QCoreApplication.translate
        length_dist = [0, 0, 0, 0, 0]
        total_length = 0
        # length_range = ["20-50", "50-100", "100-150", "150-200", "200+"]
        length_range = [0, 1, 2, 3, 4]
        for i in range(len(segmented_axons)):
            length = pixel_to_length(cal_dist(segmented_axons[i]))
            total_length += length
            if length <= 50:
                length_dist[0] += 1
            elif length <= 100:
                length_dist[1] += 1
            elif length <= 150:
                length_dist[2] += 1
            elif length <= 200:
                length_dist[3] += 1
            else:
                length_dist[4] += 1
        average_length = total_length / len(segmented_axons)
        self.ui.info.setText(_translate("MainWindow",
                                        "Information:\n"
                                        "Cells count: " + str(nr_cell) + "\n" +
                                        "Axons count: " + str(len(segmented_axons)) + "\n" +
                                        "Average axon length: " + "{:.2f}".format(average_length) + "μm\n"))
        graphWidget = pg.PlotWidget()
        graphWidget.setBackground((255, 255, 255, 0))
        graphWidget.plot(length_range, length_dist, symbol='o', symbolPen=None, symbolSize=10,
                         symbolBrush=(100, 100, 255, 255))
        self.ui.infoboxLayout.addWidget(graphWidget)
        # self.display_img(im)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = ImgSeg()
    sys.exit(app.exec())
