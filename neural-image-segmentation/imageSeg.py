# This is the main script to start the application for our Neural Image Segmentation project
import warnings
import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui, QtCharts
from PIL import Image, ImageQt
from UI.ut import Ui_MainWindow
from unet.unet_utils import gamma_correction, unet_predict
from postImgProc.utils import *
from postImgProc.alg import *
import pyqtgraph as pg
from skimage.morphology import flood_fill

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

THRESHOLD0 = 40
THRESHOLD1 = 75
THRESHOLD2 = 100
THRESHOLD3 = 150


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
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.installEventFilter(self)

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
        self.pixmap_item = self.scene.addPixmap(pix)
        self.ui.graphicsView2.setPhotoByScnen(self.scene)

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
        segmentation_analysis = SegmentationAnalysis()
        segmented_axons = segmentation_analysis.run(self.file_name, self.img)
        img_annotated = self.img.copy()
        # print(len(img_annotated), len(img_annotated[0]))
        height = len(img_annotated)
        width = len(img_annotated[0])
        # print(img_annotated[0][0])
        for i in range(height):
            for j in range(width):
                if segmentation_analysis.cell_boundary_mask[i][j] != 0:
                    img_annotated[max(i - 1, 0): min(i + 1, height - 1),
                    max(j - 1, 0): min(j + 1, width - 1)] = [255, 255, 0]
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
        axon_set = [[0]*4 for i in range(5)]
        rangeMax = 0
        # length_range = ["20-50", "50-100", "100-150", "150-200", "200+"]
        length_range = [20, THRESHOLD0, THRESHOLD1, THRESHOLD2, THRESHOLD3]
        for i in range(len(segmented_axons)):
            length = pixel_to_length(cal_dist(segmented_axons[i]))
            orientation = getOrientation(segmented_axons[i][0], segmented_axons[i][-1])
            total_length += length
            index = 0
            if length <= THRESHOLD0:
                index = 0
            elif length <= THRESHOLD1:
                index = 1
            elif length <= THRESHOLD2:
                index = 2
            elif length <= THRESHOLD3:
                index = 3
            else:
                index = 4
            length_dist[index] += 1
            if orientation <= 45 or orientation >= 315:
                axon_set[index][0] += 1
            elif orientation <= 135:
                axon_set[index][1] += 1
            elif orientation <= 225:
                axon_set[index][2] += 1
            else:
                axon_set[index][3] += 1
        length_range = max(max(x) for x in axon_set)
        average_length = total_length / len(segmented_axons)
        set0 = QtCharts.QBarSet("20-" + str(THRESHOLD0))
        set1 = QtCharts.QBarSet(str(THRESHOLD0) + "-" + str(THRESHOLD1))
        set2 = QtCharts.QBarSet(str(THRESHOLD1) + "-" + str(THRESHOLD2))
        set3 = QtCharts.QBarSet(str(THRESHOLD2) + "-" + str(THRESHOLD3))
        set4 = QtCharts.QBarSet(str(THRESHOLD3))
        set0.append(axon_set[0])
        set1.append(axon_set[1])
        set2.append(axon_set[2])
        set3.append(axon_set[3])
        set4.append(axon_set[4])
        self.ui.info.setText(_translate("MainWindow",
                                        "Information:\n"
                                        "Cells count: " + str(segmentation_analysis.nr_cell) + "\n" +
                                        "Axons count: " + str(len(segmented_axons)) + "\n" +
                                        "Average axon length: " + "{:.2f}".format(average_length) + "μm\n"))
        series = QtCharts.QBarSeries()
        series.append(set0)
        series.append(set1)
        series.append(set2)
        series.append(set3)
        series.append(set4)
        axonLengthWidget = QtCharts.QChart()
        axonLengthWidget.addSeries(series)
        axonLengthWidget.setTitle("Axon Length Distribution")
        # axonLengthWidget.setLabel('left', 'Count', color="b", size="12pt")
        # axonLengthWidget.setLabel('bottom', 'Length', color="b", size="12pt")
        categories = ["N", "E", "S", "W"]
        axisX = QtCharts.QBarCategoryAxis()
        axisX.append(categories)
        axonLengthWidget.addAxis(axisX, QtCore.Qt.AlignmentFlag.AlignBottom)

        axisY = QtCharts.QValueAxis()
        axisY.setRange(0, length_range)
        axisY.setLabelFormat("%d")
        axonLengthWidget.addAxis(axisY, QtCore.Qt.AlignmentFlag.AlignLeft)

        axonLengthWidget.legend().setVisible(True)
        axonLengthWidget.legend().adjustSize()
        axonLengthWidget.legend().setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)

        # chartView.setRenderHint(QtGui.QPainter.RenderHints.Antialiasing)
        axonLengthWidget.resize(100, 75)
        axonLengthWidget.setBackgroundBrush(QtGui.QColor(140, 255, 255, 127))
        # axonLengthWidget.setBackground((255, 255, 255, 0))
        # axonLengthWidget.plot(length_range, length_dist, symbol='o', symbolPen=None, symbolSize=10,
        #                  symbolBrush=(100, 100, 255, 255))
        chartView = QtCharts.QChartView(axonLengthWidget)
        chartView.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.ui.infoboxLayout.addWidget(chartView)
        axonOrientationWidget = QtCharts.QPolarChart
        # self.display_img(im)

    def eventFilter(self, source, event):
        if source is self.scene and event.type() == QtCore.QEvent.Type.GraphicsSceneMouseDoubleClick:
            spf = event.scenePos()
            lpf = self.pixmap_item.mapFromScene(spf)
            brf = self.pixmap_item.boundingRect()
            if brf.contains(lpf):
                lp = lpf.toPoint()
                print(lp)
                # if self.seg_image is not None:
                #     flood_fill(self.seg_image, (lp.x()-1,lp.y()-1,), np.array([255,0,0]))
                #     print(self.seg_image)
        return super(ImgSeg, self).eventFilter(source, event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = ImgSeg()
    myapp.show()
    sys.exit(app.exec())
