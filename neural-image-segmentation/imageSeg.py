# This is the main script to start the application for our Neural Image Segmentation project
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import warnings
import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui, QtCharts
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PIL import Image, ImageQt
from UI.ut import Ui_MainWindow
from unet.unet_utils import gamma_correction, unet_predict
from postImgProc.utils import *
from postImgProc.alg import *
import pyqtgraph as pg
import pandas as pd
import xlsxwriter
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

THRESHOLD0 = 40
THRESHOLD1 = 75
THRESHOLD2 = 100
THRESHOLD3 = 150

TYPICALCELLAREA = 1200  # Approximation from 20*20*3.14
CELLTHRESHOLD= [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ImgSeg(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow(self)
        self.ui.toolButton.clicked.connect(self.tool_button_callback)
        self.ui.pushButton.clicked.connect(self.process_button_callback)
        self.ui.preButton.clicked.connect(self.pre_button_callback)
        self.ui.analyzeButton.clicked.connect(self.analyze_button_callback)
        self.ui.clearButton.clicked.connect(self.clear_button_callback)
        # self.ui.saveButton.clicked.connect(self.save_button_callback)
        self.ui.setFilterButton.clicked.connect(self.set_filter_callback)
        self.ui.getCellIndexButton.clicked.connect(self.get_cell_index_callback)
        self.ui.getAxonIndexButton.clicked.connect(self.get_axon_index_callback)
        # self.ui.exportButton.clicked.connect(self.export_button_callback)
        self.input_file_name = ""
        self.output_file_name = ""
        self.img = None
        self.gamma_image = None
        self.seg_image = None
        self.img_annotated = None
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.installEventFilter(self)
        self.cell_filter_size = 500
        self.cell_index = None
        self.axon_index = None
        self.chartView = None
        self.areaChartView = None
        self.segmentation_analysis = ""
        self.segmented_axons = []
        self.pixmap_item = None

        self.movie = QtGui.QMovie("UI/Spin.gif")
        self.ui.giflabel.setMovie(self.movie)

    def set_filter_callback(self):
        try:
            self.cell_filter_size = int(self.ui.cellFilterTextbox.text())
            self.ui.cellFilterTextbox.setPlaceholderText(self.ui.cellFilterTextbox.text())
            self.ui.cellFilterTextbox.setText("")
        except ValueError:
            self.ui.cellFilterTextbox.setText("")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Input cell filter size is not an integer")
            msg.show()

    def get_cell_index_callback(self):
        try:
            self.cell_index = int(self.ui.cellIndexTextbox.text())
            self.ui.cellIndexTextbox.setPlaceholderText(self.ui.cellIndexTextbox.text())
            self.ui.cellIndexTextbox.setText("")
        except ValueError:
            self.ui.cellIndexTextbox.setText("")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Input cell index is not an integer")
            msg.show()
            return
        if self.cell_index <= 0 or self.cell_index > self.segmentation_analysis.nr_cell:
            self.ui.cellIndexTextbox.setText("")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Input cell index out of range")
            msg.show()
            return

        img_annotated = self.img_annotated.copy()
        # print(len(img_annotated), len(img_annotated[0]))
        height = len(img_annotated)
        width = len(img_annotated[0])
        found = False
        # display the result
        for p in self.segmentation_analysis.cell_pixels[self.cell_index - 1]:
            i = p[0]
            j = p[1]
            found = True
            img_annotated[i][j] = [140, 255, 255]
            if self.segmentation_analysis.cell_boundary_mask[i][j] != 0:
                img_annotated[max(i - 1, 0): min(i + 1, height - 1),
                max(j - 1, 0): min(j + 1, width - 1)] = [255, 255, 0]
        if not found:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Input cell index is filtered out due to size limit")
            msg.show()
            return

        # Clear the previous cell information
        _translate = QtCore.QCoreApplication.translate
        self.ui.cellInfo.setText(_translate("MainWindow", "Cell Information:"))
        self.display_img(img_annotated)
        cell_index_str = ""
        length = len(self.segmentation_analysis.cell_axons_map[self.cell_index - 1])
        for i in self.segmentation_analysis.cell_axons_map[self.cell_index - 1]:
            cell_index_str += " " + str(i)
            if self.segmentation_analysis.show_orientation[i]:
                cell_index_str += " (" +\
                                  degree_to_ori(getOrientation(self.segmented_axons[i][0], self.segmented_axons[i][-1])) + ")"
            else:
                cell_index_str += " (" +\
                                  degree_to_ori(getOrientation(self.segmented_axons[i][0], self.segmented_axons[i][-1]))\
                                  + "," +\
                                  degree_to_ori(getOrientation(self.segmented_axons[i][-1], self.segmented_axons[i][0])) + ")"
        self.ui.cellInfo.setText(_translate(
            "MainWindow",
            "Cell Cluster Information:\n"
            "Cell Cluster Index: " + str(
                self.cell_index) + "\nCell Area: " + "{:.2f}".format(
                np.isclose(self.cell_index,
                           self.segmentation_analysis.labeled_cell).sum() / 2.22 / 2.22) +
            "µm^2\nEstimated Cell Count: " +
            "{:.0f}".format((self.segmentation_analysis.cell_area_map[self.cell_index] / TYPICALCELLAREA).round()) +
            "\nConnected Axons count: " + str(length) + "\n"
            "Connected Axon Indexes: " + cell_index_str + "\n"))

    def get_axon_index_callback(self):
        try:
            self.axon_index = int(self.ui.axonIndexTextbox.text())
            self.ui.axonIndexTextbox.setPlaceholderText(self.ui.axonIndexTextbox.text())
            self.ui.axonIndexTextbox.setText("")
        except ValueError:
            self.ui.axonIndexTextbox.setText("")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Input cell index is not an integer")
            msg.show()
            return
        if self.axon_index < 0 or self.axon_index >= len(self.segmented_axons):
            self.ui.cellIndexTextbox.setText("")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Input axon index out of range")
            msg.show()
            return
        img_annotated = self.img_annotated.copy()
        height = len(self.img_annotated)
        width = len(self.img_annotated[0])
        for pixel in self.segmented_axons[self.axon_index]:
            img_annotated[max(pixel[0] - 3, 0): min(pixel[0] + 3, height),
            max(pixel[1] - 3, 0): min(pixel[1] + 3, width)] = [0, 255, 255]
        im = Image.fromarray(img_annotated.astype(np.uint8))
        im.save("result/result.png", format="png")
        pix = QtGui.QPixmap("result/result.png")
        self.pixmap_item = self.scene.addPixmap(pix)
        self.ui.graphicsView2.setPhotoByScnen(self.scene)

    def get_cell_onclick_callback(self, x, y):
        print(x, y)
        if self.img_annotated is not None:
            self.cell_index = self.segmentation_analysis.clickOnPixel(x, y)
            self.ui.cellIndexTextbox.setText(str(self.cell_index))
            self.get_cell_index_callback()

    def save_button_callback(self):
        if self.seg_image is not None:
            img = Image.fromarray(self.seg_image.astype(np.uint8), mode='RGB')
            head_tail = os.path.split(self.input_file_name)
            img.save("result/segmentation/result-" + head_tail[1][:-4] + ".png", format="png")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Success")
            msg.setText("Segmentation Image Successfully saved")
            msg.show()
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Save Failed")
            msg.show()

    def export_button_callback(self, cell_count, avg_length):
        if self.segmentation_analysis != "":
            # Write
            info_dict = {"Filtered Cell Clusters Count": self.segmentation_analysis.nr_filtered_cell,
                         "Cell Clusters Count": self.segmentation_analysis.nr_cell,
                         "Axons Count": len(self.segmented_axons), "Segmented Axons": {}}
            # info_dict["Axons to Cells"] = self.segmentation_analysis.axons_to_cells
            # info_dict["Info List"] = self.segmentation_analysis.info_list
            for i in range(len(self.segmented_axons)):
                info_dict["Segmented Axons"][i] = self.segmented_axons[i]
            # info_dict["Cell to Axons"] = self.segmentation_analysis.cell_axons_map
            info_dict["Cell areas"] = self.segmentation_analysis.cell_area_map
            head_tail = os.path.split(self.input_file_name)
            with open("result/data/" + head_tail[1][:-4] + ".json", "w") as outfile:
                json.dump(info_dict, outfile, cls=NpEncoder)

            columns = ["Axon Index", "Axon Length", "Axon Orientation"]
            axon_indexes = []
            connected_clusters = []
            axon_lengths = []
            axon_orientation = []
            for i in range(len(self.segmented_axons)):
                axon_indexes.append(i)
                axon_lengths.append(pixel_to_length(self.segmentation_analysis.segmented_axons_dist[i]))
                if self.segmentation_analysis.show_orientation[i]:
                    axon_orientation.append(
                        degree_to_ori(getOrientation(self.segmented_axons[i][0], self.segmented_axons[i][-1])))
                else:
                    axon_orientation.append(
                        degree_to_ori(getOrientation(self.segmented_axons[i][0], self.segmented_axons[i][-1])) + "," +
                        degree_to_ori(getOrientation(self.segmented_axons[i][-1], self.segmented_axons[i][0])))
            df = pd.DataFrame(list(zip(axon_indexes, axon_lengths, axon_orientation)),
                              columns=columns)
            writer = pd.ExcelWriter("result/data/" + head_tail[1][:-4] + ".xlsx", engine="xlsxwriter")
            df.to_excel(writer, index=False, sheet_name='axon-tracing')
            # Automatically adjust width of the columns
            for column in df:
                column_width = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                writer.sheets['axon-tracing'].set_column(col_idx, col_idx, column_width)
            writer.save()

            # Writing to Summary Excel
            columns = ["Image Name", "# of Cell Clusters", "Estimated # of Cells", "# of Axons", "Average Axon Length"]
            image_names = [head_tail[1][:-4]]
            cell_clusters = [self.segmentation_analysis.nr_filtered_cell]
            cell_counts = [int(cell_count)]
            axon_counts = [len(self.segmented_axons)]
            axon_lengths = [round(avg_length, 2)]
            if os.path.isfile("result/data/summary.xlsx"):
                df = pd.DataFrame(list(zip(image_names, cell_clusters, cell_counts, axon_counts, axon_lengths)),
                                  columns=columns)
                workbook = openpyxl.load_workbook("result/data/summary.xlsx")  # load workbook if already exists
                sheet = workbook['cell_analysis']  # declare the active sheet

                # append the dataframe results to the current excel file
                for row in dataframe_to_rows(df, header=False, index=False):
                    sheet.append(row)
                workbook.save("result/data/summary.xlsx")  # save workbook
                workbook.close()  # close workbook
            else:
                df = pd.DataFrame(list(zip(image_names, cell_clusters, cell_counts, axon_counts, axon_lengths)),
                                  columns=columns)
                writer = pd.ExcelWriter("result/data/summary.xlsx", engine="xlsxwriter")
                df.to_excel(writer, index=False, sheet_name='cell_analysis')
                # Automatically adjust width of the columns
                for column in df:
                    column_width = max(df[column].astype(str).map(len).max(), len(column))
                    col_idx = df.columns.get_loc(column)
                    writer.sheets['cell_analysis'].set_column(col_idx, col_idx, column_width)
                writer.save()

            x_desc = 'Tracing'
            y_desc = 'Cluster'
            z_desc = 'Length'
            desc = [x_desc, y_desc, z_desc]

            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Success")
            msg.setText("Analysis Data Successfully exported")
            msg.show()
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Export Failed")
            msg.show()

    def display_img(self, image):
        img = Image.fromarray(image.astype(np.uint8), mode='RGB')
        img.save("result/result.png", format="png")
        pix = QtGui.QPixmap("result/result.png")
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
        model_index = self.ui.modelSelection.currentIndex()
        if self.gamma_image is not None:
            # conduct the process
            if self.seg_image is None:
                self.run_task_unet(self.gamma_image, model_index)
            else:
                # display the old result
                self.display_img(self.seg_image.astype(np.uint8))
                self.save_button_callback()
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Warning: Gamma Correction not applied!")
            msg.show()

    def clear_button_callback(self):
        self.ui.textBrowser.setText("")
        self.ui.graphicsView1.setPhotoByScnen(None)
        self.ui.graphicsView2.setPhotoByScnen(None)
        _translate = QtCore.QCoreApplication.translate
        self.ui.info.setText(_translate("MainWindow", "General Information:"))
        self.ui.cellInfo.setText(_translate("MainWindow", "Cell Information:"))
        if self.chartView is not None:
            self.ui.infoboxLayout.removeWidget(self.chartView)
        self.chartView = None
        if self.areaChartView is not None:
            self.ui.cellAreaInfoboxLayout.removeWidget(self.areaChartView)
        self.areaChartView = None
        self.input_file_name = ""
        self.output_file_name = ""
        self.img = None
        self.gamma_image = None
        self.seg_image = None
        self.img_annotated = None
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.installEventFilter(self)
        self.cell_filter_size = 500
        self.cell_index = None
        self.axon_index = None
        self.segmentation_analysis = ""
        self.segmented_axons = []
        self.pixmap_item = None

    def analyze_button_callback(self):
        self.segmentation_analysis = SegmentationAnalysis()
        if self.seg_image is None:
            self.segmented_axons = self.segmentation_analysis.run(self.input_file_name, self.img, self.cell_filter_size)
            self.img_annotated = self.img.copy()
        else:
            self.segmented_axons = self.segmentation_analysis.run(self.input_file_name, self.seg_image,
                                                                  self.cell_filter_size)
            self.img_annotated = self.seg_image.copy()
        # print(len(img_annotated), len(img_annotated[0]))
        height = len(self.img_annotated)
        width = len(self.img_annotated[0])
        # Display filaments on the picture for debug purpose
        length_filament = len(self.segmentation_analysis.fil.filaments)
        for i in range(length_filament):
            for j in range(len(self.segmentation_analysis.fil.filaments[i].pixel_coords[0])):
                x = self.segmentation_analysis.fil.filaments[i].pixel_coords[0][j]
                y = self.segmentation_analysis.fil.filaments[i].pixel_coords[1][j]
                self.img_annotated[x, y] = [255, 0, 255]
        # print(img_annotated[0][0])
        for i in range(height):
            for j in range(width):
                if self.segmentation_analysis.cell_boundary_mask[i][j] != 0:
                    self.img_annotated[max(i - 1, 0): min(i + 1, height - 1),
                    max(j - 1, 0): min(j + 1, width - 1)] = [255, 255, 0]
        for axon in self.segmented_axons:
            for pixel in axon:
                # print(pixel)
                self.img_annotated[max(pixel[0] - 3, 0): min(pixel[0] + 3, height - 1),
                max(pixel[1] - 3, 0): min(pixel[1] + 3, width - 1)] = [255, 0, 0]
        for i in self.segmentation_analysis.info_list:
            print(i)
            for j in i["touch_points"]:
                x = j[0]
                y = j[1]
                self.img_annotated[x, y] = [127, 127, 127]
        self.display_img(self.img_annotated.astype(np.uint8))
        # Save the Annotated image
        try:
            img = Image.fromarray(self.img_annotated.astype(np.uint8), mode='RGB')
            head_tail = os.path.split(self.input_file_name)
            if head_tail[1].startswith("result-"):
                img.save("result/segmentation/" + head_tail[1][:-4] + "-annotated.png", format="png")
            else:
                img.save("result/segmentation/result-" + head_tail[1][:-4] + "-annotated.png", format="png")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Success")
            msg.setText("Annotated Image Successfully saved")
            msg.show()
        except:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Failed to save the annotated image")
            msg.show()

        _translate = QtCore.QCoreApplication.translate
        # Gather information for axon length/orientation
        length_dist = [0, 0, 0, 0, 0]
        total_length = 0
        axon_set = [[0] * 4 for i in range(5)]
        for i in range(len(self.segmented_axons)):
            length = pixel_to_length(self.segmentation_analysis.segmented_axons_dist[i])
            orientation = getOrientation(self.segmented_axons[i][0], self.segmented_axons[i][-1])
            total_length += pixel_to_length(self.segmentation_analysis.segmented_axons_dist[i])
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
            if not self.segmentation_analysis.show_orientation[i]:
                continue
            if orientation <= 45 or orientation >= 315:
                axon_set[index][0] += 1
            elif orientation <= 135:
                axon_set[index][1] += 1
            elif orientation <= 225:
                axon_set[index][2] += 1
            else:
                axon_set[index][3] += 1
        length_range = max(max(x) for x in axon_set)
        average_length = total_length / len(self.segmented_axons) if len(self.segmented_axons) != 0 else 0
        cell_count = 0
        cell_area = 0
        for v in self.segmentation_analysis.cell_area_map.values():
            cell_count += (v / TYPICALCELLAREA).round()
            cell_area += v
        # Display General information
        self.ui.info.setText(_translate("MainWindow",
                                        "General Information:\n"
                                        "Cell clusters count: " + str(
                                            self.segmentation_analysis.nr_filtered_cell) + "\n" +
                                        "Axons count: " + str(len(self.segmented_axons)) + "\n" +
                                        "Estimated cells count: " + "{:.0f}".format(cell_count) + "\n" +
                                        "Average cell area: " + "{:.0f}".format(cell_area / cell_count) + "µm^2\n" +
                                        "Average axon length: " + "{:.0f}".format(average_length) + "μm\n"))

        # Create sets for different length ranges
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
        series = QtCharts.QBarSeries()
        series.append(set0)
        series.append(set1)
        series.append(set2)
        series.append(set3)
        series.append(set4)
        # Axon Length/Orientation Table
        axonLengthWidget = QtCharts.QChart()
        axonLengthWidget.addSeries(series)
        axonLengthWidget.setTitle("Axon Length/Orientation Distribution")
        axonLengthWidget.setTitleFont(QtGui.QFont("New Times Roman", 12))
        # X-axis for Orientation
        categories = ["N", "E", "S", "W"]
        axisX = QtCharts.QBarCategoryAxis()
        axisX.append(categories)
        axonLengthWidget.addAxis(axisX, QtCore.Qt.AlignmentFlag.AlignBottom)
        # Y-axis for axon counts
        axisY = QtCharts.QValueAxis()
        axisY.setRange(0, length_range)
        axisY.setLabelFormat("%d")
        # Plot graph properties
        axonLengthWidget.addAxis(axisY, QtCore.Qt.AlignmentFlag.AlignLeft)
        axonLengthWidget.legend().setVisible(True)
        axonLengthWidget.legend().adjustSize()
        axonLengthWidget.legend().setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)
        axonLengthWidget.resize(100, 75)
        axonLengthWidget.setBackgroundBrush(QtGui.QColor(164, 172, 150, 255))
        self.chartView = QtCharts.QChartView(axonLengthWidget)
        self.chartView.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.ui.infoboxLayout.addWidget(self.chartView)
        # Cell Area Distribution
        cell_area_dist = [0] * 8
        for area in self.segmentation_analysis.cell_area_map.values():
            for j in range(8):
                if CELLTHRESHOLD[j] <= area < CELLTHRESHOLD[j + 1]:
                    cell_area_dist[j] += 1
        area_series = QtCharts.QBarSeries()
        area_set = QtCharts.QBarSet("Cell Area")
        area_categories = []
        for j in range(8):
            area_set.append([cell_area_dist[j]])
            area_categories.append(str(CELLTHRESHOLD[j]))
        area_series.append(area_set)
        # Axon Length/Orientation Table
        areaDistWidget = QtCharts.QChart()
        areaDistWidget.addSeries(area_series)
        areaDistWidget.setTitle("Cell Area Distribution")
        areaDistWidget.setTitleFont(QtGui.QFont("New Times Roman", 12))
        # X-axis for Orientation
        axisX = QtCharts.QBarCategoryAxis()
        axisX.append(area_categories)
        areaDistWidget.addAxis(axisX, QtCore.Qt.AlignmentFlag.AlignBottom)
        # Y-axis for axon counts
        axisY = QtCharts.QValueAxis()
        axisY.setRange(0, max(cell_area_dist))
        axisY.setLabelFormat("%d")
        # Plot graph properties
        areaDistWidget.addAxis(axisY, QtCore.Qt.AlignmentFlag.AlignLeft)
        areaDistWidget.legend().setVisible(True)
        areaDistWidget.legend().adjustSize()
        areaDistWidget.legend().setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)
        areaDistWidget.resize(100, 75)
        areaDistWidget.setBackgroundBrush(QtGui.QColor(164, 172, 150, 255))
        self.areaChartView = QtCharts.QChartView(areaDistWidget)
        self.areaChartView.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.ui.cellAreaInfoboxLayout.addWidget(self.areaChartView)
        # self.display_img(im)
        self.export_button_callback(cell_count, average_length)

    def eventFilter(self, source, event):
        if source is self.scene and event.type() == QtCore.QEvent.Type.GraphicsSceneMouseDoubleClick:
            spf = event.scenePos()
            lpf = self.pixmap_item.mapFromScene(spf)
            brf = self.pixmap_item.boundingRect()
            if brf.contains(lpf):
                lp = lpf.toPoint()
                self.get_cell_onclick_callback(lp.y(), lp.x())
        return super(ImgSeg, self).eventFilter(source, event)

    def run_task_unet(self, input_data, model=0):
        self.thread = QThread()
        self.worker = TaskThreadUnet(input_data, model)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run_unet)
        self.worker.finished.connect(self.return_value)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # self.worker.progress.connect(self.reportProgress)
        self.ui.pushButton.setEnabled(False)
        self.ui.graphicsView2.hide()
        self.ui.giflabel.show()
        self.movie.start()
        self.thread.start()
        #
        # self.thread.finished.connect(
        #     self.ui.pushButton.setEnabled(True)
        # )

    def return_value(self, ret_value):
        self.seg_image = ret_value
        self.display_img(self.seg_image.astype(np.uint8))
        self.save_button_callback()
        self.ui.pushButton.setEnabled(True)
        self.movie.stop()
        self.ui.giflabel.hide()
        self.ui.graphicsView2.show()


class TaskThreadUnet(QObject):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)

    def __init__(self, input_message, model=0):
        super(QObject, self).__init__()
        self.input_image = input_message
        self.input_model = model

    def run_unet(self, model=0):
        """Long-running task."""
        ret_value = unet_predict(self.input_image, self.input_model)
        # self.progress.emit('')
        self.finished.emit(ret_value)


if __name__ == '__main__':
    if not os.path.exists('./result'):
        os.makedirs('./result')
    if not os.path.exists('./result/data'):
        os.makedirs('./result/data')
    if not os.path.exists('./result/segmentation'):
        os.makedirs('./result/segmentation')
    app = QtWidgets.QApplication(sys.argv)
    myapp = ImgSeg()
    myapp.show()
    sys.exit(app.exec())
