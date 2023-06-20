from PyQt6.QtWidgets import QApplication, QPushButton, QLabel, QVBoxLayout, QWidget, \
    QTextBrowser, QSizePolicy, QFileDialog, QMessageBox, QProgressBar
from PyQt6 import QtCore
from unet.unet_utils import gamma_correction, unet_predict
from postImgProc.alg import *
import sys
import os
import pandas as pd
import xlsxwriter
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import json


THRESHOLD0 = 40
THRESHOLD1 = 75
THRESHOLD2 = 100
THRESHOLD3 = 150

TYPICALCELLAREA = 1200  # Approximation from 20*20*3.14
CELL_FILTER_SIZE = 125


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Thread(QtCore.QThread):
    _signal = QtCore.pyqtSignal(int)

    def __init__(self, data_input):
        super(Thread, self).__init__()
        self.data_inputs = data_input

    def __del__(self):
        self.wait()

    def run(self):
        total_image = len(self.data_inputs)
        processed = 0
        for index in range(total_image):
            self._signal.emit(index)
            cur_data = self.data_inputs[index]
            if cur_data.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                try:
                    cur_img = readImg(cur_data)
                    gamma_img = gamma_correction(cur_img)
                    unet_img = unet_predict(gamma_img, 1)

                    folder_name = os.path.basename(cur_data).split('.', 1)[0]
                    file_name = os.path.splitext(os.path.basename(cur_data))[0]
                    img = Image.fromarray(unet_img.astype(np.uint8), mode='RGB')
                    img.save("result/bulk_result/{:s}.png".format(file_name), format="png")
                    # img.save("result/{:s}/{:s}.png".format(folder_name, file_name), format="png")
                    # add analysis function...
                    segmentation_analysis = SegmentationAnalysis()
                    segmented_axons = segmentation_analysis.run("result/bulk_result/{:s}.png".format(file_name), unet_img, CELL_FILTER_SIZE)
                    img_annotated = unet_img.copy()
                    height = len(img_annotated)
                    width = len(img_annotated[0])
                    # Display filaments on the picture for debug purpose
                    length_filament = len(segmentation_analysis.fil.filaments)
                    for i in range(length_filament):
                        for j in range(len(segmentation_analysis.fil.filaments[i].pixel_coords[0])):
                            x = segmentation_analysis.fil.filaments[i].pixel_coords[0][j]
                            y = segmentation_analysis.fil.filaments[i].pixel_coords[1][j]
                            img_annotated[x, y] = [255, 0, 255]
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
                    for i in segmentation_analysis.info_list:
                        for j in i["touch_points"]:
                            x = j[0]
                            y = j[1]
                            img_annotated[x, y] = [127, 127, 127]
                    # Save the Annotated image
                    img = Image.fromarray(img_annotated.astype(np.uint8), mode='RGB')
                    img.save("result/bulk_result/{:s}-annotated.png".format(file_name), format="png")
                    # Export

                    # Gather information for axon length/orientation
                    length_dist = [0, 0, 0, 0, 0]
                    total_length = 0
                    axon_set = [[0] * 4 for i in range(5)]
                    for i in range(len(segmented_axons)):
                        length = pixel_to_length(segmentation_analysis.segmented_axons_dist[i])
                        orientation = getOrientation(segmented_axons[i][0], segmented_axons[i][-1])
                        total_length += pixel_to_length(segmentation_analysis.segmented_axons_dist[i])
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
                        if not segmentation_analysis.show_orientation[i]:
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
                    average_length = total_length / len(segmented_axons) if len(segmented_axons) != 0 else 0
                    cell_count = 0
                    cell_area = 0
                    for v in segmentation_analysis.cell_area_map.values():
                        cell_count += (v / TYPICALCELLAREA).round()
                        cell_area += v
                    info_dict = {"Filtered Cell Clusters Count": segmentation_analysis.nr_filtered_cell,
                                 "Cell Clusters Count": segmentation_analysis.nr_cell,
                                 "Cell Clusters with Axon Count": segmentation_analysis.nr_filtered_cell_w_axon,
                                 "Axons Count": len(segmented_axons), "Segmented Axons": {}}
                    # info_dict["Axons to Cells"] = self.segmentation_analysis.axons_to_cells
                    # info_dict["Info List"] = self.segmentation_analysis.info_list
                    for i in range(len(segmented_axons)):
                        info_dict["Segmented Axons"][i] = segmented_axons[i]
                    # info_dict["Cell to Axons"] = self.segmentation_analysis.cell_axons_map
                    info_dict["Cell areas"] = segmentation_analysis.cell_area_map
                    with open("result/bulk_result/" + file_name + ".json", "w") as outfile:
                        json.dump(info_dict, outfile, cls=NpEncoder)

                    columns = ["Axon Index", "Axon Length", "Axon Orientation"]
                    axon_indexes = []
                    connected_clusters = []
                    axon_lengths = []
                    axon_orientation = []
                    for i in range(len(segmented_axons)):
                        axon_indexes.append(i)
                        axon_lengths.append(pixel_to_length(segmentation_analysis.segmented_axons_dist[i]))
                        if segmentation_analysis.show_orientation[i]:
                            axon_orientation.append(
                                degree_to_ori(getOrientation(segmented_axons[i][0], segmented_axons[i][-1])))
                        else:
                            axon_orientation.append(
                                degree_to_ori(
                                    getOrientation(segmented_axons[i][0], segmented_axons[i][-1])) + "," +
                                degree_to_ori(getOrientation(segmented_axons[i][-1], segmented_axons[i][0])))
                    df = pd.DataFrame(list(zip(axon_indexes, axon_lengths, axon_orientation)),
                                      columns=columns)
                    writer = pd.ExcelWriter("result/bulk_result/" + file_name + ".xlsx", engine="xlsxwriter")
                    df.to_excel(writer, index=False, sheet_name='axon-tracing')
                    # Automatically adjust width of the columns
                    for column in df:
                        column_width = max(df[column].astype(str).map(len).max(), len(column))
                        col_idx = df.columns.get_loc(column)
                        writer.sheets['axon-tracing'].set_column(col_idx, col_idx, column_width)
                    writer.save()

                    # Writing to Summary Excel
                    columns = ["Image Name", "# of Cell Clusters", "Estimated # of Cells", "# of Axons",
                               "Average Axon Length"]
                    image_names = [file_name]
                    cell_clusters = [segmentation_analysis.nr_filtered_cell]
                    cell_counts = [int(cell_count)]
                    axon_counts = [len(segmented_axons)]
                    axon_lengths = [round(average_length, 2)]
                    if os.path.isfile("result/bulk_result/summary.xlsx"):
                        df = pd.DataFrame(list(zip(image_names, cell_clusters, cell_counts, axon_counts, axon_lengths)),
                                          columns=columns)
                        workbook = openpyxl.load_workbook("result/bulk_result/summary.xlsx")  # load workbook if already exists
                        sheet = workbook['cell_analysis']  # declare the active sheet

                        # append the dataframe results to the current excel file
                        for row in dataframe_to_rows(df, header=False, index=False):
                            sheet.append(row)
                        workbook.save("result/bulk_result/summary.xlsx")  # save workbook
                        workbook.close()  # close workbook
                    else:
                        df = pd.DataFrame(list(zip(image_names, cell_clusters, cell_counts, axon_counts, axon_lengths)),
                                          columns=columns)
                        writer = pd.ExcelWriter("result/bulk_result/summary.xlsx", engine="xlsxwriter")
                        df.to_excel(writer, index=False, sheet_name='cell_analysis')
                        # Automatically adjust width of the columns
                        for column in df:
                            column_width = max(df[column].astype(str).map(len).max(), len(column))
                            col_idx = df.columns.get_loc(column)
                            writer.sheets['cell_analysis'].set_column(col_idx, col_idx, column_width)
                        writer.save()
                    processed += 1
                except ValueError:
                    continue
            else:
                continue
        self._signal.emit(-1 * processed)


class ProcessWindow(QWidget):
    def __init__(self, inputs):
        super().__init__()
        self.data = inputs
        self.percent = 100.0 / len(self.data)

        self.setWindowTitle('Progressing')
        self.progressbar = QProgressBar(self)
        self.progressbar.setValue(0)
        self.label = QLabel(self)
        self.label.setObjectName("label")
        self.resize(300, 100)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.progressbar)
        self.vbox.addWidget(self.label)
        self.setLayout(self.vbox)

        self.thread = Thread(inputs)
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()

    def signal_accept(self, msg):
        if msg < 0:
            self.label.setText(
                "Processing: done. Total: " + str(msg * -1) + ", Skip: " + str(len(self.data) - msg * -1))
            self.progressbar.setValue(100)
        else:
            self.progressbar.setValue(int(self.percent * (msg + 1)))
            self.label.setText("({:d}/{:d}) Processing: {:s}".format(msg + 1, len(self.data), self.data[msg]))


class ImgProc(QWidget):

    def __init__(self):
        super().__init__()
        self.w = None
        self.input_file_name = []

        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setGeometry(QtCore.QRect(10, 10, 600, 250))
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(size_policy)
        self.textBrowser.setObjectName("textBrowser")

        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(620, 10, 120, 120))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)

        self.addButton = QPushButton('Add', self.verticalLayoutWidget)
        self.addButton.setObjectName("chooseButton")
        self.verticalLayout.addWidget(self.addButton)
        self.addButton.clicked.connect(self.choose_button_callback)

        self.clearButton = QPushButton('Clear', self.verticalLayoutWidget)
        self.clearButton.setObjectName("clearButton")
        self.verticalLayout.addWidget(self.clearButton)
        self.clearButton.clicked.connect(self.clear_button_callback)

        self.proButton = QPushButton('Process', self.verticalLayoutWidget)
        self.proButton.setObjectName("processButton")
        self.verticalLayout.addWidget(self.proButton)
        self.proButton.clicked.connect(self.process_button_callback)
        self.show()

    def choose_button_callback(self):
        self.input_file_name += QFileDialog.getOpenFileNames(self)[0]
        self.textBrowser.setText('\n'.join(self.input_file_name))

    def clear_button_callback(self):
        self.textBrowser.setText('')
        self.input_file_name = []

    def process_button_callback(self):
        if len(self.input_file_name) < 1:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setText("Please check the input file")
            msg.show()
        else:
            self.w = ProcessWindow(self.input_file_name)
            self.w.show()


if __name__ == '__main__':
    if not os.path.exists('./result/bulk_result'):
        os.makedirs('./result/bulk_result')
    app = QApplication(sys.argv)
    ex = ImgProc()
    sys.exit(app.exec())
