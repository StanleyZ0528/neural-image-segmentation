from PyQt6.QtWidgets import QApplication, QPushButton, QLabel, QVBoxLayout, QWidget, \
    QTextBrowser, QSizePolicy, QFileDialog, QMessageBox, QProgressBar
from PyQt6 import QtCore
from unet.unet_utils import gamma_correction, unet_predict
from postImgProc.alg import *
import sys
import os


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
        for i in range(total_image):
            self._signal.emit(i)
            cur_data = self.data_inputs[i]
            if cur_data.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                try:
                    cur_img = readImg(cur_data)
                    gamma_img = gamma_correction(cur_img)
                    unet_img = unet_predict(gamma_img, 1)
                    # add analysis function...

                    folder_name = os.path.basename(cur_data).split('.', 1)[0]
                    file_name = os.path.splitext(os.path.basename(cur_data))[0]
                    img = Image.fromarray(unet_img.astype(np.uint8), mode='RGB')
                    img.save("result/{:s}.png".format(file_name), format="png")
                    # img.save("result/{:s}/{:s}.png".format(folder_name, file_name), format="png")
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
    app = QApplication(sys.argv)
    ex = ImgProc()
    sys.exit(app.exec())
