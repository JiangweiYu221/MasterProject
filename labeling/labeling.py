import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QCheckBox, QPushButton, QDialog,
    QLabel, QLineEdit, QFormLayout, QComboBox, QMessageBox, QVBoxLayout
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont
import pandas as pd
import os
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")  # 声明使用 QT5
import mne




class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选择列表")
        self.resize(1000, 600)

        # 创建主窗口布局
        self.layout = QGridLayout(self)
        # 字体设置
        self.font = QFont("Arial", 12)

        # 创建一个下拉框
        self.combo_box = QComboBox()
        self.combo_box.setFont(self.font)

        # 添加选项到下拉框
        # self.combo_box.addItems(["闭眼静息态", "睁眼静息态", "垂直眼动", "水平眼动", "眨眼", "水平摇头",
        #                          "垂直点头", "舌动", "咬牙", "吞咽", "挑眉", "眨眼+水平摇头",
        #                          "眨眼+垂直点头", "眨眼+挑眉", "舌动+挑眉", "吞咽+挑眉"])
        self.combo_box.addItems(["close_base", "open_base", "ver_eyem", "hor_eyem", "blink", "hor_headm",
                                 "ver_headm", "tongue", "chew", "swallow", "eyebrow", "blink_hor_headm",
                                 "blink_ver_headm", "blink_eyebrow", "tongue_eyebrow", "swallow_eyebrow"])

        # 将下拉框添加到主布局
        self.layout.addWidget(self.combo_box,0, 0, 1, 2, Qt.AlignCenter)
        self.combo_box.setCurrentIndex(-1)
        self.combo_box.currentIndexChanged.connect(self.setcheckbox)



        # 列表数据
        self.elements = [
            'Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T8',
            'T8-P8', 'P8-O2', 'F7-Fp1', 'Fp1-Fp2', 'F7-F3', 'F3-Fz', 'Fz-F4', 'F4-F8',
            'T7-C3', 'C3-Cz', 'Cz-C4', 'C4-T8', 'P7-P3', 'P3-Pz', 'Pz-P4', 'P4-P8',
            'O1-O2', 'O2-P8', 'ALL'
        ]


        # 添加复选框到窗口，排列为两列
        self.checkboxes = []
        for idx, element in enumerate(self.elements):
            checkbox = QCheckBox(element)
            checkbox.setObjectName(element)
            checkbox.setFont(self.font)  # 设置字体大小
            self.checkboxes.append(checkbox)
            row = 1 + idx % 18  # 计算行号
            col = idx // 18   # 计算列号
            self.layout.addWidget(checkbox, row, col)



        # 添加确认按钮
        self.confirm_button = QPushButton("确认")
        self.confirm_button.setFont(self.font)  # 设置按钮字体大小
        self.confirm_button.clicked.connect(self.open_new_window)
        self.layout.addWidget(self.confirm_button, len(self.elements) // 2 + 3, 0, 1, 2, Qt.AlignCenter)


    def setcheckbox(self):
        # 获取下拉框的选择
        label = self.combo_box.currentText()
        csv_file = "label_csv/yjw2.csv"
        df = pd.read_csv(csv_file)
        filtered = df[df['label'] == label]
        select = filtered['channel'].drop_duplicates().tolist()
        for checkbox in self.checkboxes:
            objname = checkbox.objectName()
            if objname in select:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)

        self.update()


    def open_new_window(self):
        # 获取选中的元素
        self.selected_items = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        if self.selected_items:
            self.dialog = InputDialog(self.selected_items, parent=self)
            self.dialog.show()


class InputDialog(QDialog):
    def __init__(self, selected_items, parent=None):
        super().__init__(parent)
        self.selected_items = selected_items
        self.setWindowTitle("输入信息")
        self.resize(600, 600)

        # 创建布局
        layout = QFormLayout(self)

        # 字体设置
        self.font = QFont("Arial", 12)

        # 提示文字
        self.label = QLabel(f"选中的通道:\n {', '.join(selected_items)}")
        self.label.setFont(self.font)
        self.label.setWordWrap(True)  # 自动换行
        layout.addWidget(self.label)

        # 提示文本
        label_hint_sub = QLabel("subject:")
        label_hint_sub.setFont(self.font)
        layout.addWidget(label_hint_sub)
        settings = QSettings("labeling", "YJW")
        saved_subject = settings.value("subject", "")
        # 输入框
        self.input_field_sub = QLineEdit()
        self.input_field_sub.setFont(self.font)
        layout.addWidget(self.input_field_sub)
        self.input_field_sub.setText(saved_subject)
        # 提示文本
        label_font = QFont("Arial", 16)
        label_hint_sub = QLabel("label:" + self.parent().combo_box.currentText())
        label_hint_sub.setFont(label_font)
        layout.addWidget(label_hint_sub)

        # # 添加选项到下拉框
        # # self.combo_box.addItems(["闭眼静息态", "睁眼静息态", "垂直眼动", "水平眼动", "眨眼", "水平摇头",
        # #                          "垂直点头", "舌动", "咬牙", "吞咽", "挑眉", "眨眼+水平摇头",
        # #                          "眨眼+垂直点头", "眨眼+挑眉", "舌动+挑眉", "吞咽+挑眉"])
        # self.combo_box.addItems(["close_base", "open_base", "ver_eyem", "hor_eyem", "blink", "hor_headm",
        #                          "ver_headm", "tongue", "chew", "swallow", "eyebrow", "blink_hor_headm",
        #                          "blink_ver_headm", "blink_eyebrow", "tongue_eyebrow", "swallow_eyebrow"])


        # 添加按钮以打开新窗口
        self.plot_button = QPushButton("调用MNE")
        self.plot_button.setFont(self.font)
        self.plot_button.clicked.connect(self.open_plot_window)
        layout.addWidget(self.plot_button)

        # 添加确认按钮
        self.confirm_button = QPushButton("保存")
        self.confirm_button.setFont(self.font)
        self.confirm_button.clicked.connect(self.save_data)
        layout.addWidget(self.confirm_button)


        # 调用居中方法
        self.center_window()
        self.setLayout(layout)

    def open_plot_window(self):
        self.subject_name = self.input_field_sub.text()
        root_path = "D:\qq文件\交接代码\伪迹数据\mne_set/"
        path = root_path + self.subject_name + "_data.set"
        self.data = mne.io.read_raw_eeglab(path, preload=True)
        # 创建一个 Matplotlib 图形
        self.figure = Figure(figsize=(1, 1))
        self.canvas = FigureCanvas(self.figure)
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 12
        annot_file = self.subject_name + "_annotations.txt"
        start_time = 0
        if os.path.isfile(annot_file):
            with open(annot_file, 'r') as file:
                lines = file.readlines()
                if lines:
                    start, duration ,_  = lines[-1].strip().split(',')
                    start_time = float(start) + float(duration)

        self.data.plot(duration=20, n_channels=34, scalings=120, start = start_time, clipping = 10)

        # 更新画布
        self.canvas.draw()



    def center_window(self):
        # 获取父窗口的位置和大小
        if self.parent():
            parent_geometry = self.parent().geometry()
            parent_center = parent_geometry.center()
            self_geometry = self.frameGeometry()
            self_geometry.moveCenter(parent_center)
            self.move(self_geometry.topLeft())

    def save_data(self):
        # 获取选中的复选框元素
        selected_channels = self.selected_items
        # 获取下拉框的选择
        label = self.parent().combo_box.currentText()
        self.subject_name = self.input_field_sub.text()
        # 保存上次内容
        settings = QSettings("labeling","YJW")
        settings.setValue("subject", self.subject_name)

        annot_file = self.subject_name + "_annotations.txt"
        self.data.annotations.save(annot_file, overwrite=True)

        start_time = []
        stop_time = []
        annot_file = self.subject_name + "_annotations.txt"
        with open(annot_file, 'r') as file:
            for line in file:
                # 跳过注释行（以#开头）或空行
                if line.strip().startswith('#') or not line.strip():
                    continue
                # 解析有效数据行，假设以逗号分隔
                parts = line.strip().split(',')
                onset, duration, _ = parts
                start_time.append(onset)
                stop_time.append((float(onset) + float(duration)))

        # 创建 DataFrame
        data = {
            "channel": [],
            "start_time": [],
            "stop_time": [],
            "label": []
        }

        for i in range(len(start_time)):
            for channel in selected_channels:  # 遍历每个通道
                data["channel"].append(channel)
                data["start_time"].append(start_time[i])
                data["stop_time"].append(stop_time[i])
                data["label"].append(label)

        df = pd.DataFrame(data)

        # 定义保存的 CSV 文件路径
        self.file_path = self.subject_name+".csv"

        # 判断文件是否已存在
        if not os.path.isfile(self.file_path):
            # 文件不存在，写入表头和数据
            df.to_csv(self.file_path, index=False)
            self.show_create_message()
            self.accept()
        else:
            # 文件存在，追加数据，不写表头
            df.to_csv(self.file_path, mode="a", header=False, index=False)

            # 使用弹出窗口提示保存成功
            self.show_success_message()

            for checkbox in self.parent().checkboxes:
                checkbox.setChecked(False)

            # QApplication.quit()
            # 关闭窗口
            self.accept()

    def show_success_message(self):
        # 创建弹出窗口
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("保存成功")
        msg.setText(f"文件已保存到: {self.file_path}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_create_message(self):
        # 创建弹出窗口
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("创建成功")
        msg.setText("csv文件创建成功")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


def run_labeling():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

run_labeling()


