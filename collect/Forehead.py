# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hello.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer, QTime, Qt
from PyQt5.QtMultimedia import QSound
import api

'''
上额测试类，弹出上额测试窗口，提示受试者保持皱眉，每次持续10s，共2次，其间休息20s
'''

class Forehead_test_Window(QDialog):
    def __init__(self):
        super(Forehead_test_Window, self).__init__()
        self.setWindowTitle("上额伪迹采集")
        self.resize(580,580)

        # 创建显示倒计时的 QLabel
        layout = QtWidgets.QVBoxLayout(self)  # 创建一个 QVBoxLayout 布局管理器
        self.timer_label = QLabel(self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(35)  # 设置字体大小
        self.timer_label.setFont(font)
        layout.addWidget(self.timer_label, alignment=Qt.AlignCenter)

        # 创建定时器
        self.squeeze_timer = QTimer(self)
        self.rest_timer = QTimer(self)
        self.reminder_timer = QTimer(self)
        self.squeeze_timer.timeout.connect(self.squeeze_hold)
        self.rest_timer.timeout.connect(self.rest_repeat)
        self.reminder_timer.timeout.connect(self.start_alarm)

    def reminder(self):#此函数为开启上额测试的初始函数
        self.rest_num = 1 #休息间隔为1次，即皱眉，休息，皱眉
        self.squeeze()

    def squeeze(self): #调用此函数开始皱眉测试
        self.remaintime = 5 #挑眉次数
        self.squeeze_count = 0
        self.timer_label.setText("根据提示音挑眉")
        QSound.play("挑眉.wav")
        self.reminder_timer.start(3000)

    def start_alarm(self):
        self.reminder_timer.stop()
        self.squeeze_timer.start(1000)

    def rest(self): #调用此函数开始休息
        self.rest_remaintime = 20
        QSound.play("休息.wav")
        self.play_alarm()
        self.rest_timer.start(1000)

    def squeeze_hold(self):
        minutes = self.remaintime // 60
        seconds = self.remaintime % 60
        label_text = "开始挑眉"
        self.timer_label.setText(f"{label_text}:{minutes:02d}:{seconds:02d}")

        if self.remaintime != 0:
            self.play_alarm()
            self.squeeze_count += 1
            if self.squeeze_count > 1:
                api.mark(0)
            if self.squeeze_count != 0:
                api.mark(12)
            self.remaintime -= 1
        else:
            self.squeeze_timer.stop()
            api.mark(0)
            if self.rest_num == 0:#根据已休息次数判断接下来休息还是结束整轮测试
                self.play_alarm()
                QSound.play("结束.wav")
                self.timer_label.setText("上额测试结束！")
            else:
                self.rest()

    def rest_repeat(self):
        minutes = self.rest_remaintime // 60
        seconds = self.rest_remaintime % 60
        label_text = "休息时间"
        self.timer_label.setText(f"{label_text}:{minutes:02d}:{seconds:02d}")

        if self.rest_remaintime != 0:
            self.rest_remaintime -= 1
        else:
            self.rest_timer.stop()
            if self.rest_num != 0: #根据已休息次数判断是否继续皱眉测试
                self.squeeze()
                self.rest_num -= 1



    def play_alarm(self):
        # 播放声音，确保你有这个文件
        QSound.play("alert.wav")  # 音频文件路径


