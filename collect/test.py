#自动伪迹采集系统，提示受试者做出相应动作并自动打标，链接脑电设备api
#test.py为qt初始化程序，运行test.py开启自动伪迹采集系统
#此采集系统由主窗口，基线测试，眼动测试，肌动测试，舌动测试，下颚测试，上额测试组成
#作者：余江伟
#更新时间：10，17，2024

import sys
import main
from PyQt5.QtWidgets import QApplication,QMainWindow

if __name__ == '__main__':
    # 只有直接运行这个脚本，才会往下执行
    # 别的脚本文件执行，不会调用这个条件句

    # 实例化，传参
    app = QApplication(sys.argv)

    # 创建对象
    mainWindow = QMainWindow()

    # 创建ui，引用demo1文件中的Ui_MainWindow类
    ui = main.Ui_MainWindow()
    # 调用Ui_MainWindow类的setupUi，创建初始组件
    ui.setupUi(mainWindow)
    # 创建窗口
    mainWindow.show()
    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())
