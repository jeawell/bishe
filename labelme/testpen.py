import sys
from PyQt5 import QtWidgets, QtGui, QtCore

class DrawingWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(800, 600)

        # 初始化两个层：背景和绘图层
        self.background_layer = QtGui.QPixmap(self.size())
        self.background_layer.load('icons/brush.png')

        self.overlay_layer = QtGui.QPixmap(self.size())
        # 将整个绘画层填充为透明。这是关键一步，确保背景可以被看见
        self.overlay_layer.fill(QtCore.Qt.transparent)
        # 工具参数
        self.pen_color = QtGui.QColor('black')
        self.pen_width = 5
        self.eraser_width = 20
        self.mode = 'eraser'

        self.last_pos = None
        self.drawing = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self.background_layer)
        painter.drawPixmap(0, 0, self.overlay_layer)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.last_pos = event.pos()
        elif event.button() == QtCore.Qt.RightButton:
            self.mode = 'eraser' if self.mode == 'pen' else 'pen'
            print("切换为：", self.mode)

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_pos is not None:
            painter = QtGui.QPainter(self.overlay_layer)
            if self.mode == 'pen':
                pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.last_pos, event.pos())
            elif self.mode == 'eraser':
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
                painter.setBrush(QtCore.Qt.transparent)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(event.pos(), self.eraser_width / 2, self.eraser_width / 2)

            painter.end()
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False
            self.last_pos = None

    def save_overlay(self):
        self.overlay_layer.save("overlay_saved.png", "PNG")
        print("保存成功为 overlay_saved.png")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自定义 QWidget 画布")

        self.drawing_widget = DrawingWidget()

        save_button = QtWidgets.QPushButton("保存上层")
        save_button.clicked.connect(self.drawing_widget.save_overlay)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.drawing_widget)
        layout.addWidget(save_button)


        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
