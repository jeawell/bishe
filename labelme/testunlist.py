import sys
import html
from PyQt5 import QtWidgets, QtCore, QtGui


class UniqueLabelQListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shortcuts = []
        self.shortcut_callback = None
        self.gender_color = None  # 可选颜色生成函数
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.model().rowsMoved.connect(self.updateShortcuts)
    def setShortcutCallback(self, callback):
        """设置快捷键回调函数"""
        self.shortcut_callback = callback
        self.updateShortcuts()

    def findItemByLabel(self, label):
        for row in range(self.count()):
            item = self.item(row)
            if item.data(QtCore.Qt.UserRole) == label:
                return item
        return None

    def createItemFromLabel(self, label):
        if self.findItemByLabel(label):
            raise ValueError(f"Item for label '{label}' already exists")
        item = QtWidgets.QListWidgetItem()
        item.setData(QtCore.Qt.UserRole, label)
        return item

    def setItemLabel(self, item, label, color=None, shortcut_hint=None):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0)

        label_widget = QtWidgets.QLabel()
        label_text = html.escape(label)
        if color:
            label_text += ' <font color="# {:02x}{:02x}{:02x}">●</font>'.format(*color)
        label_widget.setText(label_text)
        layout.addWidget(label_widget, alignment=QtCore.Qt.AlignLeft)

        if shortcut_hint:
            shortcut_label = QtWidgets.QLabel()
            shortcut_label.setText(shortcut_hint)
            shortcut_label.setStyleSheet("color: gray;")
            layout.addWidget(shortcut_label, alignment=QtCore.Qt.AlignRight)

        widget.setLayout(layout)
        item.setSizeHint(widget.sizeHint())
        self.setItemWidget(item, widget)

    def updateShortcuts(self):
        """扫描列表，重新绑定 Ctrl+1~6 和 Alt+7~12 快捷键"""
        for sc in self.shortcuts:
            sc.disconnect()
            sc.setParent(None)
        self.shortcuts.clear()

        for i in range(self.count()):
            item = self.item(i)
            label = item.data(QtCore.Qt.UserRole)

            # 设置颜色（可选）
            if self.gender_color is not None:
                rgb = self.gender_color(label)
            else:
                rgb = None

            # 设置快捷键序列和显示文本
            if i < 6:
                key = f"Ctrl+{i + 1}"
                hint = f"Ctrl+{i + 1}"
            elif i < 12:
                key = f"Alt+{i + 1}"
                hint = f"Alt+{i + 1}"
            else:
                key = None
                hint = None

            self.setItemLabel(item, label, rgb, shortcut_hint=hint)

            if key:
                seq = QtGui.QKeySequence(key)
                sc = QtWidgets.QShortcut(seq, self.window())
                sc.activated.connect(lambda i=i: self.shortcut_callback(i) if self.shortcut_callback else None)
                self.shortcuts.append(sc)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 主部件和布局
        main_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(main_widget)

        # 标签列表
        self.list_widget = UniqueLabelQListWidget()
        layout.addWidget(self.list_widget)

        # 按钮
        btn_layout = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("添加标签")
        self.del_btn = QtWidgets.QPushButton("删除选中")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.del_btn)
        layout.addLayout(btn_layout)

        self.setCentralWidget(main_widget)

        # 初始化数据
        self.label_count = 0
        self.list_widget.setShortcutCallback(self.select_item)
        self.init_labels(["a", "b", "c", "d"])

        # 信号连接
        self.add_btn.clicked.connect(self.add_label)
        self.del_btn.clicked.connect(self.delete_label)

    def init_labels(self, labels):
        for label in labels:
            self.add_label(label)

    def add_label(self, label=None):
        if label is None:
            label = f"item_{self.label_count}"
        try:
            item = self.list_widget.createItemFromLabel(label)
        except ValueError:
            return
        self.list_widget.addItem(item)
        self.label_count += 1
        self.list_widget.updateShortcuts()

    def delete_label(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)
            self.list_widget.updateShortcuts()

    def select_item(self, index):
        if index < self.list_widget.count():
            self.list_widget.setCurrentRow(index)
            item = self.list_widget.item(index)
            label = item.data(QtCore.Qt.UserRole)
            print(f"Selected: {label}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.setWindowTitle("Label List with Shortcuts")
    win.resize(400, 300)
    win.show()
    sys.exit(app.exec_())
