# -*- encoding: utf-8 -*-

import html

from PyQt5 import QtWidgets,QtCore
from PyQt5.QtCore import Qt
from PyQt5 import  QtGui

# 导入自定义的可退出列表控件
from .escapable_qlist_widget import EscapableQListWidget
from functools import partial


# 定义唯一标签列表控件类，继承自可退出列表控件，专门用于显示和管理唯一标签
class UniqueLabelQListWidget(EscapableQListWidget):
    # 定义自定义信号：项目选择变化信号，参数为当前选中的项目
    itemSelectionChangedSignal  =QtCore.pyqtSignal(QtWidgets.QListWidgetItem)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化快捷键列表
        self.shortcuts = []
        # 初始化快捷键回调函数
        self.shortcut_callback = None
        # 初始化性别颜色函数
        self.gender_color = None
        # 设置拖放模式为内部移动
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        # 连接行移动信号到更新快捷键函数
        self.model().rowsMoved.connect(self.updateShortcuts)
        # 连接当前项目变化信号到自定义处理函数
        self.currentItemChanged.connect(self._emitCurrentItemChanged)  # 添加这一行

    # 处理当前项目变化的函数
    def _emitCurrentItemChanged(self, current, previous):
        # 如果有当前选中项目
        if current:
            # 获取项目的标签数据
            label = current.data(QtCore.Qt.UserRole)
            # 如果有性别颜色函数，获取颜色
            color = self.gender_color(label) if self.gender_color else None
            # 发射项目选择变化信号
            self.itemSelectionChangedSignal.emit(current)
    # 鼠标按下事件处理
    # def mousePressEvent(self, event):
    #     super(UniqueLabelQListWidget, self).mousePressEvent(event)
    #     if not self.indexAt(event.pos()).isValid():
    #            self.clearSelection()

    # 设置快捷键回调函数
    def setShortcutCallback(self, callback):
        """设置快捷键回调函数"""
        self.shortcut_callback = callback
        self.updateShortcuts()

    # 设置颜色函数
    def setcolor(self,gender_color):
        self.gender_color = gender_color

    # 更新函数（可能是拼写错误，应该是update）
    def undate(self):
        self.updateShortcuts()

    # 根据标签查找项目
    def findItemByLabel(self, label):
        # 遍历所有行
        for row in range(self.count()):
            # 获取该行的项目
            item = self.item(row)
            # 如果项目的标签数据匹配，返回该项目
            if item.data(Qt.UserRole) == label:
                return item
        # 如果找不到，返回None
        return None

    # 根据标签创建项目
    def createItemFromLabel(self, label):
        # 检查是否已存在相同标签的项目
        if self.findItemByLabel(label):
            raise ValueError(f"Item for label '{label}' already exists")
        # 创建新的列表项
        item = QtWidgets.QListWidgetItem()
        # 设置项目的标签数据
        item.setData(Qt.UserRole, label)
        return item

    # 设置项目的标签显示
    def setItemLabel(self, item, label, color=None, shortcut_hint=None):
        # 创建自定义显示控件
        widget = QtWidgets.QWidget()
        # 创建水平布局
        layout = QtWidgets.QHBoxLayout()
        # 设置布局边距
        layout.setContentsMargins(8, 0, 8, 0)

        # 创建标签显示控件
        label_widget = QtWidgets.QLabel()
        # 如果没有颜色，只显示文本
        if color is None:
            label_widget.setText("{}".format(label))
        # 如果有颜色，显示文本和彩色圆点
        else:
            label_widget.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(label), *color
                )
            )
        # 添加标签控件到布局
        # label_widget.setAlignment(Qt.AlignBottom)  # type: ignore[attr-defined]
        layout.addWidget(label_widget, alignment=Qt.AlignLeft)

        # 如果有快捷键提示，添加快捷键标签
        if shortcut_hint:
            shortcut_label = QtWidgets.QLabel()
            shortcut_label.setText(f"{shortcut_hint}")
            # 设置快捷键标签样式为灰色
            shortcut_label.setStyleSheet("color: gray;")
            layout.addWidget(shortcut_label, alignment=Qt.AlignRight)

        # 设置控件的布局
        widget.setLayout(layout)
        # 设置项目的大小提示
        item.setSizeHint(widget.sizeHint())
        # 设置项目的显示控件
        self.setItemWidget(item, widget)

    # 更新快捷键
    def updateShortcuts(self):
        """扫描列表，重新绑定 Ctrl+1~6 和 Alt+1-6 快捷键"""
        # 清除现有快捷键
        for sc in self.shortcuts:
            sc.disconnect()
            sc.setParent(None)
        self.shortcuts.clear()

        # 遍历所有项目
        for i in range(self.count()):
            # 获取项目
            item = self.item(i)
            # 获取项目的标签数据
            label = item.data(Qt.UserRole)
            # 如果有性别颜色函数，获取颜色
            if self.gender_color is not None:
                rgb = self.gender_color(label)
            else:
                rgb = None
            # 根据索引分配快捷键
            if i <6:
                key = f"Ctrl+{i + 1}"
                hint = f"Ctrl+{i + 1}"
            elif i <= 12 :
                key = f"Alt+{i -5}"
                hint = f"Alt+{i -5}"
            else:
                key = None
                hint = None

            self.setItemLabel(item, label, rgb, shortcut_hint=hint)
            # 如果有快捷键，创建并绑定
            if key:
                # 创建快捷键序列
                seq = QtGui.QKeySequence(key)
                # 创建快捷键（绑定到窗口）
                sc = QtWidgets.QShortcut(seq, self.window())

                # 这里 index=i 被“固化”进回调里，activated 不需要传参数
                if self.shortcut_callback is not None:
                    sc.activated.connect(partial(self.shortcut_callback, i))

                self.shortcuts.append(sc)

    # 如果列表不为空，选择第一个项目
    def selectFirstItemIfNotEmpty(self):
        if self.count() > 0:
            self.setCurrentRow(0)

    # 获取当前选中的项目
    def getCurrentItem(self):
        return self.currentItem()