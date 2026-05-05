import re

from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import labelme.utils

"""
确认标签弹出对话框
"""
# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.计算最佳位置，以避免超出屏幕区域


class LabelQLineEdit(QtWidgets.QLineEdit):
    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:  # type: ignore[attr-defined]
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)


class LabelDialog(QtWidgets.QDialog):
    def __init__(
            self,
            text="输入标签名称",  # 输入框的占位符文本
            parent=None,  # 父窗口
            labels=None,  # 可选的标签列表
            sort_labels=True,  # 是否对标签排序
            show_text_field=True,  # 是否显示文本输入框
            completion="startswith",  # 自动补全模式
            fit_to_content=None,  # 调整列表大小以适应内容
            flags=None,  # 标签标志选项
            shortcuts=None  # 快捷键设置
    ):
        # 设置内容适应选项默认为列适应
        if fit_to_content is None:
            fit_to_content = {"row": False, "column": True}
        self._fit_to_content = fit_to_content

        # 调用父类构造函数
        super(LabelDialog, self).__init__(parent)

        # 创建标签输入框，使用自定义的QLineEdit
        self.edit = LabelQLineEdit()
        # 设置输入框的占位符文本
        self.edit.setPlaceholderText(text)
        # 设置输入验证器，确保输入的标签有效
        self.edit.setValidator(labelme.utils.labelValidator())
        # 连接编辑完成信号到后处理函数
        self.edit.editingFinished.connect(self.postProcess)
        # 如果有标志选项，连接文本变化信号到更新标志函数
        if flags:
            self.edit.textChanged.connect(self.updateFlags)

        # 创建组ID输入框
        self.edit_group_id = QtWidgets.QLineEdit()
        self.edit_group_id.setPlaceholderText("组id")
        self.edit_group_id.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp(r"\d*"), None)
        )

        # 创建主布局
        layout = QtWidgets.QVBoxLayout()

        # 如果显示文本输入框，创建水平布局放置标签输入和组ID输入
        if show_text_field:
            layout_edit = QtWidgets.QHBoxLayout()
            # 添加标签输入框，权重为6
            layout_edit.addWidget(self.edit, 6)
            # 添加组ID输入框，权重为2
            layout_edit.addWidget(self.edit_group_id, 2)
            # 将水平布局添加到主布局
            layout.addLayout(layout_edit)

        # 创建按钮框（确定和取消按钮）
        self.buttonBox = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,  # type: ignore[attr-defined]
            self,
        )
        # 为确定按钮设置快捷键
        bb.button(bb.Ok).setShortcut(shortcuts['confirm_the_label'])
        # 为确定按钮设置图标
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon("done"))  # type: ignore[union-attr]
        # 为取消按钮设置图标
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon("undo"))  # type: ignore[union-attr]
        # 连接确定按钮的accepted信号到验证函数
        bb.accepted.connect(self.validate)
        # 连接取消按钮的rejected信号到拒绝函数
        bb.rejected.connect(self.reject)
        # 将按钮框添加到主布局
        layout.addWidget(bb)

        # 创建标签列表
        self.labelList = QtWidgets.QListWidget()
        # 如果启用行内容适应，隐藏水平滚动条
        if self._fit_to_content["row"]:
            self.labelList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        # 如果启用列内容适应，隐藏垂直滚动条
        if self._fit_to_content["column"]:
            self.labelList.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]

        # 设置标签排序选项
        self._sort_labels = sort_labels
        # 如果有预定义的标签，添加到列表
        if labels:
            self.labelList.addItems(labels)
        # 如果需要排序，对标签排序
        if self._sort_labels:
            self.labelList.sortItems()
        else:
            # 否则启用内部拖拽排序
            self.labelList.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        # 连接标签选择变化信号到处理函数
        self.labelList.currentItemChanged.connect(self.labelSelected)
        # 连接标签双击信号到处理函数
        self.labelList.itemDoubleClicked.connect(self.labelDoubleClicked)
        # 设置标签列表的固定高度
        self.labelList.setFixedHeight(150)
        # 将列表组件设置到输入框中（用于自动补全）
        self.edit.setListWidget(self.labelList)
        # 将标签列表添加到主布局
        layout.addWidget(self.labelList)

        # 标签标志选项处理
        if flags is None:
            flags = {}
        self._flags = flags
        # 创建标志选项布局
        self.flagsLayout = QtWidgets.QVBoxLayout()

        self.resetFlags()# 重置标志选项

        layout.addItem(self.flagsLayout)# 将标志布局添加到主布局

        self.edit.textChanged.connect(self.updateFlags)# 连接文本变化信号到更新标志函数


        self.editDescription = QtWidgets.QTextEdit() # 创建标签描述文本框

        self.editDescription.setPlaceholderText("标签描述")# 设置描述文本框的占位符文本

        self.editDescription.setFixedHeight(50)# 设置描述文本框的固定高度

        layout.addWidget(self.editDescription)# 将描述文本框添加到主布局

        # 设置对话框的主布局
        self.setLayout(layout)

        # 设置自动补全
        completer = QtWidgets.QCompleter()
        # 根据补全模式设置不同的补全方式
        if completion == "startswith":
            # 内联补全模式（在输入框内直接显示补全）
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)  # type: ignore[attr-defined]
            # 默认使用前缀匹配（注释掉了，系统默认就是这个）
            completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        elif completion == "contains":
            # 弹出式补全模式
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)  # type: ignore[attr-defined]
            # 设置包含匹配模式
            completer.setFilterMode(QtCore.Qt.MatchContains)  # type: ignore[attr-defined]
        else:
            # 不支持的补全模式则抛出错误
            raise ValueError("Unsupported completion: {}".format(completion))

        # 设置补全器的数据模型为标签列表的模型
        completer.setModel(self.labelList.model())
        # 为输入框设置补全器
        self.edit.setCompleter(completer)

    def addLabelHistory(self, label):
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):  # type: ignore[attr-defined]
            return
        self.labelList.addItem(label)
        if self._sort_labels:
            self.labelList.sortItems()

    def labelSelected(self, item):
        self.edit.setText(item.text())

    def validate(self):
        if not self.edit.isEnabled():
            self.accept()
            return

        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()  # type: ignore[attr-defined]
        if text:
            self.accept()

    def labelDoubleClicked(self, item):
        self.validate()

    def postProcess(self):
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()  # type: ignore[attr-defined]
        self.edit.setText(text)

    def updateFlags(self, label_new):
        # keep state of shared flags
        flags_old = self.getFlags()

        flags_new = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label_new):
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        self.setFlags(flags_new)

    def deleteFlags(self):
        for i in reversed(range(self.flagsLayout.count())):
            item = self.flagsLayout.itemAt(i).widget()  # type: ignore[union-attr]
            self.flagsLayout.removeWidget(item)
            item.setParent(None)  # type: ignore[union-attr]

    def resetFlags(self, label=""):
        flags = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label):
                for key in keys:
                    flags[key] = False
        self.setFlags(flags)

    def setFlags(self, flags):
        self.deleteFlags()
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flagsLayout.addWidget(item)
            item.show()

    def getFlags(self):
        flags = {}
        for i in range(self.flagsLayout.count()):
            item = self.flagsLayout.itemAt(i).widget()  # type: ignore[union-attr]
            flags[item.text()] = item.isChecked()  # type: ignore[union-attr]
        return flags

    def getGroupId(self):
        group_id = self.edit_group_id.text()
        if group_id:
            return int(group_id)
        return None

    def popUp(self, text=None, move=True, flags=None, group_id=None, description=None):
        if self._fit_to_content["row"]:
            self.labelList.setMinimumHeight(
                self.labelList.sizeHintForRow(0) * self.labelList.count() + 2
            )
        if self._fit_to_content["column"]:
            self.labelList.setMinimumWidth(self.labelList.sizeHintForColumn(0) + 2)
        # if text is None, the previous label in self.edit is kept
        if text is None:
            text = self.edit.text()
        # description is always initialized by empty text c.f., self.edit.text
        if description is None:
            description = ""
        self.editDescription.setPlainText(description)
        if flags:
            self.setFlags(flags)
        else:
            self.resetFlags(text)
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        if group_id is None:
            self.edit_group_id.clear()
        else:
            self.edit_group_id.setText(str(group_id))
        items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)  # type: ignore[attr-defined]
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.labelList.setCurrentItem(items[0])
            row = self.labelList.row(items[0])
            self.edit.completer().setCurrentRow(row)  # type: ignore[union-attr]
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)  # type: ignore[attr-defined]
        if move:
            self.move(QtGui.QCursor.pos())
        if self.exec_():
            return (
                self.edit.text(),
                self.getFlags(),
                self.getGroupId(),
                self.editDescription.toPlainText(),
            )
        else:
            return None, None, None, None
