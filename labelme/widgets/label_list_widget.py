from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QStyle
"""
多边形标签列表
"""

# https://stackoverflow.com/a/2039745/4158863
class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super(HTMLDelegate, self).__init__()
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        painter.save()

        options = QtWidgets.QStyleOptionViewItem(option)

        self.initStyleOption(options, index)
        self.doc.setHtml(options.text)
        options.text = ""

        style = (
            QtWidgets.QApplication.style()
            if options.widget is None
            else options.widget.style()
        )
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)  # type: ignore[attr-defined,union-attr]

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if option.state & QStyle.State_Selected:  # type: ignore[attr-defined]
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.HighlightedText),
            )
        else:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.Text),
            )

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)  # type: ignore[attr-defined,union-attr]

        if index.column() != 0:
            textRect.adjust(5, 0, 0, 0)

        thefuckyourshitup_constant = 4
        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - thefuckyourshitup_constant
        textRect.setTop(textRect.top() + margin)

        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)  # type: ignore[union-attr]

        painter.restore()

    def sizeHint(self, option, index):
        thefuckyourshitup_constant = 4
        return QtCore.QSize(
            int(self.doc.idealWidth()),
            int(self.doc.size().height() - thefuckyourshitup_constant),
        )


class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or "这是labellistweigetitem")
        self.setShape(shape)

        self.setCheckable(1)
        self.setCheckState(Qt.Checked)  # type: ignore[attr-defined]
        self.setEditable(0)
        self.setTextAlignment(Qt.AlignBottom)  # type: ignore[attr-defined]

    def clone(self):
        return LabelListWidgetItem(self.text(), self.shape())

    def setShape(self, shape):
        self.setData(shape, Qt.UserRole)  # type: ignore[attr-defined]

    def shape(self):
        return self.data(Qt.UserRole)  # type: ignore[attr-defined]

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class StandardItemModel(QtGui.QStandardItemModel):
    itemDropped = QtCore.pyqtSignal()

    def removeRows(self, *args, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        self.itemDropped.emit()
        return ret

# 自定义的列表视图控件，专门用于显示和管理标签项目
class LabelListWidget(QtWidgets.QListView):
    # 定义自定义信号：项目双击信号，参数为被双击的项目
    itemDoubleClicked = QtCore.pyqtSignal(LabelListWidgetItem)
    # 定义自定义信号：项目选择变化信号，参数为选中的项目列表和取消选中的项目列表
    itemSelectionChanged = QtCore.pyqtSignal(list, list)

    def __init__(self):
        super(LabelListWidget, self).__init__()
        # 初始化选中项目列表
        self._selectedItems = []
        # 设置窗口标志为独立窗口
        self.setWindowFlags(Qt.Window)  # type: ignore[attr-defined]
        # 设置数据模型为自定义的标准项模型
        self.setModel(StandardItemModel())
        # 设置项目原型（用于创建新项目的模板）
        self.model().setItemPrototype(LabelListWidgetItem())  # type: ignore[union-attr]
        # 设置项目委托（控制项目的显示和编辑）
        self.setItemDelegate(HTMLDelegate())
        # 设置选择模式为扩展选择（可以多选）
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # 设置拖放模式为内部移动
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        # 设置默认的拖放动作为移动
        self.setDefaultDropAction(Qt.MoveAction)  # type: ignore[attr-defined]
        # 连接双击事件到自定义处理函数
        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        # 连接选择变化事件到自定义处理函数，selectionModel 才是管理选中状态的对象
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)  # type: ignore[union-attr]

    # 获取列表长度（项目数量）
    def __len__(self):
        return self.model().rowCount()  # type: ignore[union-attr]

    # 通过索引获取项目（支持下标访问）
    def __getitem__(self, i):
        return self.model().item(i)  # type: ignore[union-attr]

    # 迭代器，支持for循环遍历
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # 属性：获取项目拖放完成信号，把 model 的信号“转发”出来，让 LabelListWidget 看起来像自己拥有这些信号一样
    @property
    def itemDropped(self):
        return self.model().itemDropped  # type: ignore[union-attr]

    # 属性：获取项目变化信号
    @property
    def itemChanged(self):
        return self.model().itemChanged  # type: ignore[union-attr]

    # 处理选择变化事件
    def itemSelectionChangedEvent(self, selected, deselected):
        # 将选中的索引转换为项目对象
        selected = [self.model().itemFromIndex(i) for i in selected.indexes()]  # type: ignore[union-attr]
        # 将取消选中的索引转换为项目对象
        deselected = [self.model().itemFromIndex(i) for i in deselected.indexes()]  # type: ignore[union-attr]
        # 发射自定义的选择变化信号
        self.itemSelectionChanged.emit(selected, deselected)

    # 处理双击事件
    def itemDoubleClickedEvent(self, index):
        # 发射自定义的双击信号，参数为被双击的项目
        self.itemDoubleClicked.emit(self.model().itemFromIndex(index))  # type: ignore[union-attr]

    # 返回已选中的项目
    def selectedItems(self):
        return [self.model().itemFromIndex(i) for i in self.selectedIndexes()]  # type: ignore[union-attr]

    # 滚动到指定项目
    def scrollToItem(self, item):
        self.scrollTo(self.model().indexFromItem(item))  # type: ignore[union-attr]

    # 添加项目到列表
    def addItem(self, item):
        # 类型检查，确保添加的是LabelListWidgetItem类型
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")
        # 在模型末尾添加项目
        self.model().setItem(self.model().rowCount(), 0, item)  # type: ignore[union-attr]
        # 设置项目的大小提示
        item.setSizeHint(self.itemDelegate().sizeHint(None, None))  # type: ignore[arg-type,union-attr]

    # 从列表中移除项目
    def removeItem(self, item):
        # 获取项目的索引
        index = self.model().indexFromItem(item)  # type: ignore[union-attr]
        # 从模型中移除项目所在的行
        self.model().removeRows(index.row(), 1)  # type: ignore[union-attr]

    # 选中指定项目
    def selectItem(self, item):
        # 获取项目的索引
        index = self.model().indexFromItem(item)  # type: ignore[union-attr]
        # 在选择模型中选中该项目
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)  # type: ignore[attr-defined,union-attr]

    # 根据形状对象查找对应的项目
    def findItemByShape(self, shape):
        # 遍历所有行
        for row in range(self.model().rowCount()):  # type: ignore[union-attr]
            # 获取该行的项目
            item = self.model().item(row, 0)  # type: ignore[union-attr]
            # 如果项目的形状匹配，返回该项目
            if item.shape() == shape:
                return item
        # 如果找不到，抛出异常
        raise ValueError("cannot find shape: {}".format(shape))

    # 清空列表
    def clear(self):
        self.model().clear()  # type: ignore[union-attr]
