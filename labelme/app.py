# -*- coding: utf-8 -*-

import functools
import html
import math
import os
import os.path as osp
import re
import shutil
import webbrowser

import imgviz
import natsort
import numpy as np
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from labelme import __appname__, shape
from labelme._automation import bbox_from_text
from labelme._automation import sam_patch
from labelme._automation import polygon_from_mask as _polygon_from_mask

# 在应用启动时立即打上补丁，确保所有 osam SAM 推理都使用可调阈值
sam_patch.apply_patches()
from labelme.config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.shape import Shape
from labelme.widgets import AiPromptWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
from labelme.widgets import Penwidget
from labelme import export_dataset
from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window
# 将最大缩放值设置为足够大的数值，以支持“适应宽度/窗口”

# TODO(unknown):
# - Zoom is too "steppy".缩放操作太“跳跃”，不够平滑

# 定义标签颜色映射，用于给不同标签自动分配颜色
LABEL_COLORMAP = imgviz.label_colormap()


class _PathColumnDelegate(QtWidgets.QStyledItemDelegate):
    """文件路径列专用：单元格只显示文件名，悬停显示完整路径，无省略号。"""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.textElideMode = Qt.ElideNone  # type: ignore[attr-defined]
        # 只渲染文件名，完整路径留给 tooltip
        full_path = index.data(Qt.DisplayRole)  # type: ignore[attr-defined]
        if full_path:
            option.text = osp.basename(full_path)


class _CheckBoxHeader(QtWidgets.QHeaderView):
    """在指定表头列放置一个真实的复选框控件。"""

    checkStateChanged = QtCore.pyqtSignal(int)

    def __init__(self, orientation, parent=None, checkable_section=1):
        super().__init__(orientation, parent)
        self._checkable_section = checkable_section
        self._check_state = Qt.Unchecked  # type: ignore[attr-defined]
        self.setSectionsClickable(True)
        self._checkbox = QtWidgets.QCheckBox(self.viewport())
        self._checkbox.setTristate(True)
        self._checkbox.setFocusPolicy(Qt.NoFocus)  # type: ignore[attr-defined]
        self._checkbox.setStyleSheet("background: transparent; margin: 0px;")
        self._checkbox.stateChanged.connect(self._on_checkbox_state_changed)
        self.sectionResized.connect(lambda *_: self._update_checkbox_geometry())
        self.sectionMoved.connect(lambda *_: self._update_checkbox_geometry())
        self.geometriesChanged.connect(self._update_checkbox_geometry)
        self._update_checkbox_geometry()

    def checkState(self):
        return self._check_state

    def setCheckState(self, state):
        if self._check_state == state:
            return
        self._check_state = state
        self._checkbox.blockSignals(True)
        self._checkbox.setCheckState(state)
        self._checkbox.blockSignals(False)
        self.updateSection(self._checkable_section)

    def _on_checkbox_state_changed(self, state):
        if state == Qt.PartiallyChecked:  # type: ignore[attr-defined]
            state = Qt.Checked  # type: ignore[attr-defined]
            self._checkbox.blockSignals(True)
            self._checkbox.setCheckState(state)
            self._checkbox.blockSignals(False)
        self._check_state = state
        self.checkStateChanged.emit(state)

    def _update_checkbox_geometry(self):
        x = self.sectionViewportPosition(self._checkable_section)
        width = self.sectionSize(self._checkable_section)
        size = self._checkbox.sizeHint()
        self._checkbox.setGeometry(
            x + (width - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )
        self._checkbox.raise_()
        self._checkbox.show()

    def mousePressEvent(self, event):
        if self.logicalIndexAt(event.pos()) == self._checkable_section:
            state = (
                Qt.Unchecked  # type: ignore[attr-defined]
                if self._check_state == Qt.Checked  # type: ignore[attr-defined]
                else Qt.Checked  # type: ignore[attr-defined]
            )
            self.setCheckState(state)
            self.checkStateChanged.emit(state)
            event.accept()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_checkbox_geometry()


class _AdaptiveSection(QtWidgets.QScrollArea):
    """
    工具栏区段容器：
      - 区段 宽于 按钮总宽：内容自动拉伸至区段宽度，按钮均匀填充（Expanding 策略）
      - 区段 窄于 按钮总宽：内容保持自然宽度，viewport 裁剪右侧溢出（按钮不压缩）
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._natural_width: int = 0
        self.setFrameShape(QtWidgets.QFrame.NoFrame)  # type: ignore[attr-defined]
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setWidgetResizable(False)
        self.setStyleSheet("background: transparent; border: none;")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        w = self.widget()
        if w is None:
            return
        # 首次 resize 时确定按钮自然总宽（sizeHint 在此时已可靠）
        if self._natural_width == 0:
            self._natural_width = w.sizeHint().width()
        if self._natural_width > 0:
            target = max(self.viewport().width(), self._natural_width)
            if w.width() != target:
                w.setFixedWidth(target)


# 主窗口类，继承自QtWidgets.QMainWindow
class MainWindow(QtWidgets.QMainWindow):
    '''
    三个变量定义了图像显示的缩放模式：
    FIT_WINDOW：适应窗口大小。
    FIT_WIDTH：适应窗口的宽度。
    MANUAL_ZOOM：手动缩放。
    '''
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    # --- 初始化函数 ---
    def __init__(
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,

    ):
        # 处理已弃用的 output 参数
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output
        # 加载配置，如果未提供则从默认文件加载
        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # 根据配置设置标注图形(Shape)的默认颜色
        # set default shape colors
        # 设置形状轮廓线的颜色，从配置中读取并转换为QColor对象
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])  # type: ignore[assignment]

        # 设置形状填充颜色，从配置中读取并转换为QColor对象
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])  # type: ignore[assignment]
        # 设置选中状态下形状轮廓线的颜色
        Shape.select_line_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["select_line_color"]
        )
        # 设置选中状态下形状的填充颜色
        Shape.select_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["select_fill_color"]
        )
        # 设置顶点（控制点）的填充颜色
        Shape.vertex_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["vertex_fill_color"]
        )
        # 设置高亮顶点（hover状态下的控制点）的填充颜色
        Shape.hvertex_fill_color = QtGui.QColor(  # type: ignore[assignment]
            *self._config["shape"]["hvertex_fill_color"]
        )
        # 从配置文件设置顶点的大小
        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        # 调用父类的构造函数
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)  # 设置窗口标题

        '''
        dirty：记录是否有未保存的更改。
        _noSelectionSlot：用于防止触发不必要的信号槽。
        _copied_shapes：用于复制粘贴标注。
        '''
        # Whether we need to save or not.
        self.dirty = False  # 文件是否被修改的标志

        # 用于判断是否执行前一张后一张操作
        self._openNextImg = False
        self._openPrevImg = False
        self._openDir = False

        self._noSelectionSlot = False

        self._copied_shapes = None

        # --- 创建核心UI组件 ---

        # 标签对话框 (LabelDialog)，用于新建或编辑标签时弹出
        # Main widgets and related state.

        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],#是否显示文本输入框
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
            shortcuts=self._config["shortcuts"])

        # 已标注图形列表 (LabelListWidget)，显示当前图片上所有已标注的多边形
        #标签列表（LabelListWidget）
        self.labelList = LabelListWidget()
        self.lastOpenDir = None  # 记录上一次打开的目录

        # 标志位面板 (Flag Dock)，用于给整个图像设置标志（例如，'is_blurry', 'has_occlusion'）
        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.loadFlags({k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)  # 当标志状态改变时，将文件标记为已修改
        # ---  信号与槽的连接 (UI组件交互逻辑) ---
        # 当多边形列表中的选中项改变时，触发 labelSelectionChanged 函数
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        # 当双击列表项时，触发 _edit_label 函数进行编辑
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        # 当列表项（如可见性复选框）改变时，触发 labelItemChanged 函数
        self.labelList.itemChanged.connect(self.labelItemChanged)
        # 当拖拽排序时，触发 labelOrderChanged 函数
        self.labelList.itemDropped.connect(self.labelOrderChanged)

        # 创建容纳多边形列表的停靠窗口
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        #唯一标签面板（UniqueLabelQListWidget），显示所有可用的标签类别
        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setcolor(self._get_rgb_by_label)  # 设置颜色获取回调
        self.uniqLabelList.setShortcutCallback(self.select_item)  # 设置快捷键回调
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. " "Press 'Esc' to deselect."
            )
        )
        # 加载配置文件中的预定义标签
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                # rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, None)
        self.uniqLabelList.undate() # 更新列表显示


        self.uniqLabelList.selectFirstItemIfNotEmpty() # 默认选中第一项
        self.uniqLabelList.itemSelectionChangedSignal.connect(self.emitCurrentItemChanged) # 选中项改变时发射信号

        # 创建容纳标签列表的停靠窗口
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        #文件列表面板
        self.fileSearch = QtWidgets.QLineEdit() # 文件搜索框
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged) # 文本改变时触发搜索
        # 文件列表改用 QTableWidget，五列：编号、选中复选框、文件路径、标注状态、已有标签
        self.fileListWidget = QtWidgets.QTableWidget()
        self._updating_file_checks = False
        self.fileListHeader = _CheckBoxHeader(
            QtCore.Qt.Horizontal, self.fileListWidget, checkable_section=1
        )
        self.fileListWidget.setHorizontalHeader(self.fileListHeader)
        self.fileListWidget.setColumnCount(5)
        self.fileListWidget.setHorizontalHeaderLabels([
            self.tr("id"),      # 编号
            self.tr(""),       # 复选框列
            self.tr("文件路径"),
            self.tr("标注状态"),
            self.tr("已有标签"),
        ])
        # 列宽模式：编号/复选框固定，路径和标注状态可手动拖动，已有标签自动填满
        header = self.fileListWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)       # 编号：固定
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)       # 复选框：固定
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)  # 文件路径：可拖动
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Interactive)  # 标注状态：可拖动
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)      # 已有标签：填满剩余
        header.setStretchLastSection(True)
        self.fileListWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.fileListWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fileListWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.fileListWidget.verticalHeader().setVisible(False)
        self.fileListWidget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        # 各列默认宽度
        self.fileListWidget.setColumnWidth(0, 32)   # 编号（固定）
        self.fileListWidget.setColumnWidth(1, 36)   # 复选框（固定）
        self.fileListWidget.setColumnWidth(2, 160)  # 文件路径（可拖动）
        self.fileListWidget.setColumnWidth(3, 56)   # 标注状态（可拖动）
        # 第4列由 Stretch 自动填满，不需要手动设置
        # 文件路径列不显示省略号，有多宽就显示多少字符
        self.fileListWidget.setItemDelegateForColumn(2, _PathColumnDelegate(self.fileListWidget))
        # 安装事件过滤器，处理 Ctrl+C 复制完整路径
        self.fileListWidget.installEventFilter(self)
        # 缩小字号，让路径尽量显示更多字符
        table_font = self.fileListWidget.font()
        table_font.setPointSize(max(8, table_font.pointSize() - 2))
        self.fileListWidget.setFont(table_font)
        # 行高也随之紧凑
        self.fileListWidget.verticalHeader().setDefaultSectionSize(20)
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        self.fileListWidget.itemChanged.connect(self._update_file_header_check_state)
        self.fileListHeader.checkStateChanged.connect(self._set_all_file_checks)

        # 创建一个垂直布局管理器（QVBoxLayout）
        fileListLayout = QtWidgets.QVBoxLayout()# 垂直布局会按从上到下的顺序排列子控件
        fileListLayout.setContentsMargins(0, 0, 0, 0)# 设置布局的外边距为0（左、上、右、下都没有外边距）
        fileListLayout.setSpacing(0)# 设置布局内子控件之间的间距为0
        fileListLayout.addWidget(self.fileSearch)# 将文件搜索框控件添加到布局中

        # 表头（放在搜索框下面，即现在文件列表从上到下是搜索框，表头，文件）
        fileListLayout.addWidget(self.fileListWidget)
        # 底部计数标签（左对齐）
        self.fileCountLabel = QtWidgets.QLabel(self.tr("共 0 张"))
        self.fileCountLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)  # type: ignore[attr-defined]
        count_font = self.fileCountLabel.font()
        count_font.setPointSize(max(8, count_font.pointSize() - 1))
        self.fileCountLabel.setFont(count_font)
        self.fileCountLabel.setContentsMargins(4, 2, 0, 2)
        fileListLayout.addWidget(self.fileCountLabel)
        # 创建容纳文件列表的停靠窗口
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)  # 布局
        self.file_dock.setWidget(fileListWidget)

        #缩放控件和画笔
        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)
        # 单独的三个宽度控件
        self.penWidget = Penwidget(value=20)  # 画笔宽度
        self.eraserWidget = Penwidget(value=10)  # 橡皮宽度
        self.polygonWidget = Penwidget(value=Shape.PEN_WIDTH)  # 多边形等图形的线宽

        # 创建画布 (Canvas)，并进行配置
        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        # 画布信号连接
        self.canvas.zoomBrushRequest.connect(self.zoomBrushRequest)  # 画笔缩放请求
        self.canvas.zoomRequest.connect(self.zoomRequest) # 画布缩放请求
        self.canvas.mouseMoved.connect(
            lambda pos: self.status(f"Mouse is at: x={pos.x()}, y={pos.y()}") # 鼠标移动时在状态栏显示坐标
        )
        self.emitCurrentItemChanged(self.uniqLabelList.getCurrentItem()) # 初始化当前标签

        # 创建中心滚动区域，将画布放入其中
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = { # 获取水平和垂直滚动条
            Qt.Vertical: scrollArea.verticalScrollBar(),  # type: ignore[attr-defined]
            Qt.Horizontal: scrollArea.horizontalScrollBar(),  # type: ignore[attr-defined]
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)  # 滚动请求
        # 更多画布信号连接
        self.canvas.newShape.connect(self.newShape)  # 创建新图形
        self.canvas.shapeMoved.connect(self.setDirty)  # 移动图形
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)  # 选中图形
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)  # 正在绘制时
        self.canvas.contact_del_shape.connect(self.remLabels)  # 删除图形
        self.canvas.setDirty.connect(self.setDirty)  # 标记为已修改
        self.canvas.rotationModeChanged.connect(self._onRotationModeChanged)  # 旋转模式切换

        # 将滚动区域设置为主窗口的中心控件
        self.setCentralWidget(scrollArea)

        # ---  配置和添加停靠窗口 (Dock Widgets) ---

        # 根据配置文件设置每个停靠窗口的特性（可关闭、可浮动、可移动）
        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        #添加侧边栏 Dock
        # 将所有停靠窗口添加到主窗口的右侧区域
        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)  # type: ignore[attr-defined]
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)  # type: ignore[attr-defined]
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)  # type: ignore[attr-defined]
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)  # type: ignore[attr-defined]

        # --- 5. 创建所有操作 (Actions) ---
        # Action是菜单项、工具栏按钮的抽象表示
        # Actions
        action = functools.partial(utils.newAction, self) # 创建一个偏函数，简化Action的创建
        shortcuts = self._config["shortcuts"]



        # 文件操作
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        # 打开图片
        open_ = action(
            self.tr("&Open\n"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )
        # 打开目录
        opendir = action(
            self.tr("Open Dir"),
            self.openDirDialog,
            shortcuts["open_dir"],
            "file",
            self.tr("Open Dir"),
        )
        # 下一幅
        openNextImg = action(
            self.tr("&Next Image"),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        # 上一幅
        openPrevImg = action(
            self.tr("&Prev Image"),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        # 保存
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        # 删除标签文件
        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )
        #  清空所有按钮动作
        clearAll = action(
            self.tr("Clear All"),
            self.clearAllShapes,
            None,
            "reject",  # 使用叉号图标
            self.tr("Clear all shapes and brush strokes"),
            enabled=False,
        )
        # 自动保存：启用时图标+文字正常显示，关闭时整体变灰（利用 Qt setEnabled 原生效果）
        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=None,
            tip=self.tr("Auto-save labels when annotating"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setIcon(utils.newIcon("save"))
        saveAuto.setChecked(self._config["auto_save"])
        saveAuto.setEnabled(self._config["auto_save"])

        def _on_save_auto_toggled(checked):
            # checked=True  → 正常显示（可见、可用）
            # checked=False → Qt 原生灰显（图标+文字同步变灰）
            saveAuto.setEnabled(checked)

        saveAuto.toggled.connect(_on_save_auto_toggled)

        # 同时保存图像数据
        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )
        # 关闭
        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            "close",
            self.tr("Close current file"),
        )
        # --编辑模型--
        # 标签自动复制到下一张，"保留最后的标注"
        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        # 创建多边形
        createMode = action(
            self.tr("Create Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
           "Polygons",
            self.tr("Start drawing polygons"),
            enabled=False,
            checkable=True,
        )
        # 刷子
        createBrush = action(
            self.tr("Create brush"),
            lambda: self.toggleDrawMode(False, createMode="pen"),
            None,
            "brush",
            self.tr("Start drawing brush"),
            enabled=False,
            checkable=True,
        )
        # 橡皮
        createEraser = action(
            self.tr("Create eraser"),
            lambda: self.toggleDrawMode(False, createMode="eraser"),
            None,
            "eraser",
            self.tr("Start drawing eraser"),
            enabled=False,
        )
        # 旋转标注（R 键激活旋转模式）
        rotateShape = action(
            self.tr("旋转标注"),
            self.canvas.toggleRotationMode,
            None,
            "rotate",  # 复用内置图标，无专属旋转图标时用此占位
            self.tr("进入旋转模式 (R)：左键拖拽以围绕质心旋转选中的标注"),
            enabled=False,
            checkable=True,
        )
        # 导出 COCO 数据集
        exportCoco = action(
            self.tr("Export COCO"),
            self._export_checked_to_coco,
            None,
            "save",
            self.tr("Export checked images to COCO dataset"),
            enabled=True,
        )
        # 导出 VOC 数据集
        exportVoc = action(
            self.tr("Export VOC"),
            self._export_checked_to_voc,
            None,
            "save",
            self.tr("Export checked images to VOC segmentation dataset"),
            enabled=True,
        )
        # 创建矩形
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "Rectangle",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        # 创建圆形
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "Circle",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        # 创建直线
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "Line",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        # 创建控制点
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "Point",
            self.tr("Start drawing points"),
            enabled=False,
        )
        # 创建折现
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "LineStrip",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )

        # 创建ai多边形
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "AI",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        # 创建ai蒙版
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self.toggleDrawMode(False, createMode="ai_mask"),
            None,
            "mb",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )

        # 创建ai标注框
        createAiBboxMode = action(
            self.tr("创建AI多边形(框提示)"),
            lambda: self.toggleDrawMode(False, createMode="ai_bbox"),
            None,
            "fk",
            self.tr("Drag to draw a bounding box and get AI mask annotation."),
            enabled=False,
        )

        # 编辑模式
        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        # 删除多边形
        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        # 复制多边形
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        # 复制多边形
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        # 粘贴多边形
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        # 撤销最后的控制点
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        # 移除选中的控制点
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )
        # 撤销
        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        # --视图模式--
        # 隐藏多边形
        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        # 显示多边形
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        # 开关多边形
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        # 帮助
        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="help",
            tip=self.tr("Show tutorial page"),
        )

        # --- 6. 创建特殊UI组件 (QWidgetAction) ---

        # 缩放控件
        zoom = QtWidgets.QWidgetAction(self)  # 缩放控件的容器
        zoomBoxLayout = QtWidgets.QVBoxLayout()  # 垂直布局
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        # 文本标签水平居中对齐
        zoomLabel.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        zoomBoxLayout.addWidget(self.zoomWidget)  # 将缩放控件（滑块）添加到垂直布局中
        zoomBoxLayout.addWidget(zoomLabel)  # 将文本标签也添加到垂直布局中
        zoom.setDefaultWidget(QtWidgets.QWidget())  # 设置一个默认的空白容器控件
        # 将设计好的布局应用到这个容器控件上
        zoom.defaultWidget().setLayout(zoomBoxLayout)  # type: ignore[union-attr]
        # self.tr()用于支持多语言翻译。设置“这是什么？”(What's This?) 帮助文本。
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        # 初始时禁用缩放控件
        self.zoomWidget.setEnabled(False)


        # 画笔宽度控件（只控制画笔）
        pen = QtWidgets.QWidgetAction(self)
        penBoxLayout = QtWidgets.QVBoxLayout()  # 垂直布局
        penLabel = QtWidgets.QLabel(self.tr("Brush width"))  # 画笔宽度标签
        # penLabel.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        penBoxLayout.addWidget(self.penWidget) # 将画笔宽度控件放入布局
        penBoxLayout.addWidget(penLabel) # 把标签放入布局
        pen.setDefaultWidget(QtWidgets.QWidget())  # 设置一个默认的空白容器控件
        pen.defaultWidget().setLayout(penBoxLayout)  # type: ignore[union-attr]
        # 初始时可以使用画笔宽度控件
        self.penWidget.setEnabled(True)

        # 橡皮宽度控件（只控制橡皮）
        eraser = QtWidgets.QWidgetAction(self)
        eraserBoxLayout = QtWidgets.QVBoxLayout()
        eraserLabel = QtWidgets.QLabel(self.tr("Eraser width"))
        eraserBoxLayout.addWidget(self.eraserWidget)
        eraserBoxLayout.addWidget(eraserLabel)
        eraser.setDefaultWidget(QtWidgets.QWidget())
        eraser.defaultWidget().setLayout(eraserBoxLayout)  # type: ignore[union-attr]
        self.eraserWidget.setEnabled(True)

        # 多边形 / 线条 / 矩形等图形的线宽控件
        poly = QtWidgets.QWidgetAction(self)
        polyBoxLayout = QtWidgets.QVBoxLayout()
        polyLabel = QtWidgets.QLabel(self.tr("Shape width"))
        polyBoxLayout.addWidget(self.polygonWidget)
        polyBoxLayout.addWidget(polyLabel)
        poly.setDefaultWidget(QtWidgets.QWidget())
        poly.defaultWidget().setLayout(polyBoxLayout)  # type: ignore[union-attr]
        self.polygonWidget.setEnabled(True)

        # 缩放操作
        # 缩小
        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        # 放大
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        # 原始大小
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        # 保存最后的比例
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        # 适应窗口
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        # 适应宽度
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )

        # 图像调整
        # 亮度 对比度
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "color",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )

        # 将缩放相关的Action分组，方便统一管理
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)  # type: ignore[attr-defined]
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        # 多边形标签（修改选中的多边形的标签）
        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        # 填充所绘制多边形
        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()


        # --- 7. 创建菜单和工具栏 ---

        # 多边形标签列表的右键菜单
        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # 将所有Action存储到一个结构体中，方便管理和访问
        # Store actions for further handling.
        self.actions = utils.struct(  # type: ignore[assignment,method-assign]
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            save=save,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            clearAll=clearAll,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createBrush=createBrush,
            createEraser=createEraser,
            rotateShape=rotateShape,
            exportCoco=exportCoco,
            exportVoc=exportVoc,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            createAiBboxMode=createAiBboxMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            pen=pen,
            eraserWidth=eraser,  # 橡皮宽度控件
            shapeWidth=poly,  # 多边形/其他图形宽度控件
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            fileMenuActions=(open_, opendir, save, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                copy,
                paste,
                delete,
                None,
                rotateShape,
                None,
                undo,
                undoLastPoint,
                None,
                removePoint,
                None,
                toggle_keep_prev_mode,
            ),
            # menu shown at right click
            menu=(
                createMode,
                createBrush,
                createEraser,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                # createAiPolygonMode,
                # createAiMaskMode,
                editMode,
                None,
                rotateShape,
                None,
                edit,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createBrush,
                createEraser,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                # createAiPolygonMode,
                # createAiMaskMode,
                editMode,
                brightnessContrast,
                clearAll,
            ),
            onShapesPresent=(hideAll, showAll, toggleAll),
        )
        # 移除选中的点
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)  # type: ignore[attr-defined]
        #画刷信号连接光标变化
        # 修改信号连接，明确传递参数
        '''self.actions.createBrush.toggled.connect(
            lambda checked: self.canvas.toggleEditMode(checked)
        )
        self.actions.createEraser.toggled.connect(
            lambda checked: self.canvas.toggleEditMode(checked)
        )

        # 调试信号
        self.actions.createBrush.toggled.connect(
            lambda checked: print(f"createBrush toggled: {checked}")
        )
        self.actions.createEraser.toggled.connect(
            lambda checked: print(f"createEraser toggled: {checked}")
        )'''
        # 创建顶级菜单
        self.menus = utils.struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            tools=self.menu(self.tr("&Tools")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        # 向菜单中添加Actions
        utils.addActions(
            self.menus.file,  # type: ignore[attr-defined]
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,  # type: ignore[attr-defined]
                save,
                saveAuto,
                saveWithImageData,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help,))  # type: ignore[attr-defined]
        utils.addActions(
            self.menus.view,  # type: ignore[attr-defined]
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,

            ),
        )
        # Tools 菜单：导出数据集
        utils.addActions(
            self.menus.tools,  # type: ignore[attr-defined]
            (
                exportCoco,
                exportVoc,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)  # type: ignore[attr-defined]
        # 允许点击灰显的 saveAuto 菜单项来重新启用自动保存
        self.menus.file.installEventFilter(self)  # type: ignore[attr-defined]

        # 为画布添加右键菜单
        # Custom context menu for the canvas widget:
        # 为画布的第一个菜单添加一组预定义的通用操作
        utils.addActions(self.canvas.menus[0], self.actions.menu)  # type: ignore[attr-defined]
        # 为画布的第二个菜单添加一组特定于“移动 / 复制”的操作
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        # AI模型选择下拉框
        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())  # type: ignore[union-attr]

        selectAiModelLabel = QtWidgets.QLabel(self.tr("AI Mask Model"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)  # type: ignore[attr-defined]
        selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)  # type: ignore[union-attr]

        self._selectAiModelComboBox = QtWidgets.QComboBox()
        selectAiModel.defaultWidget().layout().addWidget(self._selectAiModelComboBox)  # type: ignore[union-attr]
        MODEL_NAMES: list[tuple[str, str]] = [
            ("efficientsam:10m", "EfficientSam (speed)"),
            ("efficientsam:latest", "EfficientSam (accuracy)"),
            ("sam:100m", "SegmentAnything (speed)"),
            ("sam:300m", "SegmentAnything (balanced)"),
            ("sam:latest", "SegmentAnything (accuracy)"),
            ("sam2:small", "Sam2 (speed)"),
            ("sam2:latest", "Sam2 (balanced)"),
            ("sam2:large", "Sam2 (accuracy)"),
        ]
        for model_name, model_ui_name in MODEL_NAMES:
            self._selectAiModelComboBox.addItem(model_ui_name, userData=model_name)
        model_ui_names: list[str] = [model_ui_name for _, model_ui_name in MODEL_NAMES]
        if self._config["ai"]["default"] in model_ui_names:
            model_index = model_ui_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.setCurrentIndex(model_index)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda index: self.canvas.initializeAiModel(
                model_name=self._selectAiModelComboBox.itemData(index)
            )
            if self.canvas.createMode in ["ai_polygon", "ai_mask"]
            else None
        )

        # AI文本提示输入框
        self._ai_prompt_widget: QtWidgets.QWidget = AiPromptWidget(
            on_submit=self._submit_ai_prompt, parent=self
        )
        ai_prompt_action = QtWidgets.QWidgetAction(self)
        ai_prompt_action.setDefaultWidget(self._ai_prompt_widget)

        # 创建并填充工具栏
        self.tools = self.toolbar("Tools")

        # 创建参数调节滑块（在 populateModeActions 之前构建，避免重复创建）
        self._logit_container, self._logit_slider, self._logit_value_lbl = (
            self._create_param_slider(
                name="像素阈值",
                min_val=-5.0, max_val=5.0,
                default_val=0.0, step=0.1,
                fmt="{:+.1f}",
                callback=sam_patch.set_logit_threshold,
            )
        )
        self._rdp_container, self._rdp_slider, self._rdp_value_lbl = (
            self._create_param_slider(
                name="多边形精度",
                min_val=0.001, max_val=0.030,
                default_val=0.004, step=0.001,
                fmt="{:.3f}",
                callback=_polygon_from_mask.set_rdp_factor,
            )
        )
        # self.actions.tool = (  # type: ignore[attr-defined]
        #     open_,
        #     opendir,
        #     openPrevImg,
        #     openNextImg,
        #     save,
        #     deleteFile,
        #     None,
        #     createMode,
        #     editMode,
        #     duplicate,
        #     delete,
        #     undo,
        #     brightnessContrast,
        #     None,
        #     fitWindow,
        #     zoom,
        #     None,
        #     selectAiModel,
        #     None,
        #     ai_prompt_action,
        # )
        self.actions.tool = (  # type: ignore[attr-defined]
            open_,
            opendir,
            openPrevImg,
            openNextImg,
            save,
            # deleteFile,
            clearAll,
            None,
            editMode,
            createMode,
            createBrush,
            createEraser,
            createRectangleMode,
            # createPointMode,
            # createLineMode,
            # createLineStripMode,
            # createCircleMode,
            # createAiMaskMode,
            # createAiPolygonMode,
            None,
            brightnessContrast,
            fitWindow,
            None,
            # ai_prompt_action,
        )

        # --- 初始化应用程序状态和设置 ---
        # 在状态栏中显示程序启动提示。
        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)  # type: ignore[union-attr]
        self.statusBar().show()  # type: ignore[union-attr]

        # 处理输出文件和目录
        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file  # 如果开启自动保存，则会忽略该保存路径
        self.output_dir = output_dir  # 如果设置自动保存，将优先使用output_dir参数

        # 初始化应用程序状态变量
        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []  # type: ignore[var-annotated]
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        '''        
        imagePath 当前图片路径。
        recentFiles 最近打开的文件列表。
        zoom_level 当前缩放倍数。
        fit_window 是否适配窗口。
        '''
        '''
        分别存储每张图像的缩放、亮度对比度、滚动条位置。
        '''
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {  # type: ignore[var-annotated]
            Qt.Horizontal: {},  # type: ignore[attr-defined]
            Qt.Vertical: {},  # type: ignore[attr-defined]
        }  # key=filename, value=scroll_value

        # 处理启动时传入的文件名或目录
        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename
            # 启动时传入单张图片，也添加到文件列表
            if filename is not None and osp.isfile(filename):
                self._add_file_row_to_table(filename)

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # 恢复应用程序上一次关闭时的设置（窗口大小、位置等）
        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)
        '''
        加载菜单项、准备图像加载任务。
        '''
        # Populate the File menu dynamically.
        self.updateFileMenu() # 动态填充文件菜单
        # Since loading the file may take some time,
        # make sure it runs in the background.
        # 使用事件队列在后台加载文件，避免UI阻塞
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # 最终的回调连接
        # 当缩放控件变化时，重新绘制画布。
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        # 三个独立的宽度控件
        self.penWidget.valueChanged.connect(self.onBrushWidthChanged)
        self.eraserWidget.valueChanged.connect(self.onEraserWidthChanged)
        self.polygonWidget.valueChanged.connect(self.zoomBrush_)

        # 添加自定义的控件进入主窗口
        self.populateModeActions()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()

    # --- 辅助函数 ---
    def menu(self, title, actions=None):
        """创建一个顶级菜单并返回"""
        menu = self.menuBar().addMenu(title)  # type: ignore[union-attr]
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        """创建一个顶级菜单并返回"""
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  # type: ignore[attr-defined]
        toolbar.setIconSize(QtCore.QSize(36, 36))
        small_font = toolbar.font()
        small_font.setPointSize(9)
        toolbar.setFont(small_font)
        if actions:
            utils.addActions(toolbar, actions)
        # 将工具栏添加到主窗口
        # self.addToolBar() 是 QMainWindow 的一个标准方法。
        # Qt.TopToolBarArea 指定了工具栏的停靠区域在窗口的顶部。
        toolbar.setFixedHeight(100)
        self.addToolBar(Qt.TopToolBarArea, toolbar)  # type: ignore[attr-defined]
        return toolbar

    def _create_param_slider(self, name, min_val, max_val, default_val, step, fmt, callback):
        """
        创建一个供工具栏 QSplitter 区段使用的参数滑块控件。
        返回 (QWidget_container, QSlider, QLabel_value) 三元组。

        布局（垂直，紧凑）：
            [  名称标签  ]
            [slider ──●──] [当前值]
        """
        container = QtWidgets.QWidget()
        v_layout = QtWidgets.QVBoxLayout(container)
        v_layout.setContentsMargins(6, 2, 6, 2)
        v_layout.setSpacing(1)

        small_font = QtGui.QFont()
        small_font.setPointSize(7)

        name_lbl = QtWidgets.QLabel(name)
        name_lbl.setAlignment(QtCore.Qt.AlignCenter)  # type: ignore[attr-defined]
        name_lbl.setFont(small_font)

        row = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(row)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # type: ignore[attr-defined]
        int_min = round(min_val / step)
        int_max = round(max_val / step)
        int_default = round(default_val / step)
        slider.setMinimum(int_min)
        slider.setMaximum(int_max)
        slider.setValue(int_default)
        slider.setFixedWidth(90)
        slider.setFixedHeight(16)

        value_lbl = QtWidgets.QLabel(fmt.format(default_val))
        value_lbl.setFont(small_font)
        value_lbl.setFixedWidth(50)  # 足够显示 "+0.050" / "0.030" 完整数值
        value_lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)  # type: ignore[attr-defined]

        h_layout.addWidget(slider)
        h_layout.addWidget(value_lbl)

        v_layout.addWidget(name_lbl)
        v_layout.addWidget(row)

        def on_change(int_val, _step=step, _fmt=fmt, _cb=callback, _lbl=value_lbl):
            real_val = int_val * _step
            _lbl.setText(_fmt.format(real_val))
            _cb(real_val)

        slider.valueChanged.connect(on_change)

        return container, slider, value_lbl

    def _build_toolbar_splitter(self, sections: list) -> QtWidgets.QSplitter:
        """
        将各区段封装进横向 QSplitter：
          - splitter 撑满工具栏全部宽度，随窗口伸缩
          - 手柄可拖拽，动态重新分配各区段宽度
          - 工具按钮区段用 QScrollArea（禁滚动条）：区段变窄时右侧按钮被
            viewport 裁剪消失，变宽时重新出现；按钮始终保持原始尺寸不压缩
          - 参数调节区段用普通 QWidget，滑块随区段宽度自然伸缩

        sections: List[List[QAction | QWidget]]
            最后一个 section 被视为参数调节区（元素为 QWidget）。
        """
        class _FlexSection(QtWidgets.QScrollArea):
            """宽于内容时按钮均匀填满；窄于内容时右侧裁剪。"""
            def resizeEvent(self_s, event):  # noqa: N805
                super(_FlexSection, self_s).resizeEvent(event)
                w = self_s.widget()
                if w is None or w.layout() is None:
                    return
                vw = self_s.viewport().width()
                natural_w = w.layout().sizeHint().width()
                w.setFixedWidth(max(vw, natural_w))

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)  # type: ignore[attr-defined]
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,    # type: ignore[attr-defined]
            QtWidgets.QSizePolicy.Preferred,    # type: ignore[attr-defined]
        )
        splitter.setStyleSheet(
            "QSplitter { background: transparent; }"
            "QSplitter::handle { background: palette(mid); }"
            "QSplitter::handle:hover { background: palette(highlight); }"
        )

        btn_font = QtGui.QFont()
        btn_font.setPointSize(9)

        for i, section_items in enumerate(sections):
            is_param_section = (i == len(sections) - 1)

            if is_param_section:
                # ── 参数调节区：左对齐，尾部 stretch 防止控件撑满整个区段 ──
                container = QtWidgets.QWidget()
                container.setStyleSheet("background: transparent;")
                container.setMinimumWidth(20)
                layout = QtWidgets.QHBoxLayout(container)
                layout.setSpacing(8)
                layout.setContentsMargins(4, 0, 4, 0)
                layout.setAlignment(  # type: ignore[call-overload]
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter  # type: ignore[attr-defined]
                )
                for item in section_items:
                    if isinstance(item, QtWidgets.QWidget):
                        layout.addWidget(item)
                        # QWidgetAction.setDefaultWidget() 内部会显式 hide() 控件，
                        # layout.addWidget 不覆盖显式隐藏标志，必须手动 show()
                        item.show()
                layout.addStretch(1)   # 剩余空间留白，控件靠左堆叠
                splitter.addWidget(container)

            else:
                # ── 工具按钮区：_AdaptiveSection 负责"宽时拉伸、窄时裁剪" ──
                content = QtWidgets.QWidget()
                content.setStyleSheet("background: transparent;")
                content_layout = QtWidgets.QHBoxLayout(content)
                content_layout.setSpacing(0)
                content_layout.setContentsMargins(0, 0, 0, 0)
                for item in section_items:
                    if item is None or isinstance(item, QtWidgets.QWidget):
                        continue
                    if isinstance(item, QtWidgets.QWidgetAction):
                        # QWidgetAction 直接取出其内嵌控件放入布局
                        w = item.defaultWidget()
                        if w is not None:
                            content_layout.addWidget(w)
                    else:
                        btn = QtWidgets.QToolButton()
                        btn.setDefaultAction(item)
                        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  # type: ignore[attr-defined]
                        btn.setIconSize(QtCore.QSize(20, 20))
                        btn.setFont(btn_font)
                        btn.setStyleSheet("""
                            QToolButton {
                                border: 1px solid transparent;
                                border-radius: 4px;
                                padding: 3px 5px;
                                background-color: transparent;
                            }
                            QToolButton:hover {
                                background-color: rgba(100, 170, 230, 0.25);
                                border: 1px solid rgba(100, 170, 230, 0.55);
                            }
                            QToolButton:pressed {
                                background-color: rgba(70, 140, 200, 0.40);
                                border: 1px solid rgba(70, 140, 200, 0.75);
                            }
                            QToolButton:checked {
                                background-color: rgba(70, 140, 200, 0.30);
                                border: 1px solid rgba(70, 140, 200, 0.65);
                            }
                        """)
                        # Expanding：区段宽时按钮均匀填充；配合 _AdaptiveSection
                        # 保证内容不会窄于自然宽度，所以按钮不会被压缩
                        btn.setSizePolicy(
                            QtWidgets.QSizePolicy.Expanding,    # type: ignore[attr-defined]
                            QtWidgets.QSizePolicy.Fixed,        # type: ignore[attr-defined]
                        )
                        content_layout.addWidget(btn)

                scroll = _FlexSection()
                scroll.setFrameShape(QtWidgets.QFrame.NoFrame)  # type: ignore[attr-defined]
                scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
                scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
                scroll.setWidgetResizable(False)
                scroll.setWidget(content)
                scroll.setAlignment(  # type: ignore[call-overload]
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter  # type: ignore[attr-defined]
                )
                scroll.setStyleSheet("background: transparent; border: none;")
                scroll.viewport().setStyleSheet("background: transparent;")
                scroll.setMinimumWidth(20)

                splitter.addWidget(scroll)

        return splitter

    # Support Functions

    def noShapes(self):
        """检查当前是否没有任何标注图形"""
        return not len(self.labelList)

    # 初始化并填充工具栏、菜单栏和编辑菜单的绘制与编辑相关的操作。
    def populateModeActions(self):
        '''
        作用：初始化并填充工具栏、菜单栏和编辑菜单的绘制与编辑相关的操作。
        主要内容：
        清空旧的工具和菜单。
        向工具栏、画布菜单和编辑菜单中添加预定义的绘图和编辑操作（如绘制矩形、圆形、线条、AI辅助模式等）。
        '''
        tool, menu = self.actions.tool, self.actions.menu  # type: ignore[attr-defined]

        # 在 clear() 之前先把所有复用控件 reparent 到主窗口，防止随 toolbar 一起被销毁
        for wa in (
            self.actions.zoom,          # type: ignore[attr-defined]
            self.actions.pen,           # type: ignore[attr-defined]
            self.actions.eraserWidth,   # type: ignore[attr-defined]
            self.actions.shapeWidth,    # type: ignore[attr-defined]
        ):
            w = wa.defaultWidget()
            if w is not None:
                w.setParent(self)
        self._logit_container.setParent(self)
        self._rdp_container.setParent(self)

        self.tools.clear()

        # 将 actions.tool 按 None 分隔符拆分成多个区段
        sections: list = []
        current: list = []
        for act in tool:
            if act is None:
                if current:
                    sections.append(current)
                    current = []
            else:
                current.append(act)
        if current:
            sections.append(current)

        # 参数调节区作为最后一个区段（直接放 QWidget）
        # 将滑块类 QWidgetAction 的内嵌控件也并入此区段
        param_widgets = []
        for wa in (
            self.actions.zoom,          # type: ignore[attr-defined]
            self.actions.pen,           # type: ignore[attr-defined]
            self.actions.eraserWidth,   # type: ignore[attr-defined]
            self.actions.shapeWidth,    # type: ignore[attr-defined]
        ):
            w = wa.defaultWidget()
            if w is not None:
                param_widgets.append(w)
        param_widgets += [self._logit_container, self._rdp_container]
        sections.append(param_widgets)

        # 用 addWidget() 直接挂载 splitter，使其能随工具栏宽度自适应伸缩
        splitter = self._build_toolbar_splitter(sections)
        self.tools.addWidget(splitter)

        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()  # type: ignore[attr-defined]
        actions = (
            self.actions.createBrush,
            self.actions.createEraser,
            self.actions.createMode,  # type: ignore[attr-defined]
            self.actions.createRectangleMode,  # type: ignore[attr-defined]
            self.actions.createCircleMode,  # type: ignore[attr-defined]
            self.actions.createLineMode,  # type: ignore[attr-defined]
            self.actions.createPointMode,  # type: ignore[attr-defined]
            self.actions.createLineStripMode,  # type: ignore[attr-defined]
            self.actions.createAiPolygonMode,  # type: ignore[attr-defined]
            self.actions.createAiMaskMode,  # type: ignore[attr-defined]
            self.actions.createAiBboxMode,  # type: ignore[attr-defined]
            self.actions.editMode,  # type: ignore[attr-defined]
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)  # type: ignore[attr-defined]

    def setDirty(self):
        '''
        作用：标记当前文件为“已更改”状态（脏），可触发保存。
        功能：
        启用撤销按钮（如果有可恢复的形状）。
        若开启自动保存，将当前标注保存为 JSON 文件。
        更新窗口标题，显示“*”表示有未保存更改。
        启用保存按钮。
        '''
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)  # type: ignore[attr-defined]
        if self._config["auto_save"] or self.actions.saveAuto.isChecked():  # type: ignore[attr-defined]
            label_file = self._get_label_json_path(self.imagePath)  # type: ignore[arg-type]
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)  # type: ignore[attr-defined]
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        '''
        作用：将当前状态设置为“干净”（即无更改待保存）。
        功能细节：
        禁用保存按钮。
        启用所有绘制模式。
        根据是否存在标签文件启用或禁用“删除文件”按钮。
        更新窗口标题为当前文件名。`
        '''
        self.dirty = False
        self.actions.save.setEnabled(False)  # type: ignore[attr-defined]
        self.actions.createMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createRectangleMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createCircleMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createLineMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createPointMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createLineStripMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createAiPolygonMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createAiMaskMode.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.createAiBboxMode.setEnabled(True)  # type: ignore[attr-defined]
        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)
        # 如果有标签文件，则启用删除控件
        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)  # type: ignore[attr-defined]
        # 如果没有标签文件，则关闭删除控件
        else:
            self.actions.deleteFile.setEnabled(False)  # type: ignore[attr-defined]

    def toggleActions(self, value=True):
        '''作用：根据是否加载图像启用 / 禁用与图像操作相关的工具（如缩放、加载后可用的操作等）。'''
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:  # type: ignore[attr-defined]
            z.setEnabled(value)
        for action in self.actions.onLoadActive:  # type: ignore[attr-defined]
            action.setEnabled(value)

    def queueEvent(self, function):
        '''
        作用：异步将函数添加到 Qt 的事件队列中执行（避免阻塞 GUI）。
        '''
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        '''
        作用：在窗口底部状态栏显示一条临时消息，持续一定时间（默认5秒）。
        '''
        self.statusBar().showMessage(message, delay)  # type: ignore[union-attr]

    def _submit_ai_prompt(self, _) -> None:
        """提交AI文本提示，使用YOLO-World模型检测对象并生成标注框"""
        texts = self._ai_prompt_widget.get_text_prompt().split(",")
        boxes, scores, labels = bbox_from_text.get_bboxes_from_texts(
            model="yoloworld",
            image=utils.img_qt_to_arr(self.image)[:, :, :3],
            texts=texts,
        )

        for shape in self.canvas.shapes:
            if shape.shape_type != "rectangle" or shape.label not in texts:
                continue
            box = np.array(
                [
                    shape.points[0].x(),
                    shape.points[0].y(),
                    shape.points[1].x(),
                    shape.points[1].y(),
                ],
                dtype=np.float32,
            )
            boxes = np.r_[boxes, [box]]
            scores = np.r_[scores, [1.01]]
            labels = np.r_[labels, [texts.index(shape.label)]]

        boxes, scores, labels = bbox_from_text.nms_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=self._ai_prompt_widget.get_iou_threshold(),
            score_threshold=self._ai_prompt_widget.get_score_threshold(),
            max_num_detections=100,
        )

        keep = scores != 1.01
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        shape_dicts: list[dict] = bbox_from_text.get_shapes_from_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            texts=texts,
        )

        shapes: list[Shape] = []
        for shape_dict in shape_dicts:
            shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                description=shape_dict["description"],
            )
            for point in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(*point))
            shapes.append(shape)

        self.canvas.storeShapes()
        self.loadShapes(shapes, replace=False)
        self.setDirty()

    def resetState(self):
        '''
        作用：重置程序状态（通常在关闭图像或加载新图像时使用）。
        清除：文件名、图像路径、图像数据、标签文件、画布状态等。
        '''
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()
        # 重置为编辑模式，恢复默认光标（避免上一张图片的绘制模式残留）
        #if hasattr(self, "actions") and hasattr(self.actions, "createMode"):
           # self.setEditMode()

    def currentItem(self):
        '''作用：返回当前选中的标签项（用于修改、删除等操作）。'''
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        '''作用：将文件添加到“最近打开的文件列表”中，避免重复，限制最大数量。'''
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        '''作用：撤销上一步的形状修改操作，并刷新标签列表。'''
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)  # type: ignore[attr-defined]

    def tutorial(self):
        """打开在线教程网页"""
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        '''作用：在绘图过程中禁用其他与模式切换、删除相关的按钮，防止误操作。'''
        """Toggle drawing sensitive.
        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)  # type: ignore[attr-defined]
        self.actions.undoLastPoint.setEnabled(drawing)  # type: ignore[attr-defined]
        self.actions.undo.setEnabled(not drawing)  # type: ignore[attr-defined]
        self.actions.delete.setEnabled(not drawing)  # type: ignore[attr-defined]

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        """切换绘图模式（编辑模式和创建模式）"""
        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "circle": self.actions.createCircleMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
            "ai_bbox": self.actions.createAiBboxMode,
            "pen": self.actions.createBrush,
            "eraser": self.actions.createEraser,
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode

        # 重置所有绘图按钮的选中状态
        for draw_action in draw_actions.values():
            draw_action.setChecked(False)

        # 设置当前激活的绘图按钮为选中状态
        if not edit and createMode in draw_actions:
            draw_actions[createMode].setChecked(True)

        # 启用/禁用按钮
        if edit:
            # 编辑模式：所有绘图按钮都可用
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            # 创建模式：只有当前模式对应的按钮禁用，其他可用
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)

        self.actions.editMode.setEnabled(not edit)

        # 如果切换到 AI 模式（ai_polygon / ai_mask），立刻初始化当前选中的 AI 模型
        if (not edit) and createMode in ["ai_polygon", "ai_mask", "ai_bbox"]:
            if hasattr(self, "_selectAiModelComboBox"):
                model_name = self._selectAiModelComboBox.itemData(
                    self._selectAiModelComboBox.currentIndex()
                )
                self.canvas.initializeAiModel(model_name=model_name)

    def setEditMode(self):
        '''作用：快捷切换为“编辑模式”。'''
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        """动态更新“最近文件”子菜单"""
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles  # type: ignore[attr-defined]
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        '''作用：在标签列表区域右键点击时弹出右键菜单（用于修改标签等操作）。'''
        index = self.labelList.indexAt(point)
        if index.isValid() and not self.labelList.selectionModel().isSelected(index):
            self.labelList.selectionModel().select(  # type: ignore[union-attr]
                index,
                QtCore.QItemSelectionModel.ClearAndSelect
                | QtCore.QItemSelectionModel.Rows,
            )
        if not self.canvas.editing():
            self.setEditMode()
        self._sync_label_list_selection_to_canvas()
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))  # type: ignore[attr-defined]

    def _sync_label_list_selection_to_canvas(self):
        """把右侧标签列表的选中项同步到 canvas，并刷新编辑/删除 action 状态。"""
        selected_shapes = []
        for item in self.labelList.selectedItems():
            if item is None:
                continue
            try:
                shape = item.shape()
            except Exception:
                logger.exception("_sync_label_list_selection_to_canvas: 获取 item.shape() 时出错")
                continue
            if shape is not None:
                selected_shapes.append(shape)

        if selected_shapes:
            self.canvas.selectShapes(selected_shapes)
        else:
            self.canvas.deSelectShape()

    def validateLabel(self, label):
        '''作用：根据配置判断输入标签是否合法。
            规则：如果启用了 validate_label，检查标签是否存在于已知标签列表中。
        '''
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)  # type: ignore[attr-defined,union-attr]
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _edit_label(self, value=None):
        '''
        作用：对选中的标签（单个或多个）进行编辑，如标签名、属性、组ID、描述等。
        步骤：
        判断是否为一致性编辑（多选时字段一致才允许改动）。
        弹出标签编辑对话框。
        验证输入合法性并保存。
        更新列表显示和颜色。`
        '''
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(True)  # type: ignore[union-attr]
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_flags:
            for i in range(self.labelDialog.flagsLayout.count()):
                self.labelDialog.flagsLayout.itemAt(i).setDisabled(False)  # type: ignore[union-attr]
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()  # type: ignore[no-redef]

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            if shape.group_id is None:
                item.setText(
                    '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                        html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                    )
                )
            else:
                item.setText("{} ({})".format(shape.label, shape.group_id))
            self.setDirty()
            if self.uniqLabelList.findItemByLabel(shape.label) is None:
                item = self.uniqLabelList.createItemFromLabel(shape.label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(shape.label)
                self.uniqLabelList.setItemLabel(item, shape.label, rgb)
                self.uniqLabelList.undate()

    def fileSearchChanged(self):
        """当文件搜索框文本改变时，重新过滤并显示文件列表"""
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        """当文件列表中选中项改变时，加载对应的文件"""
        row = self.fileListWidget.currentRow()
        if row < 0:
            return
        path_item = self.fileListWidget.item(row, 2)
        if not path_item:
            return
        filename = path_item.text()
        if not (self._openNextImg or self._openPrevImg):
            if not self.mayContinue():
                print("fileSelectionChanged的mayContinue()")
                return
        if self._openNextImg:
            self._openNextImg = False
        if self._openPrevImg:
            self._openPrevImg = False
        if filename in self.imageList:
            self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        '''作用: 更新canvas和label列表中图形的选择状态，激活相关操作按钮（删除、复制、编辑等）。'''
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.duplicate.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.copy.setEnabled(n_selected)  # type: ignore[attr-defined]
        self.actions.edit.setEnabled(n_selected)  # type: ignore[attr-defined]
        # 有可旋转的图形（非 mask / circle / point）才启用旋转 action
        _UNROTATABLE = ('mask', 'circle', 'point')
        can_rotate = any(s.shape_type not in _UNROTATABLE for s in selected_shapes)
        self.actions.rotateShape.setEnabled(can_rotate)  # type: ignore[attr-defined]

    def _onRotationModeChanged(self, active: bool) -> None:
        """旋转模式切换时同步 action 的 checked 状态，并更新状态栏提示."""
        self.actions.rotateShape.setChecked(active)  # type: ignore[attr-defined]
        if active:
            self.status(self.tr("旋转模式已开启 — 鼠标左键按住标注拖动即可旋转，再按 R 或 Esc 退出"))
        else:
            self.status(self.tr("旋转模式已关闭"))


    def addLabel(self, shape):
        """向标签列表(labelList)中添加一个新的图形项"""
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
            self.uniqLabelList.undate()
        self.labelDialog.addLabelHistory(shape.label)
        # 同步新标签到 label.txt
        self._append_label_to_txt(self.imagePath, shape.label)
        for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _update_shape_color(self, shape):
        """根据标签名称更新图形的颜色"""
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        """根据标签名称和配置获取RGB颜色值"""
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                # rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, None)
                self.uniqLabelList.undate()
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        '''作用: 从 labelList 中移除给定的图形。'''
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=False):
        '''作用: 加载多个图形到 canvas 和 label 列表中。如果 replace=True 则清空旧图形。'''
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):
        '''作用: 从字典结构数据中构建 Shape 对象并加载，用于从标注文件还原标注。'''
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags: dict = shape["flags"] or {}
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            # 2. 从 other_data 字典中读取 'line_width'，如果找不到，再用默认值
            loaded_pen_width = other_data.get('line_width', Shape.PEN_WIDTH)

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
                pen_width= loaded_pen_width
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    def loadFlags(self, flags):
        """加载标志位到flag_widget"""
        self.flag_widget.clear()  # type: ignore[union-attr]
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # type: ignore[attr-defined]
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)  # type: ignore[attr-defined]
            self.flag_widget.addItem(item)  # type: ignore[union-attr]

    def saveLabels(self, filename):
        '''作用: 将当前所有标注和图像数据保存为文件（如 .json 或 .xml）。包括图形数据、flags、图像元信息等。'''
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            line_width_to_save = s.pen_width
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                    line_width=line_width_to_save,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        # 是否存在矢量图形
        has_shapes = bool(shapes)

        # 检查画笔图层是否有内容（pixmap2 是否完全透明）
        has_brush = False
        pm2 = getattr(self.canvas, "pixmap2", None)
        if pm2 is not None and not pm2.isNull():
            img = pm2.toImage()
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.frombuffer(ptr, np.uint8)
            has_brush = bool(arr.any())  # 只要有一个非 0 像素，就说明画过东西

        has_annotations = has_shapes or has_brush

        # 无标注时：删除 Label 目录全套文件（JSON + 画笔 PNG + 图片副本），同步 UI
        if not has_annotations:
            self._delete_label_files(self.imagePath)
            self.labelFile = None
            self._update_file_row(self.imagePath, status="", labels="")
            return True

        flags = {}
        for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
            item = self.flag_widget.item(i)  # type: ignore[union-attr]
            key = item.text()  # type: ignore[union-attr]
            flag = item.checkState() == Qt.Checked  # type: ignore[attr-defined,union-attr]
            flags[key] = flag
        try:
            # 确保 Label 目录存在
            label_dir = self._ensure_label_dir(self.imagePath)
            # JSON 保存路径：Label/<basename>.json
            label_json = self._get_label_json_path(self.imagePath)
            # JSON 中的 imagePath 记录图片文件名（与 JSON 同目录）
            img_basename = osp.basename(self.imagePath)
            imagePath = img_basename
            imageData = self.imageData if self._config["store_data"] else None
            lf.save(
                filename=label_json,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            # 将原图复制到 Label 目录（若 Label 目录中还没有）
            label_img_copy = osp.join(label_dir, img_basename)
            if not osp.exists(label_img_copy):
                shutil.copy2(self.imagePath, label_img_copy)
            # 保存画笔 PNG 到 Label 目录
            label_png = self._get_label_png_path(self.imagePath)
            self.canvas.pixmap2.save(label_png)
            self.labelFile = lf
            # 更新文件列表中该行的标注状态和已有标签
            status = self.tr("已标注") if has_annotations else ""
            labels = list({s["label"] for s in shapes})
            labels_str = ",".join(sorted(labels)) if labels else ""
            self._update_file_row(self.imagePath, status=status, labels=labels_str)
            # 将本次保存用到的标签同步到 label.txt
            for lbl in labels:
                self._append_label_to_txt(self.imagePath, lbl)

            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        '''作用: 复制并粘贴当前选中的图形，相当于克隆。'''
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        '''作用: 粘贴先前复制的图形。'''
        # 安全检查：确保剪贴板里有东西
        if self._copied_shapes is None:
            return

        # --- 核心修改：为剪贴板中的每个形状创建新的副本 ---
        # 遍历 self._copied_shapes 列表 (里面是上次Ctrl+C创建的对象)
        # 对于其中的每一个 shape 对象 s，再次调用 s.copy() 创建一个全新的副本
        # 将这些全新的副本收集到一个新的列表 shapes_to_paste 中
        shapes_to_paste = [s.copy() for s in self._copied_shapes]
        self.loadShapes(shapes_to_paste, replace=False)
        # self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        '''作用: 将选中图形复制到内部缓存 _copied_shapes。'''
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)  # type: ignore[attr-defined]

    def labelSelectionChanged(self):
        """当标签列表中选中项发生变化时，更新 canvas 中选中的图形。"""
        if self._noSelectionSlot:
            return

        # 不是在编辑模式时，什么都不做，避免无意义的操作
        if not self.canvas.editing():
            return

        try:
            selected_shapes = []
            for item in self.labelList.selectedItems():
                # 有些 item 可能已经被 Qt 删除，或没有 shape() 方法
                if item is None:
                    continue
                try:
                    shape = item.shape()
                except Exception:
                    logger.exception("labelSelectionChanged: 获取 item.shape() 时出错")
                    continue

                if shape is not None:
                    selected_shapes.append(shape)

            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

        except Exception:
            # 任何未预料到的异常，都记录日志但不让程序崩掉
            logger.exception("labelSelectionChanged 运行时发生未处理异常")

    def labelItemChanged(self, item):
        """当标签列表中的 item 被勾选/取消勾选时，设置对应图形是否可见。"""
        try:
            if item is None:
                return
            shape = item.shape()
            self.canvas.setShapeVisible(
                shape,
                item.checkState() == Qt.Checked  # type: ignore[attr-defined]
            )
        except Exception:
            logger.exception("labelItemChanged 运行时发生未处理异常")

    def labelOrderChanged(self):
        '''作用: 当标签排序变化时，重新加载图形顺序。'''
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        '''作用: 弹出标签输入框，创建一个新的标注图形并添加到 canvas。'''
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        # 获取当前在唯一标签列表中选中的项目
        items = self.uniqLabelList.selectedItems()
        # 初始化文本变量为None
        text = None
        # 如果有选中的项目
        if items:
            # 从选中项中获取用户角色数据（通常是标签文本）
            text = items[0].data(Qt.UserRole)  # type: ignore[attr-defined]

        # 初始化空字典用于存储标签标志
        flags = {}
        # 初始化分组ID为None
        group_id = None
        # 初始化描述为空字符串
        description = ""

        # 如果配置要求显示标签弹窗，或者没有获取到文本（即右侧没有选中标签）
        if self._config["display_label_popup"] or not text:
            # 保存标签对话框当前的文本内容
            previous_text = self.labelDialog.edit.text()
            # 弹出标签编辑对话框，获取用户输入的文本、标志、分组ID和描述
            text, flags, group_id, description = self.labelDialog.popUp(text)
            # 如果用户没有输入文本（取消或清空）
            if not text:
                # 恢复对话框原来的文本内容
                self.labelDialog.edit.setText(previous_text)

        # 如果用户输入了文本但验证不通过
        if text and not self.validateLabel(text):
            # 显示错误消息
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            # 清空文本
            text = ""

        # 如果最终有有效的文本
        if text:
            # 清空标签列表的选择状态
            self.labelList.clearSelection()
            # 在画布上设置最后一个形状的标签，并获取形状对象
            shape = self.canvas.setLastLabel(text, flags)
            # 设置形状的分组ID
            shape.group_id = group_id
            # 设置形状的描述
            shape.description = description
            # 将带标签的形状添加到标签列表中
            self.addLabel(shape)
            # 启用编辑模式动作
            self.actions.editMode.setEnabled(True)  # type: ignore[attr-defined]
            # 禁用撤销最后点动作
            self.actions.undoLastPoint.setEnabled(False)  # type: ignore[attr-defined]
            # 启用撤销动作
            self.actions.undo.setEnabled(True)  # type: ignore[attr-defined]
            # 标记文档为已修改（脏状态）
            self.setDirty()
        else:
            # 如果没有有效文本，撤销画布上的最后一条线
            self.canvas.undoLastLine()
            # 弹出形状备份堆栈中的最后一个备份
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        """响应鼠标滚轮的滚动请求"""
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units  # type: ignore[union-attr]
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        """设置滚动条位置并记录"""
        self.scrollBars[orientation].setValue(int(value))  # type: ignore[union-attr]
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        """设置手动缩放级别"""
        self.actions.fitWidth.setChecked(False)  # type: ignore[attr-defined]
        self.actions.fitWindow.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        """按比例增加或减少缩放"""
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomBrush_(self, v):
        """
        多边形/矩形/线条等图形的线宽控件：
        改变除画笔、橡皮之外所有 Shape 的 pen_width
        """
        # 临时预览线条的宽度
        self.canvas.line.pen_width = v

        # 新创建图形的默认宽度
        Shape.PEN_WIDTH = v

        # 更新当前已经存在的图形宽度
        for shape in self.canvas.shapes:
            shape.pen_width = v

        self.canvas.update()

    def onBrushWidthChanged(self, v: int):
        """
        画笔宽度控件：只修改画笔笔刷的宽度
        """
        self.canvas.pen_width = v

    def onEraserWidthChanged(self, v: int):
        """
        橡皮宽度控件：只修改橡皮的宽度
        """
        self.canvas.eraser_width = v

    def zoomBrushRequest(self, delta):
        """
        Alt + 鼠标滚轮：根据当前绘制模式调整对应的宽度
        pen    -> 画笔宽度（Brush width）
        eraser -> 橡皮宽度（Eraser width）
        其它    -> 图形宽度（Shape width）
        """
        steps = 0
        if delta > 0:
            steps = 1
        elif delta < 0:
            steps = -1

        if steps == 0:
            return

        mode = self.canvas.createMode
        step_amount = steps * 8  # 保留你之前的步长手感

        # 一个小工具函数：安全地设置数值（不越界），并避免选中
        def set_spin_value(spin, delta_value):
            value = spin.value() + delta_value
            # 如果 Penwidget 有最小/最大值，就按它来裁剪
            if hasattr(spin, "minimum") and hasattr(spin, "maximum"):
                value = max(spin.minimum(), min(spin.maximum(), value))
            else:
                value = max(1, value)

            spin.setValue(value)  # 不再使用 stepBy()，不会选中文本
            # 可选：确保没有蓝色选中状态（如果有 lineEdit）
            try:
                le = spin.lineEdit()
                le.deselect()
            except Exception:
                pass

        if mode == "pen":
            # 调整画笔宽度
            set_spin_value(self.penWidget, step_amount)

        elif mode == "eraser":
            # 调整橡皮宽度
            set_spin_value(self.eraserWidget, step_amount)

        else:
            # 调整多边形 / 矩形 / 线条等图形宽度
            set_spin_value(self.polygonWidget, step_amount)

        self.update()

    def zoomRequest(self, delta, pos):
        """Ctrl + 滚轮缩放，并尽量以鼠标为中心缩放"""

        # 1. 旧的缩放因子、居中偏移和滚动条位置
        old_scale = self.canvas.scale
        if old_scale == 0:
            return

        old_offset = self.canvas.offsetToCenter()  # QPointF

        h_bar = self.scrollBars[Qt.Horizontal]  # type: ignore[attr-defined]
        v_bar = self.scrollBars[Qt.Vertical]  # type: ignore[attr-defined]
        h_old = h_bar.value()
        v_old = v_bar.value()

        # 2. 根据滚轮方向决定放大或缩小
        factor = 1.1
        if delta < 0:
            factor = 0.9

        # 触发缩放（会改变 zoomWidget，从而在 paintCanvas 里更新 canvas.scale）
        self.addZoom(factor)

        # 3. 缩放后的比例和居中偏移
        new_scale = self.canvas.scale
        new_offset = self.canvas.offsetToCenter()

        # 4. 计算新的滚动条位置，使得“鼠标下的图像点”不变
        # 推导： H_new = H_old + pos * (s_new/s_old - 1) + (o_new - o_old) * s_new

        sx = new_scale / old_scale
        sy = sx  # x,y 同一个 scale

        new_h = h_old + pos.x() * (sx - 1.0) + (new_offset.x() - old_offset.x()) * new_scale
        new_v = v_old + pos.y() * (sy - 1.0) + (new_offset.y() - old_offset.y()) * new_scale

        self.setScroll(Qt.Horizontal, new_h)  # type: ignore[attr-defined]
        self.setScroll(Qt.Vertical, new_v)  # type: ignore[attr-defined]

    def setFitWindow(self, value=True):
        """设置“适应窗口”缩放模式"""
        if value:
            self.actions.fitWidth.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        """设置“适应宽度”缩放模式"""
        if value:
            self.actions.fitWindow.setChecked(False)  # type: ignore[attr-defined]
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        """启用/禁用“保持上一张图片的缩放比例”"""
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)  # type: ignore[attr-defined]

    def onNewBrightnessContrast(self, qimage):
        """当亮度和对比度改变时，更新画布显示"""
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value):
        """打开亮度对比度调节对话框"""
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        '''
        【功能说明】
        该函数用于统一设置标签列表中所有项目的选中状态（Checked/Unchecked），控制它们在界面上的可见或选中状态。
        【运行逻辑】
        首先将 flag 设置为传入的参数 value；
        遍历 self.labelList 中的每个项目（即所有标签）：
        如果传入值为 None，说明是“自动切换”，此时根据当前每个项目的选中状态设置 flag；
        否则，直接将每个项目的状态设置为 Checked 或 Unchecked，取决于 flag；
        最终统一修改每个项目的 checkState（Qt 中用于表示选中状态的属性）
        '''
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked  # type: ignore[attr-defined]
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)  # type: ignore[attr-defined]

    def loadFile(self, filename=None):
        """加载一张图片和对应的标注文件（.json），并在界面中显示图像及其标签信息，"""
        """
        文件路径和初始化处理
        如果要加载的文件已经在图像列表中并且当前未选中，强制选中；
        重置状态（清空已有数据）、禁用画布。
        """
        # changing fileListWidget loads file
        if filename in self.imageList:
            row = self._find_file_row(filename)
            if row >= 0 and self.fileListWidget.currentRow() != row:
                self.fileListWidget.setCurrentCell(row, 2)
                self.fileListWidget.repaint()
                return

        self.resetState()
        self.canvas.setEnabled(False)
        """处理文件名和存在性检查"""
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # 从 Label 目录读取对应的标注文件
        self.status(str(self.tr("Loading %s...")) % osp.basename(str(filename)))
        label_file = self._get_label_json_path(filename)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            # imageData 优先；若未嵌入则直接从原始图片路径读取
            if self.imageData is None:
                self.imageData = LabelFile.load_image_file(filename)
            self.imagePath = filename
            self.otherData = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        image = QtGui.QImage.fromData(self.imageData)

        # 如果图像不存在，输出错误日志并返回False
        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        # 从 Label 目录读取画笔 mask
        brush_mask_file_path = self._get_label_png_path(filename)
        if QtCore.QFile.exists(brush_mask_file_path):
            brush_mask = QtGui.QImage(brush_mask_file_path)
        else:
            brush_mask = None
        self.brush_mask = brush_mask

        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        '''载入画笔标注到画布'''
        if self.brush_mask:
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image), QtGui.QPixmap.fromImage(self.brush_mask))
        else:
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        '''加载标注形状与 flags 信息'''
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self.loadFlags(flags)
        # 从 label.txt 恢复该目录下全部标签历史（确保标签列表完整）
        self._load_label_txt_to_dialog(filename)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        '''
            重新绘制画布；
            添加到最近打开的文件记录；
            启用相关操作按钮；
            设置画布焦点，确保用户交互流畅。
        '''
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        # 同步文件列表选中行
        row = self._find_file_row(self.filename)
        if row >= 0 and self.fileListWidget.currentRow() != row:
            self.fileListWidget.blockSignals(True)
            self.fileListWidget.setCurrentCell(row, 2)
            self.fileListWidget.blockSignals(False)
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        return True

    def resizeEvent(self, event):
        """当窗口大小改变时，如果处于自动缩放模式，则调整图像缩放"""
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        """根据当前的缩放值重绘画布"""
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        """根据当前的缩放模式计算并设置缩放值"""
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """计算“适应窗口”模式下的缩放比例"""
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e  # type: ignore[union-attr]
        h1 = self.centralWidget().height() - e  # type: ignore[union-attr]
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        """计算“适应宽度”模式下的缩放比例"""
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0  # type: ignore[union-attr]
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        """启用/禁用“在标签文件中保存图像数据”"""
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)  # type: ignore[attr-defined]

    def closeEvent(self, event):
        """在关闭窗口前，检查是否有未保存的更改，并保存窗口设置"""
        if not self.mayContinue():
            print("closeEvent的mayContinue()")
            event.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def eventFilter(self, obj, event):
        """
        1. 拦截文件列表的 Ctrl+C → 复制完整路径到剪切板。
        2. 拦截文件菜单的鼠标释放 → 允许点击灰显的「自动保存」菜单项。
        """
        if obj is self.fileListWidget and event.type() == QtCore.QEvent.KeyPress:  # type: ignore[attr-defined]
            if event.matches(QtGui.QKeySequence.Copy):  # type: ignore[attr-defined]
                row = self.fileListWidget.currentRow()
                path_item = self.fileListWidget.item(row, 2)
                if path_item:
                    QtWidgets.QApplication.clipboard().setText(path_item.text())
                return True
        if (
            hasattr(self, "menus")
            and obj is self.menus.file  # type: ignore[attr-defined]
            and event.type() == QtCore.QEvent.MouseButtonRelease  # type: ignore[attr-defined]
        ):
            hit = self.menus.file.actionAt(event.pos())  # type: ignore[attr-defined]
            if hit is not None and hasattr(self, "actions") and hit is self.actions.saveAuto and not hit.isEnabled():  # type: ignore[attr-defined]
                # 灰显状态下点击：临时启用 → 触发切换 → 菜单关闭
                hit.setEnabled(True)
                hit.trigger()
                self.menus.file.close()  # type: ignore[attr-defined]
                return True
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event):
        """处理文件拖入事件，如果拖入的是支持的图像格式，则接受事件"""
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """处理文件放下事件，加载拖入的图像文件"""
        if not self.mayContinue():
            print("dropEvent的mayContinue")
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        """加载最近文件"""
        if self.mayContinue():
            print("loadRecent的mayContinue")
            self.loadFile(filename)

    def openPrevImg(self, _value=False):
        """打开上一张图片"""
        keep_prev = self._config["keep_prev"]
        self._openPrevImg = True
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier  # type: ignore[attr-defined]
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            print("openPrevImg的mayContinue")
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev

    def openNextImg(self, _value=False, load=True):
        """打开下一张图片"""
        keep_prev = self._config["keep_prev"]
        self._openNextImg= True

        # 检查用户是否按下了 Ctrl + Shift 组合键。
        # `keyboardModifiers()` 获取当前所有按下的修饰键（Ctrl, Shift, Alt）。
        # `|` (按位或) 将 Ctrl 和 Shift 组合成一个代表“两者都被按下”的值。
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier  # type: ignore[attr-defined]
        ):
            # 如果按下了组合键，就“临时”开启“保留标注”功能。
            # 这是一个高级用户快捷键：按住 Ctrl+Shift 再点“下一张”，可以把当前图片的标注复制到下一张。
            self._config["keep_prev"] = True

        # 调用 mayContinue() 函数，检查当前文件是否有未保存的更改。
        # 这个函数会弹出一个“是否保存？”的对话框。
        if not self._openDir:
            if not self.mayContinue():
                print("openNextImg的mayContinue")
                return
        self._openDir = False
        if len(self.imageList) <= 0:
            return

        filename = None
        # 如果当前没展示任何图片，则从第一张开始
        if self.filename is None:
            filename = self.imageList[0]
        else:
        # 获取当前正在展示的图片的index
            currIndex = self.imageList.index(self.filename)
            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]
            else:
                filename = self.imageList[-1]
        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev

    def openFile(self, _value=False):
        """点击打开按钮时执行的函数"""
        # 1. 检查是否有未保存的更改
        if not self.mayContinue():
            return

        # 2. 设置默认路径（当前文件所在目录或home目录）
        path = osp.dirname(str(self.filename)) if self.filename else "."

        # 3. 设置文件过滤器（支持的图片格式）
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]  # 支持图片和标注文件
        )

        fileDialog = FileDialogPreview(self)

        fileDialog.setFileMode(FileDialogPreview.ExistingFile)

        fileDialog.setNameFilter(filters)

        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )

        fileDialog.setWindowFilePath(path)

        fileDialog.setViewMode(FileDialogPreview.Detail)

        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]  # 获取用户选择的文件
            if fileName:
                # 切换到单张模式时，先清空文件列表和画布
                self.fileListWidget.setRowCount(0)
                self._update_file_count_label()
                self._update_file_header_check_state()
                self.resetState()
                self.canvas.setEnabled(False)
                # 将图片添加到文件列表
                self._add_file_row_to_table(fileName)
                self.loadFile(fileName)
                # 7. 禁用前后导航按钮（单文件模式）
                self.actions.openNextImg.setEnabled(False)
                self.actions.openPrevImg.setEnabled(False)  # type: ignore[attr-defined]

    def changeOutputDirDialog(self, _value=False):
        """弹出“更改输出目录”对话框"""
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(  # type: ignore[union-attr]
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()  # type: ignore[union-attr]

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        row = self._find_file_row(current_filename)
        if row >= 0:
            self.fileListWidget.setCurrentCell(row, 2)
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        """保存文件——始终保存到图片同级的 Label 目录"""
        assert not self.image.isNull(), "cannot save empty image"
        if self.output_file:
            # 命令行指定了输出文件时才走旧路径
            self._saveFile(self.output_file)
            self.close()
        else:
            # 始终保存到 <image_dir>/Label/<basename>.json
            self._saveFile(self._get_label_json_path(self.filename))

    def saveFileAs(self, _value=False):
        """弹出“另存为”对话框来保存文件"""
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        """创建并显示“另存为”对话框"""
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename  # type: ignore[assignment]
        return filename

    def _saveFile(self, filename):
        """实际执行保存文件的内部函数"""
        if filename and self.saveLabels(filename):

            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        """关闭当前文件"""
        if not self.mayContinue():
            print("closeFile的mayContinue()")
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)

    def getLabelFile(self):
        """获取当前图像对应的标注文件路径（Label 目录中）"""
        return self._get_label_json_path(self.filename)

    def deleteFile(self):
        """删除当前图像对应的标注文件"""
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, " "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        if self.filename and osp.exists(self.getLabelFile()):
            self._delete_label_files(self.filename)
            logger.info("Label files removed for: {}".format(self.filename))
            self._update_file_row(self.imagePath, status="", labels="")
            self.resetState()

    def clearAllShapes(self):
        """清空当前图片上的所有标注（包括矢量图形和画笔涂鸦）"""
        # 如果没有加载图片，直接返回
        if not self.canvas.pixmap:
            return

        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to clear all annotations (shapes & brush), proceed anyway?"
        )
        # 弹出确认框
        if yes == QtWidgets.QMessageBox.warning(
                self, self.tr("Attention"), msg, yes | no, yes
        ):
            # 1. 清空右侧标签列表 UI
            self.labelList.clear()

            # 2. 清空矢量图形
            self.canvas.loadShapes([], replace=True)

            # 3. 清空画笔绘制层 (Pen/Eraser)
            if self.canvas.pixmap2:
                self.canvas.pixmap2.fill(Qt.transparent)  # type: ignore[attr-defined]

            # 4. 同步删除 Label 目录中的全套文件（JSON + 画笔 PNG + 图片副本）
            if self.imagePath:
                self._delete_label_files(self.imagePath)
                self.labelFile = None

            # 5. 刷新画布、更新文件列表、标记为无待保存修改
            self.canvas.update()
            self._sync_file_row_live()
            self.setClean()

    # Message Dialogs. #
    def hasLabels(self):
        """检查是否至少有一个标注"""
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        """检查是否存在对应的标注文件"""
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        """检查是否有未保存的更改，并询问用户是否保存"""
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        """显示一个错误消息对话框"""
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        """获取当前文件所在的目录"""
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        """作用：
            切换配置项 "keep_prev" 的值。该配置通常用于控制是否保留当前图像上的标注（即上一张图像的标注）在加载新图像时继续保留。
            运行逻辑：
            直接将 self._config["keep_prev"] 的布尔值取反，实现开关切换。
"""
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        """
        作用：
        从当前编辑状态中移除选中的关键点（控制点）。
        运行逻辑：
        调用 self.canvas.removeSelectedPoint() 从当前图形中删除被选中的点。
        调用 self.canvas.update() 使画布重绘以反映删除后的变化。
        检查当前高亮/编辑中的图形（存储在 self.canvas.hShape 中）的 points 列表是否为空。
        如果删除后该图形已无控制点，则调用 self.canvas.deleteShape(self.canvas.hShape) 删除整个图形。
        同时，从标签列表中移除该图形对应的标签，通过 self.remLabels([self.canvas.hShape])。
        如果图形列表为空（通过 self.noShapes() 判断），则禁用所有依赖于“有标注图形存在”的操作（遍历 self.actions.onShapesPresent 并将它们禁用）。
        最后调用 self.setDirty() 标记当前工作状态为“已更改”，提示需要保存修改。
        """
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:  # type: ignore[union-attr]
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        """
        作用：
            删除当前选中的所有图形（例如选中的多边形），这个操作是不可逆的，需要用户确认后才能删除。
            运行逻辑：
            定义对话框按钮常量 Yes 与 No。
            构造提示信息，告知用户将删除多少个图形，并询问是否确认删除。
            调用 QtWidgets.QMessageBox.warning() 显示警告对话框，等待用户作出选择。
            如果用户选择“Yes”，则：
            调用 self.canvas.deleteSelected() 删除所有选中的图形，并使用 self.remLabels(...) 将相应标签从标签列表移除。
            调用 self.setDirty() 标记状态为“已更改”。
            如果删除后没有剩余图形（通过 self.noShapes() 判断），禁用所有与“有图形存在”相关的操作（遍历 self.actions.onShapesPresent 并禁用）。
            如果用户选择“No”，则操作取消，不执行删除。
        """
        if not self.canvas.selectedShapes and self.labelList.selectedItems():
            self._sync_label_list_selection_to_canvas()
        if not self.canvas.selectedShapes:
            return
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, " "proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:  # type: ignore[attr-defined]
                    action.setEnabled(False)

    def copyShape(self):
        """
        作用：
            复制当前选中的图形，并将复制出的图形添加到标签列表中。
            运行逻辑：
            调用 self.canvas.endMove(copy=True) 完成形状复制操作。这个调用确保图形复制过程结束，并且新图形被生成。
            遍历 self.canvas.selectedShapes（当前选中的图形集合），对每个图形调用 self.addLabel(shape)，将复制后的图形添加到标签列表中。
            清除标签列表中的选择状态：调用 self.labelList.clearSelection()。
            调用 self.setDirty() 标记当前操作导致状态改变，需要保存。
        """
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        """
        作用：
            确认或完成对图形的移动（平移）操作。
            运行逻辑：
            调用 self.canvas.endMove(copy=False)，这一步将结束正在进行的图形移动操作（与复制操作不同，此处参数为 False 表示不复制，而只是移动）。
            调用 self.setDirty() 标记状态为“已更改”，提示需要保存修改。
        """
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        """打开“选择目录”对话框并导入该目录下的所有图片"""
        if not self.mayContinue():  # 检查是否可以继续操作（如是否有未保存的修改，需提前实现mayContinue方法）
            print("openDirDialog的mayContinue")  # 打印日志，提示进入mayContinue判断（可用于调试）
            return  # 若不可继续，直接返回，终止打开目录流程
        self._openDir = True  # 设置标记，标识当前正在执行“打开目录”操作（供其他逻辑判断使用）
        defaultOpenDirPath = dirpath if dirpath else "."  # 初始化默认打开路径：有传入dirpath则用，无则用当前目录（.）
        if self.lastOpenDir and osp.exists(self.lastOpenDir):  # 若存在“上次打开的目录”且该目录真实存在
            defaultOpenDirPath = self.lastOpenDir  # 默认路径设为上次打开的目录（提升用户体验，记住历史路径）
        else:  # 若没有上次打开的目录或目录不存在
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."  # 用当前打开文件的所在目录，无则用当前目录


        targetDirPath = str(  # 将对话框返回的路径转为字符串（兼容不同系统路径格式）
            QtWidgets.QFileDialog.getExistingDirectory(  # 调用PyQt的文件夹选择对话框（仅选目录，不选文件）
                self,  # 父窗口对象，确保对话框在当前窗口上方显示  通过将 self（MainWindow实例）作为父窗口参数传递  是Qt框架内置的功能
                self.tr("%s - Open Directory") % __appname__,  # 对话框标题，支持国际化翻译，拼接应用名
                defaultOpenDirPath,  # 对话框默认打开的路径（上面逻辑计算出的路径）

                # 对话框选项：用|拼接多个选项
                QtWidgets.QFileDialog.ShowDirsOnly |  # 仅显示目录，隐藏文件（避免用户误选文件）
                QtWidgets.QFileDialog.DontResolveSymlinks,  # 不解析符号链接（保持路径原始性，避免跨平台问题）
            )
        )

        # 导入文件夹下面的所有图片
        if targetDirPath:  # 若用户选择了目录（未点击“取消”，targetDirPath不为空字符串）
            self.importDirImages(targetDirPath)  # 调用导入目录图片的方法，传入选择的目录路径
        else:  # 若用户点击“取消”，targetDirPath为空字符串
            pass  # 不执行任何操作，直接结束流程

    @property
    def imageList(self):
        """以列表形式返回文件列表控件中的所有文件路径"""
        lst = []
        for i in range(self.fileListWidget.rowCount()):
            path_item = self.fileListWidget.item(i, 2)
            if path_item:
                lst.append(path_item.text())
        return lst

    # ------------------------------------------------------------------
    # Label 目录约定：<image_dir>/Label/
    # 所有标注 JSON、画笔 PNG 和图片副本均存于此目录
    # ------------------------------------------------------------------
    def _get_label_dir(self, image_path):
        """返回 image_path 同级的 Label 目录路径（不创建）"""
        return osp.join(osp.dirname(osp.abspath(image_path)), "Label")

    def _get_label_json_path(self, image_path):
        """返回对应的 JSON 标注文件路径"""
        return osp.join(
            self._get_label_dir(image_path),
            osp.splitext(osp.basename(image_path))[0] + ".json",
        )

    def _get_painter_dir(self, image_path):
        """返回画笔 PNG 存放目录：<image_dir>/Label/Painter/"""
        return osp.join(self._get_label_dir(image_path), "Painter")

    def _get_label_png_path(self, image_path):
        """返回对应的画笔 PNG 路径（Label/Painter/<name>.png）"""
        return osp.join(
            self._get_painter_dir(image_path),
            osp.splitext(osp.basename(image_path))[0] + ".png",
        )

    def _get_label_image_copy_path(self, image_path):
        """返回 Label 目录中原图副本的路径：Label/<image_basename>"""
        return osp.join(self._get_label_dir(image_path), osp.basename(image_path))

    def _delete_label_files(self, image_path):
        """删除 Label 目录中与 image_path 对应的全套文件（JSON + 画笔 PNG + 图片副本）"""
        targets = (
            self._get_label_json_path(image_path),
            self._get_label_png_path(image_path),
            self._get_label_image_copy_path(image_path),
        )
        for fpath in targets:
            try:
                if osp.exists(fpath):
                    os.remove(fpath)
            except Exception as e:
                logger.warning("_delete_label_files: 删除失败 %s: %s", fpath, e)

    def _ensure_label_dir(self, image_path):
        """确保 Label 目录和 Label/Painter 子目录均存在，返回 Label 目录路径"""
        label_dir = self._get_label_dir(image_path)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(self._get_painter_dir(image_path), exist_ok=True)
        return label_dir

    # ------------------------------------------------------------------
    # label.txt 相关辅助方法
    # ------------------------------------------------------------------

    def _get_label_txt_path(self, image_path):
        """返回该图片所在目录的 Label/label.txt 路径"""
        return osp.join(self._get_label_dir(image_path), "label.txt")

    def _read_label_txt(self, image_path):
        """读取 label.txt，返回标签名列表（去空行）"""
        txt_path = self._get_label_txt_path(image_path)
        if not osp.exists(txt_path):
            return []
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        except Exception:
            return []

    def _append_label_to_txt(self, image_path, label):
        """将新标签追加到 label.txt（已存在则跳过）"""
        if not image_path or not label:
            return
        existing = self._read_label_txt(image_path)
        if label in existing:
            return
        os.makedirs(self._get_label_dir(image_path), exist_ok=True)
        try:
            with open(self._get_label_txt_path(image_path), "a", encoding="utf-8") as f:
                f.write(label + "\n")
        except Exception:
            pass

    def _load_label_txt_to_dialog(self, image_path):
        """从 label.txt 读取所有标签，同步到：
        1. 标签对话框历史列表（弹出对话框内的候选列表）
        2. 左侧 Label List 面板（uniqLabelList，用户直接可见的标签列表）
        """
        for label in self._read_label_txt(image_path):
            # 1. 标签对话框历史
            self.labelDialog.addLabelHistory(label)
            # 2. 左侧唯一标签列表面板
            if self.uniqLabelList.findItemByLabel(label) is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
                self.uniqLabelList.undate()

    # ------------------------------------------------------------------

    def _get_labels_from_json_file(self, label_file):
        """从 json 标注文件中提取唯一的 label 类型（非 shape_type）"""
        if not (QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file)):
            return []
        try:
            lf = LabelFile(label_file)
            labels = list({s["label"] for s in lf.shapes})
            return sorted(labels)
        except Exception:
            return []

    def _get_annotation_status(self, filename):
        """判断图片是否已标注：查看 Label 目录中是否有对应的 JSON 标注"""
        label_file = self._get_label_json_path(filename)
        if not QtCore.QFile.exists(label_file):
            return ""
        if not LabelFile.is_label_file(label_file):
            return ""
        try:
            lf = LabelFile(label_file)
            if lf.shapes:
                return self.tr("已标注")
        except Exception:
            pass
        brush_file = self._get_label_png_path(filename)
        if QtCore.QFile.exists(brush_file):
            return self.tr("已标注")
        return ""

    _ANNOTATED_ROW_COLOR = QtGui.QColor(208, 228, 255)  # 淡蓝色

    def _set_row_annotated_color(self, row, annotated: bool):
        """设置行背景色：已标注→淡蓝，未标注→默认"""
        color = self._ANNOTATED_ROW_COLOR if annotated else QtGui.QColor(QtCore.Qt.white)  # type: ignore[attr-defined]
        for col in range(self.fileListWidget.columnCount()):
            item = self.fileListWidget.item(row, col)
            if item:
                item.setBackground(color)

    def _set_all_file_checks(self, state):
        """表头复选框触发：批量勾选或取消所有文件行。"""
        self._updating_file_checks = True
        try:
            target_state = (
                Qt.Checked  # type: ignore[attr-defined]
                if state == Qt.Checked  # type: ignore[attr-defined]
                else Qt.Unchecked  # type: ignore[attr-defined]
            )
            for row in range(self.fileListWidget.rowCount()):
                check_item = self.fileListWidget.item(row, 1)
                if check_item is not None:
                    check_item.setCheckState(target_state)
        finally:
            self._updating_file_checks = False
        self._update_file_header_check_state()

    def _update_file_header_check_state(self, item=None):
        """根据每行复选框状态同步表头：全选、未选、半选。"""
        if getattr(self, "_updating_file_checks", False):
            return
        if item is not None and item.column() != 1:
            return
        if not hasattr(self, "fileListHeader"):
            return

        total = self.fileListWidget.rowCount()
        checked = 0
        for row in range(total):
            check_item = self.fileListWidget.item(row, 1)
            if check_item is not None and check_item.checkState() == Qt.Checked:  # type: ignore[attr-defined]
                checked += 1

        if total == 0 or checked == 0:
            state = Qt.Unchecked  # type: ignore[attr-defined]
        elif checked == total:
            state = Qt.Checked  # type: ignore[attr-defined]
        else:
            state = Qt.PartiallyChecked  # type: ignore[attr-defined]
        self.fileListHeader.setCheckState(state)

    def _add_file_row_to_table(self, filename):
        """向文件列表表格添加一行：编号、复选框、路径、标注状态、已有标签"""
        row = self.fileListWidget.rowCount()
        self.fileListWidget.insertRow(row)
        # 列0：行编号（1-based，居中，只读）
        num_item = QtWidgets.QTableWidgetItem(str(row + 1))
        num_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
        num_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.fileListWidget.setItem(row, 0, num_item)
        # 列1：选中复选框（居中）
        check_item = QtWidgets.QTableWidgetItem()
        check_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)  # type: ignore[attr-defined]
        check_item.setCheckState(Qt.Unchecked)  # type: ignore[attr-defined]
        check_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.fileListWidget.setItem(row, 1, check_item)
        # 列2：存完整路径（供内部逻辑使用），delegate 只渲染文件名，tooltip 显示完整路径
        path_item = QtWidgets.QTableWidgetItem(filename)
        path_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
        path_item.setToolTip(filename)
        self.fileListWidget.setItem(row, 2, path_item)
        # 列3：标注状态（居中）
        status = self._get_annotation_status(filename)
        status_item = QtWidgets.QTableWidgetItem(status)
        status_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
        status_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.fileListWidget.setItem(row, 3, status_item)
        # 列4：已有标签（从 Label 目录读取）
        label_file = self._get_label_json_path(filename)
        labels = self._get_labels_from_json_file(label_file)
        labels_str = ",".join(labels) if labels else ""
        labels_item = QtWidgets.QTableWidgetItem(labels_str)
        labels_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # type: ignore[attr-defined]
        self.fileListWidget.setItem(row, 4, labels_item)
        # 已标注则整行染淡蓝色
        self._set_row_annotated_color(row, bool(status))
        # 更新底部计数
        self._update_file_count_label()
        self._update_file_header_check_state()

    def _sync_file_row_live(self):
        """根据当前内存状态（不依赖磁盘文件）实时更新文件列表中当前图片的行。"""
        if not self.imagePath:
            return
        labels = sorted({item.shape().label for item in self.labelList})
        has_shapes = bool(labels)
        has_brush = False
        pm2 = getattr(self.canvas, "pixmap2", None)
        if pm2 is not None and not pm2.isNull():
            img = pm2.toImage()
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.frombuffer(ptr, np.uint8)
            has_brush = bool(arr.any())
        has_annotations = has_shapes or has_brush
        status = self.tr("已标注") if has_annotations else ""
        labels_str = ",".join(labels) if labels else ""
        self._update_file_row(self.imagePath, status=status, labels=labels_str)

    def _update_file_count_label(self):
        """更新底部计数标签"""
        if hasattr(self, "fileCountLabel"):
            n = self.fileListWidget.rowCount()
            self.fileCountLabel.setText(self.tr(f"共 {n} 张"))

    def _update_file_row(self, filename, status=None, labels=None):
        """更新表格中某行的标注状态、已有标签及行背景色"""
        for i in range(self.fileListWidget.rowCount()):
            path_item = self.fileListWidget.item(i, 2)
            if path_item and path_item.text() == filename:
                if status is not None:
                    status_item = self.fileListWidget.item(i, 3)
                    if status_item:
                        status_item.setText(status)
                    self._set_row_annotated_color(i, bool(status))
                if labels is not None:
                    labels_item = self.fileListWidget.item(i, 4)
                    if labels_item:
                        labels_item.setText(labels)
                return

    def _find_file_row(self, filename):
        """查找文件路径在表格中的行索引，不存在返回 -1"""
        for i in range(self.fileListWidget.rowCount()):
            path_item = self.fileListWidget.item(i, 2)
            if path_item and path_item.text() == filename:
                return i
        return -1

    def importDroppedImageFiles(self, imageFiles):
        """导入拖放的图片文件"""
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            self._add_file_row_to_table(file)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)  # type: ignore[attr-defined]
            self.actions.openPrevImg.setEnabled(True)  # type: ignore[attr-defined]

        self.openNextImg()

    def importDirImages(self, dirpath, pattern=None, load=True):
        """导入指定目录下的所有图片"""
        self.actions.openNextImg.setEnabled(True)  # type: ignore[attr-defined]
        self.actions.openPrevImg.setEnabled(True)  # type: ignore[attr-defined]
        if not self._openDir:
            if not self.mayContinue() or not dirpath:
                print("importDirImages的mayContinue")
                return
        self.lastOpenDir = dirpath
        # 切换目录时清空画布和所有状态，避免上一张图片残留
        self.resetState()
        self.canvas.setEnabled(False)
        self.filename = None
        self.fileListWidget.setRowCount(0)
        self._update_file_count_label()
        self._update_file_header_check_state()

        filenames = self.scanAllImages(dirpath)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            self._add_file_row_to_table(filename)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        """扫描 folderPath 顶级目录下的图片（不递归子目录，自动排除 Label/Painter 子文件夹）"""
        extensions = (".jpg", ".jpeg", ".png")
        images = []
        # 只扫描顶级目录，不递归，避免把 Label/ 里的图片副本也扫进来
        try:
            entries = os.listdir(folderPath)
        except OSError:
            return images
        for entry in entries:
            full = os.path.normpath(osp.join(folderPath, entry))
            if osp.isfile(full) and entry.lower().endswith(extensions):
                images.append(full)
        images = natsort.os_sorted(images)
        return images

    # ----------------------------------------------------------------------
    # 导出 COCO / VOC 数据集
    # ----------------------------------------------------------------------
    def _get_checked_image_paths(self):
        """获取文件列表中被勾选（列1复选框）的图片路径列表."""
        paths = []
        for i in range(self.fileListWidget.rowCount()):
            check_item = self.fileListWidget.item(i, 1)
            path_item = self.fileListWidget.item(i, 2)
            if check_item and path_item and check_item.checkState() == Qt.Checked:
                paths.append(path_item.text())
        return paths

    def _ask_export_directory(self, title: str, default_name: str) -> str | None:
        """弹出对话框，选择导出根目录 + 子目录名，返回最终路径."""
        base_dir = self.lastOpenDir or (osp.dirname(self.filename) if self.filename else ".")
        base_dir = base_dir or "."
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            title,
            base_dir,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not root:
            return None

        name, ok = QtWidgets.QInputDialog.getText(
            self,
            title,
            self.tr("Dataset folder name:"),
            text=default_name,
        )
        if not ok or not name.strip():
            return None
        out_dir = osp.join(root, name.strip())
        return out_dir

    def _show_export_result(self, title: str, out_dir: str, result) -> None:
        """根据导出统计结果弹出汇总对话框."""
        lines = [self.tr("导出目录：{}").format(out_dir), ""]
        lines.append(self.tr("✔ 成功导出：{} 张").format(result.success))
        lines.append(self.tr("✘ 转换失败：{} 张").format(len(result.failed)))
        lines.append(self.tr("○ 无标注跳过：{} 张").format(len(result.no_label)))

        if result.failed:
            lines.append("")
            lines.append(self.tr("— 失败文件："))
            for p in result.failed:
                lines.append("  " + osp.basename(p))

        if result.no_label:
            lines.append("")
            lines.append(self.tr("— 无标注文件："))
            for p in result.no_label:
                lines.append("  " + osp.basename(p))

        msg = "\n".join(lines)

        if result.success == 0:
            QtWidgets.QMessageBox.warning(self, title, msg)
        else:
            QtWidgets.QMessageBox.information(self, title, msg)

    def _export_checked_to_coco(self):
        images = self._get_checked_image_paths()
        if not images:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Export COCO"),
                self.tr("请先在文件列表中勾选要导出的图片。"),
            )
            return

        out_dir = self._ask_export_directory(self.tr("Export COCO"), "coco_dataset")
        if not out_dir:
            return

        try:
            result = export_dataset.export_to_coco(
                image_paths=images,
                output_dir=out_dir,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Export COCO"),
                self.tr("导出 COCO 失败：{}").format(str(e)),
            )
            return

        self._show_export_result(self.tr("Export COCO"), out_dir, result)

    def _export_checked_to_voc(self):
        images = self._get_checked_image_paths()
        if not images:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Export VOC"),
                self.tr("请先在文件列表中勾选要导出的图片。"),
            )
            return

        out_dir = self._ask_export_directory(self.tr("Export VOC"), "voc_dataset")
        if not out_dir:
            return

        try:
            result = export_dataset.export_to_voc(
                image_paths=images,
                output_dir=out_dir,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Export VOC"),
                self.tr("导出 VOC 失败：{}").format(str(e)),
            )
            return

        self._show_export_result(self.tr("Export VOC"), out_dir, result)

    def select_item(self, index):
        """根据索引选中唯一标签列表中的项"""
        if index < self.uniqLabelList.count():
            self.uniqLabelList.setCurrentRow(index)
            # item = self.uniqLabelList.item(index)
            # label = item.data(Qt.UserRole)
            # print(f"Selected: {label}")

    def emitCurrentItemChanged(self, item):
        """当唯一标签列表选中项改变时，更新画布的当前标签和颜色。

        注意：在没有任何标签（item 为 None）时，不应崩溃，而是清空当前标签状态。
        """
        if item is None:
            # 启动时如果没有预设标签，允许画布当前标签为空
            self.canvas.current_label = None
            self.canvas.label_color = None
            return

        label = item.data(QtCore.Qt.UserRole)
        color = (
            self.uniqLabelList.gender_color(label)
            if self.uniqLabelList.gender_color
            else None
        )
        self.canvas.current_label = label
        self.canvas.label_color = color
    def save_pixmap(self, imagePath, filename):
        """已废弃：画笔 PNG 现由 saveLabels 直接保存到 Label 目录"""
        pass



