import collections
import math
from typing import Optional
import imgviz
import numpy as np
import osam
from PyQt5.QtGui import QPen
from loguru import logger
from PyQt5 import QtCore
from PyQt5.QtCore import QPointF
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import labelme.utils
from labelme._automation import polygon_from_mask
from labelme.shape import Shape  # 导入自定义的图形类
from labelme.brush import Brush  # 导入自定义的画笔类
from labelme.sam_worker import SamWorker  # SAM 后台推理工作者
# TODO(unknown):
# - [maybe] Find optimal epsilon value.

# 定义鼠标在不同状态下的光标样式
# 默认箭头光标
CURSOR_DEFAULT = QtCore.Qt.ArrowCursor  # type: ignore[attr-defined]
# 指向手型光标(通常用于悬停在可点击对象上)
CURSOR_POINT = QtCore.Qt.PointingHandCursor  # type: ignore[attr-defined]
# 十字光标(通常用于绘图)
CURSOR_DRAW = QtCore.Qt.CrossCursor  # type: ignore[attr-defined]
# CURSOR_DRAW = QtCore.Qt.ArrowCursor  # type: ignore[attr-defined]
# 闭合手型光标 (拖动时)
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor  # type: ignore[attr-defined]
# 张开手型光标 (可抓取时)
CURSOR_GRAB = QtCore.Qt.OpenHandCursor  # type: ignore[attr-defined]
# 定义使用键盘移动图形时的速度
MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):
    # --- 定义所有自定义信号 ---  以下是广播信号通道，只有MainWindow接收。例：self.canvas.zoomRequest.connect(self.handle_zoom)
    # 当需要缩放画笔大小时候触发   self.zoomRequest.emit(delta, pos)   发射信号 - 喊话
    zoomBrushRequest = QtCore.pyqtSignal(int)
    # 当需要缩放时发出，附带缩放的增量和鼠标位置
    zoomRequest = QtCore.pyqtSignal(int, QtCore.QPoint)
    # 当需要滚动时发出，附带滚动的增量和方向
    scrollRequest = QtCore.pyqtSignal(int, int)
    # 当一个新图形最终完成时发出
    newShape = QtCore.pyqtSignal()
    # 当画布内容被修改需要保存时发出
    setDirty = QtCore.pyqtSignal()
    # 当选中的图形集合发生变化时发出，附带新的选中图形列表
    selectionChanged = QtCore.pyqtSignal(list)
    # 当图形被移动后发出
    shapeMoved = QtCore.pyqtSignal()
    # 当进入或退出多边形绘制状态时发出，附带一个布尔值
    drawingPolygon = QtCore.pyqtSignal(bool)
    # 当有顶点被选中或取消选中时发出，附带一个布尔值
    vertexSelected = QtCore.pyqtSignal(bool)
    # 当鼠标在画布上移动时发出，附带鼠标在图像坐标系下的位置
    mouseMoved = QtCore.pyqtSignal(QtCore.QPointF)
    # 当需要删除图形时发出（例如，通过画笔接触到图形时）
    contact_del_shape = QtCore.pyqtSignal(list)
    # 旋转模式切换时发出，附带是否进入旋转模式的布尔值
    rotationModeChanged = QtCore.pyqtSignal(bool)
    # 向 SAM 后台工作者发送推理请求：(shape, createMode, model_name, image_embedding)
    _sam_request = QtCore.pyqtSignal(object, str, str, object)
    # 定义两种主要模式：创建模式和编辑模式
    CREATE, EDIT = 0, 1
    # 默认的创建模式是绘制多边形
    # polygon, rectangle, line, or point
    _createMode = "polygon"
    # 是否在绘制时填充图形的标志
    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        # 从关键字参数中获取配置，如果未提供则使用默认值
        self._cursor = CURSOR_DEFAULT  # 当前的鼠标光标样式
        self._last_cursor_mode = 'default'  # 记录上次光标模式
        self.epsilon = kwargs.pop("epsilon", 10.0)  # 顶点和边的吸附距离
        self.double_click = kwargs.pop("double_click", "close")  # 双击行为，默认是闭合多边形
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        # 撤销操作的最大备份数量
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(  # 是否在不同绘图模式下显示十字线光标
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
                "ai_bbox": True,
                "pen": False,
                'eraser':False
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)

        # Initialise local state.
        self.mode = self.EDIT  # 默认启动时为编辑模式
        self.shapes = []  # 存储当前图像上所有已完成的图形 (Shape对象列表)
        self.current_label=None  # 当前选中的标签名称
        self.label_color = None  # 当前选中标签的颜色
        self.shapesBackups = []  # 用于撤销操作的备份列表
        self.current_shape = None  # 正在绘制中的图形
        self.current_brush= None  # 正在使用的画笔
        # 存储当前选中的图形
        self.selectedShapes = []  # save the selected shapes here
        # 用于复制/移动操作的临时副本
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point

        # self.line 是一个临时的Shape对象，用于在绘图时实时显示反馈
        # 例如，画多边形时，它显示最后一点到鼠标当前位置的连线
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()  # 上一次鼠标点击的位置
        self.prevMovePoint = QtCore.QPoint()  # 上一次鼠标移动事件的位置
        self.offsets = QtCore.QPoint(), QtCore.QPoint()  # 用于计算移动多个图形时的偏移量
        self.scale = 1.0  # 当前的缩放比例
        self.pixmap = QtGui.QPixmap()  # 存储原始背景图像  PyQt提供的图像显示类
        self.pixmap2 = QtGui.QPixmap()  # 存储画笔/橡皮擦绘制的蒙版层
        self.pen_width=20  # 画笔的默认宽度
        self.eraser_width = 10  # 橡皮的默认宽度
        self.visible = {}  # 存储每个图形的可见性状态 {shape: bool}
        self._hideBackround = False  # 是否隐藏背景图形的内部标志
        self.hideBackround = False  # 控制是否隐藏背景图形的公开属性

        # --- 用于高亮和交互的状态变量 ---
        self.hShape = None  # 当前鼠标悬停在其上的图形 (h for highlight)
        self.prevhShape = None  # 上一个悬停的图形
        self.hVertex = None  # 当前鼠标悬停在其上的顶点索引
        self.prevhVertex = None  # 上一个悬停的顶点
        self.hEdge = None  # 当前鼠标悬停在其上的边的索引
        self.prevhEdge = None  # 上一个悬停的边
        self.movingShape = False  # 是否正在移动图形的标志
        self.snapping = True  # 是否启用顶点吸附功能的标志
        self.hShapeIsSelected = False  # 悬停的图形是否已被选中的标志
        self._painter = QtGui.QPainter()  # 一个可复用的QPainter对象，用于绘图
        self._cursor = CURSOR_DEFAULT  # 当前的鼠标光标样式



        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())

        # --- 设置控件选项 ---
        # Set widget options.
        self.setMouseTracking(True)  # 启用鼠标跟踪，即使没有按下按钮，mouseMoveEvent也会被触发
        # 设置焦点策略，允许控件通过滚轮获取焦点
        self.setFocusPolicy(QtCore.Qt.WheelFocus)  # type: ignore[attr-defined]
        # --- 初始化AI模型相关变量 ---
        # _sam: 当前已经加载的 AI 模型实例
        # _sam_model_name: 记录通过 UI 选择的模型名（例如 'sam2:latest'），
        #   避免因为内部的 model.name 与 UI 字符串不一致而导致重复初始化。
        self._sam: Optional[osam.types.Model] = None  # SAM AI模型实例
        self._sam_model_name: Optional[str] = None
        self._sam_embedding: collections.OrderedDict[
            bytes, osam.types.ImageEmbedding
        ] = collections.OrderedDict()  # 缓存图像嵌入，避免重复计算

        # --- SAM 异步推理相关 ---
        self._sam_preview_shape: Optional[Shape] = None  # 后台推理返回的最新预览结果
        self._sam_thread: Optional[QtCore.QThread] = None
        self._sam_worker: Optional[SamWorker] = None
        # "只推最新"机制（纯主线程管理，worker 队列里永远只有 ≤1 条信号）：
        #   _sam_worker_busy: worker 是否正在推理（由主线程读写，无竞态）
        #   _sam_latest_task: 等待派发的最新任务；worker 空闲后立即接手
        self._sam_worker_busy: bool = False
        self._sam_latest_task: Optional[tuple] = None  # (shape, mode, model_name, embedding)

        # 应用退出时清理工作线程（防止 "QThread: Destroyed while running" 警告）
        _app = QtWidgets.QApplication.instance()
        if _app is not None:
            _app.aboutToQuit.connect(self._cleanup_sam_worker)

        # --- ai_bbox 拖拽状态变量 ---
        self._ai_bbox_start: Optional[QtCore.QPointF] = None   # bbox 起始角点（图像坐标）
        self._ai_bbox_end: Optional[QtCore.QPointF] = None     # bbox 终止角点（图像坐标）
        self._ai_bbox_dragging: bool = False                    # 是否正在拖拽中

        # --- 旋转状态变量 ---
        self._rotationMode = False        # R 键激活的旋转工具模式（等待用户拖拽）
        self._isRotating = False          # 鼠标已按下、正在执行实时旋转
        self._rotatingShape = None        # 正在旋转的 Shape 对象
        self._rotationCenter = None       # 旋转中心点（质心）
        self._rotationStartPoint = None   # 鼠标开始拖动的点（用于计算角度增量）
        self._initialPoints = None        # 本次拖拽开始时，图形的原始顶点列表
        # 进入旋转模式时对所有图形打一次快照，Esc 时用于整体回滚
        # 结构：[(shape_ref, [QPointF, ...]), ...]
        self._rotationModeSnapshot = []

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon", "rectangle", "circle", "line", "point", "linestrip",
            "ai_polygon", "ai_mask", "ai_bbox", "pen", "eraser",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)

        # 切换绘图模式时，如果当前有尚未完成的图形（例如 AI 模式下已点击的点），直接丢弃
        # 避免用户切换到其他工具后，旧的 AI 提示点依然残留在画面上。
        if hasattr(self, "current_shape") and self.current_shape is not None:
            self.current_shape = None
            self.line.points = []
            self.line.point_labels = []
            self.drawingPolygon.emit(False)
            self.update()

        # 切换模式时清空 SAM 预览和待派发任务
        self._sam_preview_shape = None
        self._sam_latest_task = None

        self._createMode = value

        # 如果是画笔或橡皮擦模式，立即设置圆形光标
        if value in ["pen", "eraser"]:
            self.toggleEditMode(True)
        else:
            self.toggleEditMode(False)
    def _compute_and_cache_image_embedding(self) -> None:
        if self._sam is None:
            logger.warning("SAM model is not set yet")
            return

        sam: osam.types.Model = self._sam

        image: np.ndarray = labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        key = image.tobytes()
        if key in self._sam_embedding:
            return

        logger.debug("Computing image embeddings for model {!r}", sam.name)
        self._sam_embedding[key] = sam.encode_image(image=imgviz.asrgb(image))

        # 简单限制缓存大小，避免长时间标注时内存无限增长
        max_embeddings = 16
        while len(self._sam_embedding) > max_embeddings:
            # OrderedDict: 弹出最旧的条目
            self._sam_embedding.popitem(last=False)


    def initializeAiModel(self, model_name):
        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        # 只在「UI 选择的模型名」发生变化时，才真正重新加载模型
        if self._sam is None or self._sam_model_name != model_name:
            logger.debug("Initializing AI model {!r}", model_name)
            self._sam = osam.apis.get_model_type_by_name(model_name)()
            self._sam_model_name = model_name
            self._sam_embedding.clear()
            # 模型变更时重建后台 worker 线程
            self._setup_sam_worker()

        self._compute_and_cache_image_embedding()

    def _setup_sam_worker(self):
        """创建/重建 SAM 后台工作线程。每次模型切换时调用一次。"""
        # 停掉旧线程（带超时，避免切换模型时 UI 阻塞）
        if self._sam_thread is not None:
            self._sam_latest_task = None
            try:
                self._sam_request.disconnect()
            except TypeError:
                pass
            self._sam_thread.quit()
            if not self._sam_thread.wait(2000):
                self._sam_thread.terminate()
                self._sam_thread.wait(500)

        self._sam_worker = SamWorker()
        self._sam_thread = QtCore.QThread(self)
        self._sam_worker.moveToThread(self._sam_thread)

        # 信号连接：主线程 → worker（QueuedConnection，跨线程安全）
        self._sam_request.connect(self._sam_worker.run_inference)
        # 信号连接：worker → 主线程（自动检测跨线程，使用 QueuedConnection）
        self._sam_worker.finished.connect(self._on_sam_result)
        self._sam_worker.error.connect(
            lambda msg: logger.warning("SAM worker error: {}", msg)
        )

        self._sam_thread.start()
        logger.debug("SAM worker thread started")

    @QtCore.pyqtSlot(object)
    def _on_sam_result(self, shape):
        """
        收到后台推理结果（主线程执行）。
        解锁 busy 标志，若队列里还有更新的任务则立即继续推理。
        """
        self._sam_worker_busy = False
        if shape is not None:
            self._sam_preview_shape = shape
            self.update()
        # 若在本次推理期间鼠标又移动了，立刻跟上最新位置
        self._dispatch_latest_task()

    def _schedule_sam_inference(self, preview_shape: Shape):
        """
        记录最新推理请求，若 worker 空闲则立即派发，否则等待 worker 完成后自动接手。
        worker 队列里永远只有 ≤1 条信号，彻底消除"用旧位置多次推理"的问题。
        """
        if self._sam_worker is None or self._sam is None or self.pixmap is None:
            return
        image_key = labelme.utils.img_qt_to_arr(self.pixmap.toImage()).tobytes()
        if image_key not in self._sam_embedding:
            return
        # 总是用最新位置覆盖旧任务
        self._sam_latest_task = (
            preview_shape.copy(),
            self.createMode,
            self._sam.name,
            self._sam_embedding[image_key],
        )
        self._dispatch_latest_task()

    def _dispatch_latest_task(self):
        """若 worker 空闲且有待发任务，则发送并设置 busy 标志。"""
        if self._sam_worker_busy or self._sam_latest_task is None:
            return
        self._sam_worker_busy = True
        task = self._sam_latest_task
        self._sam_latest_task = None
        self._sam_request.emit(*task)

    def _cleanup_sam_worker(self):
        """
        优雅地停止 SAM 工作线程。
        连接到 QApplication.aboutToQuit，确保 App 退出前线程已终止，
        避免 "QThread: Destroyed while thread is still running" 警告。
        """
        self._sam_latest_task = None
        if self._sam_thread is None or not self._sam_thread.isRunning():
            return
        # 断开信号，防止退出期间触发新推理
        try:
            self._sam_request.disconnect()
        except TypeError:
            pass
        self._sam_thread.quit()
        # 最多等待 2 秒；若模型推理仍未结束则强制终止
        if not self._sam_thread.wait(2000):
            logger.warning("SAM worker thread did not stop gracefully, terminating")
            self._sam_thread.terminate()
            self._sam_thread.wait(500)

    def _clampCirclePointToImage(self, center: QPointF, p: QPointF) -> QPointF:
        """
        给定圆心 center 和原始控制点 p，
        返回一个“被限制在图像内部”的控制点，使得整圆都在 pixmap 内部。
        这里 center 和 p 都是图像坐标（transformPos 之后的坐标）。
        """
        if self.pixmap is None:
            return p

        img_w = self.pixmap.width()
        img_h = self.pixmap.height()

        # 当前半径
        dx = p.x() - center.x()
        dy = p.y() - center.y()
        r = math.hypot(dx, dy)
        if r == 0:
            return p

        # 圆心到四条边的距离（都要 >= 半径）
        max_r = min(
            center.x(),  # 离左边界的距离
            img_w - center.x(),  # 离右边界的距离
            center.y(),  # 离上边界
            img_h - center.y(),  # 离下边界
        )

        # 半径本来就不越界，原样返回
        if r <= max_r:
            return p

        # 超出的话，把控制点“拉回”到刚好能放下的最大半径
        scale = max_r / r
        return QPointF(
            center.x() + dx * scale,
            center.y() + dy * scale,
        )

    def _clamp_circle_edge_to_pixmap(self, center: QtCore.QPointF,
                                     edge: QtCore.QPointF) -> QtCore.QPointF:
        """
        保证以 center 为圆心的整圆完全落在 pixmap 内部。
        center / edge 均为图像坐标（已经经过 transformPos）。
        返回调整后的 edge 点。
        """
        # 没有图像就直接返回
        if self.pixmap is None:
            return edge

        # 计算当前鼠标相对圆心的向量和半径
        dx = edge.x() - center.x()
        dy = edge.y() - center.y()
        dist = math.hypot(dx, dy)

        # 半径太小（基本重合）就不用动
        if dist <= 1e-6:
            return edge

        # 圆心到四条边界的距离（以像素为单位）
        # 注意减去 1，避免半径刚好等于宽高时越界到下一像素
        left = center.x()
        top = center.y()
        right = self.pixmap.width() - 1 - center.x()
        bottom = self.pixmap.height() - 1 - center.y()

        # 允许的最大半径 = 圆心到四条边的最小距离
        max_r = min(left, top, right, bottom)

        # 如果圆心本身就贴在边上，max_r 可能 <= 0，此时只能退回圆心
        if max_r <= 0:
            return QtCore.QPointF(center)

        # 实际使用的半径 = min(当前半径, 允许的最大半径)
        r = min(dist, max_r)
        scale = r / dist

        # 沿着当前方向缩放到允许的半径
        clamped_x = center.x() + dx * scale
        clamped_y = center.y() + dy * scale
        return QtCore.QPointF(clamped_x, clamped_y)
    def storeShapes(self):
        """
            这个函数的作用是“备份”当前画布上所有的标注图形。
            以便之后可以恢复到这个状态（实现撤销功能）。
        """
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
            self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()
        # 1. 清除记录的坐标，防止 paintEvent 继续在边缘绘制圆圈
        self.prevMovePoint = None
        # 2. 强制重绘，立刻清除屏幕上残留的圆圈
        self.update()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):  # type: ignore[override]
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE：退出旋转模式
            if self._rotationMode:
                self._exitRotationModeInternal()
            self.unHighlight()
            self.deSelectShape()

    # ------------------------------------------------------------------
    # 旋转模式 (R 键触发)
    # ------------------------------------------------------------------

    def enterRotationMode(self):
        """进入旋转模式：按 R 键或点击菜单后激活，下次左键拖拽即可旋转."""
        if not self.editing():
            return
        # 进入时对所有图形的顶点打一次快照，供 Esc 回滚
        self._rotationModeSnapshot = [
            (shape, [QtCore.QPointF(p) for p in shape.points])
            for shape in self.shapes
        ]
        self._rotationMode = True
        self.overrideCursor(QtCore.Qt.CrossCursor)  # type: ignore[attr-defined]
        self.rotationModeChanged.emit(True)
        self.update()

    def exitRotationMode(self, restore: bool = False):
        """退出旋转模式（外部调用入口）.

        Args:
            restore: True = Esc 语义，恢复进入旋转模式前的所有顶点坐标；
                     False = Enter 语义，保留当前旋转结果。
        """
        if not self._rotationMode:
            return
        self._exitRotationModeInternal(restore=restore)

    def _exitRotationModeInternal(self, restore: bool = False):
        """内部：重置旋转模式及旋转状态.

        Args:
            restore: 是否把所有图形恢复到进入旋转模式时的快照。
        """
        if restore and self._rotationModeSnapshot:
            for shape, original_points in self._rotationModeSnapshot:
                if shape in self.shapes:
                    shape.points = original_points
            self.storeShapes()
            self.shapeMoved.emit()  # 通知主窗口标记 dirty，使恢复结果可被撤销

        self._rotationModeSnapshot = []
        self._rotationMode = False
        # 若旋转到一半，也一并清理
        if self._isRotating:
            self._isRotating = False
            self._rotatingShape = None
            self._rotationCenter = None
            self._rotationStartPoint = None
            self._initialPoints = None
        self.restoreCursor()
        self.rotationModeChanged.emit(False)
        self.update()

    def toggleRotationMode(self):
        """切换旋转模式（菜单 action 调用入口）."""
        if self._rotationMode:
            self.exitRotationMode(restore=False)
        else:
            self.enterRotationMode()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None


    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)
        self.prevMovePoint = pos

        # [已移除 restoreCursor]

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        # 场景一(创建模式)：Polygon drawing.
        if self.drawing():
            # --- 分支 A：形状类 ---
            if self.createMode in ["ai_polygon", "ai_mask", "polygon", "rectangle", "circle", "line", "point",
                                   "linestrip"]:
                if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'draw':
                    self.overrideCursor(CURSOR_DRAW)
                    self._last_cursor_mode = 'draw'

                if self.createMode in ["ai_polygon", "ai_mask"]:
                    self.line.shape_type = "points"
                else:
                    self.line.shape_type = self.createMode

                if not self.current_shape:
                    self.repaint()
                    return

                if self.outOfPixmap(pos):
                    pos = self.intersectionPoint(self.current_shape[-1], pos)
                elif (
                        self.snapping
                        and len(self.current_shape) > 1
                        and self.createMode == "polygon"
                        and self.closeEnough(pos, self.current_shape[0])
                ):
                    pos = self.current_shape[0]
                    self.overrideCursor(CURSOR_POINT)
                    self.current_shape.highlightVertex(0, Shape.NEAR_VERTEX)

                if self.createMode in ["polygon", "linestrip", "line"]:
                    self.line.points = [self.current_shape[-1], pos]
                    self.line.point_labels = [1, 1]
                elif self.createMode in ["ai_polygon", "ai_mask"]:
                    preview_label = 0 if is_shift_pressed else 1
                    self.line.points = [self.current_shape.points[-1], pos]
                    self.line.point_labels = [
                        self.current_shape.point_labels[-1],
                        preview_label,
                    ]
                    # 调度异步 SAM 预览推理（防抖 100ms）
                    _preview = self.current_shape.copy()
                    _preview.addPoint(point=pos, label=preview_label)
                    self._schedule_sam_inference(_preview)
                elif self.createMode == "rectangle":
                    self.line.points = [self.current_shape[0], pos]
                    self.line.point_labels = [1, 1]
                    self.line.close()
                elif self.createMode == "circle":
                    center = self.current_shape[0]
                    # 先做出界裁剪，再做“整圆不出界”的裁剪
                    pos = self._clamp_circle_edge_to_pixmap(center, pos)
                    self.line.points = [center, pos]
                    self.line.point_labels = [1, 1]
                    self.line.shape_type = "circle"
                elif self.createMode == "line":
                    self.line.points = [self.current_shape[0], pos]
                    self.line.point_labels = [1, 1]
                    self.line.close()
                elif self.createMode == "point":
                    self.line.points = [self.current_shape[0]]
                    self.line.point_labels = [1]
                    self.line.close()

                self.repaint()
                self.current_shape.highlightClear()
                return

            # --- 分支 B2：ai_bbox 拖拽模式 ---
            elif self.createMode == "ai_bbox":
                if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'draw':
                    self.overrideCursor(CURSOR_DRAW)
                    self._last_cursor_mode = 'draw'
                if self._ai_bbox_dragging:
                    # 将终点限制在图像范围内
                    if self.outOfPixmap(pos):
                        pos = QtCore.QPointF(
                            max(0, min(pos.x(), self.pixmap.width() - 1)),
                            max(0, min(pos.y(), self.pixmap.height() - 1)),
                        )
                    self._ai_bbox_end = pos
                    # 实时调度 SAM 推理预览（与 ai_polygon/ai_mask 相同机制）
                    if self._sam and self._ai_bbox_start is not None:
                        _dx = abs(pos.x() - self._ai_bbox_start.x())
                        _dy = abs(pos.y() - self._ai_bbox_start.y())
                        if _dx > 5 and _dy > 5:
                            _bx1 = min(self._ai_bbox_start.x(), pos.x())
                            _by1 = min(self._ai_bbox_start.y(), pos.y())
                            _bx2 = max(self._ai_bbox_start.x(), pos.x())
                            _by2 = max(self._ai_bbox_start.y(), pos.y())
                            _prev = Shape(shape_type="points")
                            _prev.addPoint(QtCore.QPointF(_bx1, _by1), label=2)  # box_lt
                            _prev.addPoint(QtCore.QPointF(_bx2, _by2), label=3)  # box_rb
                            self._schedule_sam_inference(_prev)
                    self.repaint()
                return

            # --- 分支 B：画笔或橡皮擦 ---
            elif self.createMode in ["pen", "eraser"]:
                if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'brush':
                    brush_size = self.getCurrentBrushSize()
                    self.setBrushCursor(brush_size=brush_size)
                    self._last_cursor_mode = 'brush'

                self.update()

                # 处理绘制逻辑
                if QtCore.Qt.LeftButton & ev.buttons():
                    Painter = QtGui.QPainter(self.pixmap2)

                    # [核心修正] 之前这里除以了 self.scale，导致线条在屏幕上看起来粗细不变
                    # 现在直接使用设定的宽度，这样缩放时，线条也会同步缩放（符合逻辑）
                    if self.createMode == "pen":
                        self.current_brush.pen_width = self.pen_width
                    elif self.createMode == "eraser":
                        self.current_brush.pen_width = self.eraser_width

                    self.current_brush.paint(Painter, pos)
                    self.current_brush.last_point = pos
                    Painter.end()

                return

        # 场景二：右键拖动
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'move':
                    self.overrideCursor(CURSOR_MOVE)
                    self._last_cursor_mode = 'move'
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # 场景三：左键拖动
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'point':
                    self.overrideCursor(CURSOR_POINT)
                    self._last_cursor_mode = 'point'
                elif self.selectedShapes and self.prevPoint:
                    if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'move':
                        self.overrideCursor(CURSOR_MOVE)
                        self._last_cursor_mode = 'move'
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint and not self._isRotating:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            elif self._isRotating and self._rotationCenter:
                start_vector_x = self._rotationStartPoint.x() - self._rotationCenter.x()
                start_vector_y = self._rotationStartPoint.y() - self._rotationCenter.y()
                current_vector_x = pos.x() - self._rotationCenter.x()
                current_vector_y = pos.y() - self._rotationCenter.y()
                start_angle_rad = math.atan2(start_vector_y, start_vector_x)
                current_angle_rad = math.atan2(current_vector_y, current_vector_x)
                delta_angle_rad = current_angle_rad - start_angle_rad

                cos_delta = math.cos(delta_angle_rad)
                sin_delta = math.sin(delta_angle_rad)
                new_points = []
                for p_init in self._initialPoints:
                    tx = p_init.x() - self._rotationCenter.x()
                    ty = p_init.y() - self._rotationCenter.y()
                    rx = tx * cos_delta - ty * sin_delta
                    ry = tx * sin_delta + ty * cos_delta
                    final_x = rx + self._rotationCenter.x()
                    final_y = ry + self._rotationCenter.y()
                    new_points.append(QtCore.QPointF(final_x, final_y))

                self._rotatingShape.points = new_points
                self.overrideCursor(CURSOR_MOVE)
                self.repaint()
                return
            return

        # 场景四：悬停检测
        if self.createMode not in ["pen", "eraser"]:
            self.setToolTip(self.tr("Image"))
            cursor_changed = False
            found_hover = False

            for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
                index = shape.nearestVertex(pos, self.epsilon)
                index_edge = shape.nearestEdge(pos, self.epsilon)
                if index is not None:
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex = index
                    self.prevhShape = self.hShape = shape
                    self.prevhEdge = self.hEdge
                    self.hEdge = None
                    shape.highlightVertex(index, shape.MOVE_VERTEX)
                    if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'hover_vertex':
                        self.overrideCursor(CURSOR_POINT)
                        self._last_cursor_mode = 'hover_vertex'
                        cursor_changed = True

                    self.setToolTip(self.tr("Click & Drag to move point\nALT + SHIFT + Click to delete point"))
                    self.setStatusTip(self.toolTip())
                    found_hover = True
                    break
                elif index_edge is not None and shape.canAddPoint():
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex
                    self.hVertex = None
                    self.prevhShape = self.hShape = shape
                    self.prevhEdge = self.hEdge = index_edge
                    if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'hover_edge':
                        self.overrideCursor(CURSOR_POINT)
                        self._last_cursor_mode = 'hover_edge'
                        cursor_changed = True
                    self.setToolTip(self.tr("ALT + Click to create point"))
                    self.setStatusTip(self.toolTip())
                    found_hover = True
                    break
                elif shape.containsPoint(pos):
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex
                    self.hVertex = None
                    self.prevhShape = self.hShape = shape
                    self.prevhEdge = self.hEdge
                    self.hEdge = None
                    self.setToolTip(self.tr("Click & drag to move shape '%s'") % shape.label)
                    self.setStatusTip(self.toolTip())
                    if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'hover_shape':
                        self.overrideCursor(CURSOR_GRAB)
                        self._last_cursor_mode = 'hover_shape'
                        cursor_changed = True
                    found_hover = True
                    break
            else:
                self.unHighlight()

            self.vertexSelected.emit(self.hVertex is not None)

            if not found_hover:
                if not hasattr(self, '_last_cursor_mode') or self._last_cursor_mode != 'default':
                    self.restoreCursor()
                    self._cursor = CURSOR_DEFAULT
                    self._last_cursor_mode = 'default'
                    cursor_changed = True

            if cursor_changed:
                self.update()




    def mousePressEvent(self, ev):
        """
        当鼠标按键被按下时此函数被调用。它是所有绘制和选择操作的起点。
        """
        # 1. 坐标转换：获取鼠标点击的窗口坐标，并将其转换为图像坐标。
        pos = self.transformPos(ev.localPos())
        # 2. 检查Shift键是否被按下，这在AI模式下用于区分正负样本点。
        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier  # type: ignore[attr-defined]
        # =================================================================
        # 场景一：如果按下的是“鼠标左键”
        # =================================================================
        if ev.button() == QtCore.Qt.LeftButton:  # type: ignore[attr-defined]
            # --- 分支 A：如果当前处于“创建模式” ---
            if self.drawing():
                # --- 子分支 1：创建的是“形状”类图形 (非画笔) ---
                # --- ai_bbox drag start ---
                if self.createMode == "ai_bbox":
                    if not self.outOfPixmap(pos):
                        self._sam_preview_shape = None
                        self._ai_bbox_start = pos
                        self._ai_bbox_end = pos
                        self._ai_bbox_dragging = True
                        self.update()
                    return


                if self.createMode not in ['pen', 'eraser']:
                    # 如果 self.current_shape 存在，说明我们正在为一个已开始的图形添加点
                   if self.current_shape :
                        # 对于多边形，将预览线的终点作为新顶点添加
                        # Add point to existing shape.
                        if self.createMode == "polygon":
                            self.current_shape.addPoint(self.line[1])
                            self.line[0] = self.current_shape[-1]  # 更新预览线的起点
                            if self.current_shape.isClosed():  # 如果添加点后图形自动闭合了
                                self.finalise()  # 则完成该图形
                        # 对于只需两点确定的图形（矩形、圆、直线），第二次点击即完成
                        elif self.createMode in ["rectangle","circle", "line"]:
                            assert len(self.current_shape.points) == 1
                            if self.createMode == "rectangle" and ev.modifiers() & Qt.ControlModifier:
                                bottom_right= self.get_image_bottom_right_relative()
                                if bottom_right:
                                    # 第一个点是起点，第二个点是右下角
                                    start_point = self.current_shape.points[0]
                                    self.line.points = [start_point, bottom_right]
                            # ★ 如果是圆，第二个点再做一次“整圆不越界”的裁剪
                            if self.createMode == "circle":
                                center = self.current_shape.points[0]
                                edge = self.line.points[1]
                                self.line.points[1] = self._clamp_circle_edge_to_pixmap(center, edge)
                            self.current_shape.points = self.line.points
                            self.finalise()
                            # 矩形吸附逻辑


                        # 对于线段带（折线），持续添加点，直到用户按住Ctrl键再点击
                        elif self.createMode == "linestrip":
                            self.current_shape.addPoint(self.line[1])
                            self.line[0] = self.current_shape[-1]
                            # ctrl+左键结束绘制
                            if int(ev.modifiers()) == QtCore.Qt.ControlModifier:  # type: ignore[attr-defined]
                                self.finalise()
                        # 对于AI辅助模式，添加一个带标签（正/负样本）的点
                        elif self.createMode in ["ai_polygon", "ai_mask"]:
                            self.current_shape.addPoint(
                                self.line.points[1],
                                label=self.line.point_labels[1],
                            )
                            self.line.points[0] = self.current_shape.points[-1]
                            self.line.point_labels[0] = self.current_shape.point_labels[-1]
                            # 按住Ctrl键再点击，完成AI图形的创建
                            if ev.modifiers() & QtCore.Qt.ControlModifier:  # type: ignore[attr-defined]
                                self.finalise()
                    # 如果 self.current_shape 不存在，并且点击在图像内，说明这是开始一个新图形的第一次点击
                   elif not self.outOfPixmap(pos):
                            # Create new shape.
                            # 新 shape 开始时清空旧预览
                            self._sam_preview_shape = None
                            # 创建一个新的、空的Shape对象
                            # 获取当前的默认宽度
                            current_pen_width = Shape.PEN_WIDTH  # 或者 self.penWidget.value()
                            self.current_shape = Shape(
                                shape_type="points"
                                if self.createMode in ["ai_polygon", "ai_mask"]
                                else self.createMode,
                                pen_width=current_pen_width  # 把当前宽度传给构造函数
                            )
                            # 添加第一个点
                            self.current_shape.addPoint(pos, label=0 if is_shift_pressed else 1)
                            if self.createMode == "point":
                                self.finalise()
                            # AI模式下按住Ctrl的第一次点击也直接完成
                            elif (
                                self.createMode in ["ai_polygon", "ai_mask"]
                                and ev.modifiers() & QtCore.Qt.ControlModifier  # type: ignore[attr-defined]
                            ):
                                self.finalise()
                            # 对于其他多点图形，则初始化预览线并进入持续绘图状态
                            else:

                                if self.createMode == "circle":
                                    self.current_shape.shape_type = "circle"
                                self.line.points = [pos, pos]
                                if (
                                    self.createMode in ["ai_polygon", "ai_mask"]
                                    and is_shift_pressed
                                ):
                                    self.line.point_labels = [0, 0]
                                else:
                                    self.line.point_labels = [1, 1]
                                    # ... 设置预览线的点标签 ...
                                    self.setHiding()  # 隐藏背景其他图形
                                    self.drawingPolygon.emit(True)  # 发出“正在绘图”信号
                                    self.update()  # 触发重绘
                # --- 子分支 2：创建的是“画笔”或“橡皮擦” ---
                if self.createMode in ["pen","eraser"]:

                    if self.createMode == "eraser":
                        # 创建一个橡皮擦笔刷
                        self.current_brush = Brush(brush_type='eraser')
                        self.current_brush.last_point = pos
                        self.current_brush.pen_width = self.eraser_width
                    elif self.createMode == "pen":
                        # 根据当前选中的标签和颜色，创建一个画笔笔刷
                        self.current_brush = Brush(
                            self.current_label,
                            self.label_color,
                            brush_type='pen',
                        )
                        self.current_brush.last_point = pos
                        self.current_brush.pen_width = self.pen_width  # 设置画笔宽度
                        # self.finalise_brush()  # 完成画笔初始化（主要是为了标记dirty）
            # --- 分支 B：如果当前处于"编辑模式" ---
            elif self.editing():
                # --- 旋转模式（R 键激活）：左键按下时初始化旋转 ---
                if self._rotationMode and not self._isRotating:
                    _UNROTATABLE = ('mask', 'circle', 'point')
                    shape_to_rotate = None
                    if self.hShape and self.hShape.shape_type not in _UNROTATABLE:
                        shape_to_rotate = self.hShape
                    else:
                        for _s in self.selectedShapes:
                            if _s.shape_type not in _UNROTATABLE:
                                shape_to_rotate = _s
                                break
                    if shape_to_rotate is not None:
                        _center = shape_to_rotate.get_centroid()
                        if _center is not None:
                            self._isRotating = True
                            self._rotatingShape = shape_to_rotate
                            self._rotationCenter = _center
                            self._rotationStartPoint = pos
                            self._initialPoints = list(shape_to_rotate.points)
                            if shape_to_rotate not in self.selectedShapes:
                                self.selectionChanged.emit([shape_to_rotate])
                            self.overrideCursor(CURSOR_MOVE)
                            self.update()
                            return  # 不设置 prevPoint，避免触发图形移动
                    return  # 旋转模式下点到空白处，忽略

                # 如果用户按住Alt键点击某条边，则在该边上添加一个新顶点
                if self.selectedEdge() and ev.modifiers() == QtCore.Qt.AltModifier:  # type: ignore[attr-defined]
                    self.addPointToEdge()
                # 如果用户按住Alt+Shift键点击某个顶点，则删除该顶点
                elif self.selectedVertex() and ev.modifiers() == (
                    QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier  # type: ignore[attr-defined]
                ):
                    self.removeSelectedPoint()  # 删除选中的顶点
                # 判断是否按下了Ctrl键（进入多选模式）
                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier  # type: ignore[attr-defined]
                # 调用核心选择函数，处理图形或顶点的选择逻辑
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                # 记录下本次点击的位置，为可能的拖动操作做准备
                self.prevPoint = pos
                # 触发重绘以显示选择高亮
                self.repaint()
        # =================================================================
        # 场景二：如果按下的是“鼠标右键” (并且处于编辑模式)
        # =================================================================
        elif ev.button() == QtCore.Qt.RightButton and self.editing():  # type: ignore[attr-defined]
            # 同样判断是否为多选模式
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier  # type: ignore[attr-defined]
            # 如果当前没有选中图形，或者鼠标悬停的图形不在已选中的图形中，
            # 那么就先执行一次选择操作。
            # 这是为了确保右键菜单弹出时，其操作对象是正确的。
            if not self.selectedShapes or (
                self.hShape is not None and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            # 同样记录点击位置，为“移动/复制到此处”的右键拖动做准备
            self.prevPoint = pos



    def mouseReleaseEvent(self, ev):
        """
        当鼠标按键被松开时此函数被调用。它负责完成拖动、弹出菜单等操作。
        """
        # 右键release
        if ev.button() == QtCore.Qt.RightButton:  # type: ignore[attr-defined]
            # self.selectedShapesCopy 只有在“右键拖动以准备复制/移动”时才会有内容。
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            # 恢复鼠标为默认的箭头光标
            self.restoreCursor()
            # --- 弹出菜单并处理取消操作 ---
            if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                # 这个if条件的意思是：如果用户“取消”了菜单，并且当前正处于“右键拖动复制/移动”的状态下...
                # 那么就认为用户放弃了这次复制/移动操作。
                # 清空临时副本，取消操作
                self.selectedShapesCopy = []
                # 重绘画布，以擦除拖动过程中显示的副本预览
                self.repaint()
        # 左键release
        elif ev.button() == QtCore.Qt.LeftButton:

            if self.drawing():
                if self.createMode == "ai_bbox":
                    if self._ai_bbox_dragging:
                        self._ai_bbox_dragging = False
                        if self._ai_bbox_start and self._ai_bbox_end:
                            dx = abs(self._ai_bbox_end.x() - self._ai_bbox_start.x())
                            dy = abs(self._ai_bbox_end.y() - self._ai_bbox_start.y())
                            if dx > 5 and dy > 5 and self._sam:
                                img_w = self.pixmap.width()
                                img_h = self.pixmap.height()
                                x1 = max(0.0, min(self._ai_bbox_start.x(), self._ai_bbox_end.x()))
                                y1 = max(0.0, min(self._ai_bbox_start.y(), self._ai_bbox_end.y()))
                                x2 = max(0.0, min(max(self._ai_bbox_start.x(), self._ai_bbox_end.x()), img_w - 1))
                                y2 = max(0.0, min(max(self._ai_bbox_start.y(), self._ai_bbox_end.y()), img_h - 1))
                                current_pen_width = Shape.PEN_WIDTH
                                self.current_shape = Shape(
                                    shape_type="points",
                                    pen_width=current_pen_width,
                                )
                                self.current_shape.addPoint(QtCore.QPointF(x1, y1), label=2)  # box_lt
                                self.current_shape.addPoint(QtCore.QPointF(x2, y2), label=3)  # box_rb
                            self._ai_bbox_start = None
                            self._ai_bbox_end = None
                            if self.current_shape:
                                self.finalise()
                            else:
                                self.update()
                    return

                if self.createMode in ["pen", "eraser"]:
                    self.finalise_brush()
            # 只有在“编辑模式”下才处理
            if self.editing():
                # 1. self.hShape is not None: 鼠标松开时正悬停在一个图形上。
                # 2. self.hShapeIsSelected: 在 mousePressEvent 中检测到，用户点击的这个图形“已经”是选中的了。
                # 3. not self.movingShape: 用户只是“单击”，而没有发生“拖动”。
                # 综合起来就是：用户在多个已选中的图形里，单击了其中一个。
                # 常见的UI逻辑是，这种操作会取消其他选择，只保留被单击的这一个。
                # （注意：这里的代码实现是“取消这一个”，可能是为了Ctrl+Click取消部分选择，但缺少了对Ctrl键的判断，
                #  也可能是为了在后续逻辑中重新选择这一个，具体行为依赖整体设计）
                if (
                    self.hShape is not None  # 鼠标松开时正悬停在一个图形上。
                    and self.hShapeIsSelected  # 在 mousePressEvent 中检测到，用户点击的这个图形“已经”是选中的了。
                    and not self.movingShape  # 用户只是“单击”，而没有发生“拖动”。
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )
                # --- 结束旋转 ---
                if self._isRotating:
                    points_changed = True  # 简化判断
                    if self._rotatingShape and points_changed:
                        self.storeShapes()  # 保存撤销记录
                        self.shapeMoved.emit()  # 发出信号

                    # 重置旋转状态
                    self._isRotating = False
                    self._rotatingShape = None
                    self._rotationCenter = None
                    self._rotationStartPoint = None
                    self._initialPoints = None
                    self.setToolTip(self.tr("Image"))  # 恢复提示
                    self.restoreCursor()  # 恢复光标
                    self.update()  # 更新界面
                    # !!! 关键：旋转事件已处理，直接返回 !!!
                    return
                # --- 旋转结束逻辑结束 ---
        # 完成一次成功的拖动操作(无论是左键还是右键拖动)
        if self.movingShape and self.hShape:
            # 查找被移动的图形在主列表中的索引
            index = self.shapes.index(self.hShape)

            # --- 优化：只有在图形真的发生改变时，才创建新的撤销记录 ---
            # 比较当前图形的点坐标和上一次备份中的坐标是否不同。
            # 如果用户只是点击了一下但没有移动，坐标相同，就不需要浪费一次撤销记录。
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                # 如果坐标已改变，则调用 storeShapes() 将当前所有图形的状态存入备份栈中。
                self.storeShapes()
                # 发出 shapeMoved 信号，通知主窗口有图形被移动了。
                self.shapeMoved.emit()
            # 操作完成，重置“正在移动”标志位
            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def get_image_bottom_right_relative(self):
        """获取图片右下角在图像坐标系中的坐标"""
        if self.pixmap.isNull():
            return None

        # 图片原始尺寸（图像坐标系）
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        # 在图像坐标系中，右下角就是 (width, height)
        # 注意：这里不需要考虑缩放和偏移，因为图像坐标系始终以图片左上角为(0,0)
        return QtCore.QPointF(img_width, img_height)
    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
            (self.current_shape and len(self.current_shape) > 2)
            or self.createMode in ["ai_polygon", "ai_mask"]
        )

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return

        if (
            self.createMode == "polygon" and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask"]:
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point.
        通过点选选择形状，支持多选模式
        Args:
            point: 点击的坐标点
            multiple_selection_mode: 是否启用多选模式
        """
        # 检查是否有顶点被选中（例如正在编辑形状的顶点）
        if self.selectedVertex():  # A vertex is marked for selection.
            # 获取当前高亮的顶点索引和对应的形状
            index, shape = self.hVertex, self.hShape
            # 高亮该顶点，设置为移动顶点模式
            shape.highlightVertex(index, shape.MOVE_VERTEX)  # type: ignore[union-attr]
        else:
            # 如果没有顶点被选中，则检查是否点击了某个形状
            # 反向遍历形状列表（后创建的形状在前，实现点击时选择最上层的形状）
            for shape in reversed(self.shapes):
                # 检查形状是否可见且包含点击点
                if self.isVisible(shape) and shape.containsPoint(point):
                    # 设置隐藏状态（可能是为了隐藏其他非选中形状）
                    self.setHiding()
                    # 如果点击的形状不在当前选中列表中
                    if shape not in self.selectedShapes:
                        # 根据多选模式决定如何更新选中列表
                        if multiple_selection_mode:
                            # 多选模式：将当前形状添加到选中列表中，并发出信号
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            # 单选模式：只选中当前点击的形状，并发出信号
                            self.selectionChanged.emit([shape])
                        # 标记当前高亮的形状未被选中（可能用于后续逻辑）
                        self.hShapeIsSelected = False
                    else:
                        # 如果点击的形状已经在选中列表中
                        self.hShapeIsSelected = True

                    # 计算选中形状相对于点击点的偏移量，用于后续的拖动操作
                    self.calculateOffsets(point)
                    # 找到匹配的形状后立即返回，不继续检查其他形状
                    return

            # 如果没有找到包含点击点的形状，则取消所有选中
            self.deSelectShape()

    def calculateOffsets(self, point):
        # 初始化边界值，用于寻找选中形状的边界框
        # 将left初始化为图片宽度-1，这样任何形状的左边界都会比它小
        left = self.pixmap.width() - 1
        # 将right初始化为0，这样任何形状的右边界都会比它大
        right = 0
        # 将top初始化为图片高度-1，这样任何形状的上边界都会比它小
        top = self.pixmap.height() - 1
        # 将bottom初始化为0，这样任何形状的下边界都会比它大
        bottom = 0

        # 遍历所有选中的形状，计算它们的整体边界框
        for s in self.selectedShapes:
            # 获取当前形状的边界矩形
            rect = s.boundingRect()

            # 更新左边界：如果当前形状的左边界比记录的最小左边界更小，则更新
            if rect.left() < left:
                left = rect.left()
            # 更新右边界：如果当前形状的右边界比记录的最大右边界更大，则更新
            if rect.right() > right:
                right = rect.right()
            # 更新上边界：如果当前形状的上边界比记录的最小上边界更小，则更新
            if rect.top() < top:
                top = rect.top()
            # 更新下边界：如果当前形状的下边界比记录的最大下边界更大，则更新
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        # 计算边界框左上角相对于给定点的偏移量
        # x1 = 左边界x坐标 - 给定点的x坐标
        x1 = left - point.x()
        # y1 = 上边界y坐标 - 给定点的y坐标
        y1 = top - point.y()

        # 计算边界框右下角相对于给定点的偏移量
        # x2 = 右边界x坐标 - 给定点的x坐标
        x2 = right - point.x()
        # y2 = 下边界y坐标 - 给定点的y坐标
        y2 = bottom - point.y()

        # 将计算得到的偏移量存储为两个QPointF对象
        # 第一个点表示左上角偏移，第二个点表示右下角偏移
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]  # type: ignore[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)  # type: ignore[union-attr]

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def paintEvent(self, event: Optional[QtGui.QPaintEvent]) -> None:
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        p.drawPixmap(0, 0, self.pixmap2)

        # 这一行把坐标系重置回了屏幕坐标，为了绘制 Crosshair (十字线) 等 UI 元素
        p.scale(1 / self.scale, 1 / self.scale)

        # draw crosshair
        if (
                self._crosshair[self._createMode]
                and self.drawing()
                and self.prevMovePoint
                and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y() * self.scale),
                self.width() - 1,
                int(self.prevMovePoint.y() * self.scale),
            )
            p.drawLine(
                int(self.prevMovePoint.x() * self.scale),
                0,
                int(self.prevMovePoint.x() * self.scale),
                self.height() - 1,
            )

        Shape.scale = self.scale

        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)

        if (
                self.createMode in ["pen", "eraser"]
                and self.prevMovePoint
                and not self.outOfPixmap(self.prevMovePoint)
        ):
            # 1. 保存当前画笔状态（当前是屏幕坐标系）
            p.save()

            # 2. 重新应用缩放，将坐标系对齐到“图像坐标系”
            p.scale(self.scale, self.scale)

            p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))  # 红色边框
            p.setBrush(QtCore.Qt.NoBrush)  # 空心


            base_size = self.getCurrentBrushSize()
            radius = base_size / 2.0

            # 此时 p 已经在图像坐标系下，直接传入图像坐标 prevMovePoint 即可
            p.drawEllipse(self.prevMovePoint, radius, radius)

            # 4. 恢复画笔状态
            p.restore()

        if self.current_shape:
            self.current_shape.paint(p)
            assert len(self.line.points) == len(self.line.point_labels)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        # --- 旋转模式指示器：在质心处绘制十字准星 ---
        if self._rotationMode:
            # 找当前要显示指示器的形状
            _UNROTATABLE = ('mask', 'circle', 'point')
            indicator_shape = None
            if self._isRotating and self._rotatingShape:
                indicator_shape = self._rotatingShape
            elif self.hShape and self.hShape.shape_type not in _UNROTATABLE:
                indicator_shape = self.hShape
            elif self.selectedShapes:
                for _s in self.selectedShapes:
                    if _s.shape_type not in _UNROTATABLE:
                        indicator_shape = _s
                        break
            if indicator_shape is not None:
                _c = indicator_shape.get_centroid()
                if _c is not None:
                    cx = _c.x() * self.scale
                    cy = _c.y() * self.scale
                    r = 8
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 80, 0, 220), 2))
                    p.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
                    p.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))
                    p.drawEllipse(QtCore.QPointF(cx, cy), r * 0.4, r * 0.4)

        # ai_bbox drag preview rectangle
        if (
            self.createMode == "ai_bbox"
            and self.drawing()
            and self._ai_bbox_dragging
            and self._ai_bbox_start is not None
            and self._ai_bbox_end is not None
        ):
            p.save()
            p.scale(self.scale, self.scale)
            pen = QtGui.QPen(QtGui.QColor(0, 120, 255, 220), 2.0 / self.scale)
            pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QtGui.QColor(0, 120, 255, 35))
            _bx1 = min(self._ai_bbox_start.x(), self._ai_bbox_end.x())
            _by1 = min(self._ai_bbox_start.y(), self._ai_bbox_end.y())
            _bx2 = max(self._ai_bbox_start.x(), self._ai_bbox_end.x())
            _by2 = max(self._ai_bbox_start.y(), self._ai_bbox_end.y())
            p.drawRect(QtCore.QRectF(QtCore.QPointF(_bx1, _by1), QtCore.QPointF(_bx2, _by2)))
            p.restore()


        if not self.current_shape:
            # ai_bbox 拖拽时 current_shape 为 None，但需要绘制 SAM 实时预览
            if (self.createMode == "ai_bbox" and self._sam
                    and self._sam_preview_shape is not None):
                self._sam_preview_shape.fill = self.fillDrawing()
                self._sam_preview_shape.selected = True
                self._sam_preview_shape.paint(p)
            p.end()
            return

        if (
                self.createMode == "polygon"
                and self.fillDrawing()
                and len(self.current_shape.points) >= 2
        ):
            drawing_shape = self.current_shape.copy()
            if drawing_shape.fill_color.getRgb()[3] == 0:
                logger.warning(
                    "fill_drawing=true, but fill_color is transparent,"
                    " so forcing to be opaque."
                )
                drawing_shape.fill_color.setAlpha(64)
            drawing_shape.addPoint(self.line[1])

        if not (self.createMode in ["ai_polygon", "ai_mask", "ai_bbox"] and self._sam):
            p.end()
            return

        # 使用后台线程最新推理结果绘制预览（非阻塞）
        if self._sam_preview_shape is not None:
            self._sam_preview_shape.fill = self.fillDrawing()
            self._sam_preview_shape.selected = True
            self._sam_preview_shape.paint(p)
        p.end()



    def pathNear(self, points: list, pos: QPointF, radius: float) -> bool:
        """
        判断传入的路径点列表中是否有路径段靠近 pos 点。

        参数:
            points: list[QPointF] - 路径点列表
            pos: QPointF - 橡皮中心点
            radius: float - 擦除半径

        返回:
            bool - 是否命中
        """
        if not points or len(points) < 2:
            return False
        path = QtGui.QPainterPath()
        path.moveTo(points[0])
        for pt in points[1:]:
            path.lineTo(pt)
        for t in range(0, 100):
            pt = path.pointAtPercent(t / 100)
            if (pt - pos).manhattanLength() <= radius:
                return True
        return False
    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)
#检查当前鼠标位置 pos 是否超出图像范围
    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current_shape
        if self.createMode in ["ai_polygon", "ai_mask", "ai_bbox"] and self._sam:
            # 丢弃待发任务，最终提交时同步推理一次，确保结果准确
            self._sam_latest_task = None
            _update_shape_with_sam(
                shape=self.current_shape,
                createMode=self.createMode,
                model_name=self._sam.name,
                image_embedding=self._sam_embedding[
                    labelme.utils.img_qt_to_arr(self.pixmap.toImage()).tobytes()
                ],
            )
        # 清理预览状态
        self._sam_preview_shape = None
        self.current_shape.close()

        self.shapes.append(self.current_shape)
        self.storeShapes()
        self.current_shape = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()
    def finalise_brush(self):
        assert self.current_brush
        #print("finalise_brush触发了")
        self.setDirty.emit()
        self.update()
    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)
    #获取图片的右下角坐标
    def get_image_bottom_right_absolute(self):
        """获取图片在画布上显示的右下角（绝对坐标）——以画布左上角为原点(0,0)"""
        if not self.pixmap:
            return QtCore.QPointF(0, 0)

        # 1. 获取图片本身的右下角（相对坐标）
        relative_br = self.get_image_bottom_right_relative()
        # 2. 获取图片在画布上的居中偏移量（Canvas会自动居中图片，这里要计算偏移）
        offset = self.offsetToCenter()
        # 3. 计算绝对坐标 = (相对坐标 + 偏移量) * 缩放比例
        # 因为画布有缩放（self.scale），最终显示位置需要乘以缩放系数
        absolute_x = (relative_br.x() + offset.x()) * self.scale
        absolute_y = (relative_br.y() + offset.y()) * self.scale

        return QtCore.QPointF(absolute_x, absolute_y)

        #计算 “当前图形的最后一条边” 与 “当前鼠标位置（或目标点）” 的交点

    def intersectionPoint(self, p1, p2):
        """
        计算线段与图像边界的交点，用于将超出图像范围的鼠标位置吸附到图像边缘。

        原理：通过计算从图像内点p1到图像外点p2的线段，与图像四条边界（上、右、下、左）的交点，
        选择最近的交点作为吸附点，确保图形顶点不会超出图像范围。

        参数:
            p1: QPointF - 线段的起点（已知在图像内）
            p2: QPointF - 线段的终点（可能在图像外）

        返回:
            QPointF - 线段与图像边界的最近交点（吸附点）
        """
        # 按顺时针顺序定义图像的四个顶点（边界的端点）
        # 顺序为：左上(0,0) → 右上(width-1,0) → 右下(width-1,height-1) → 左下(0,height-1)
        size = self.pixmap.size()  # 获取图像的尺寸（宽和高）
        points = [
            (0, 0),  # 左上角顶点
            (size.width() - 1, 0),  # 右上角顶点
            (size.width() - 1, size.height() - 1),  # 右下角顶点
            (0, size.height() - 1),  # 左下角顶点
        ]
        # 确保起点p1被限制在图像内（容错处理，防止p1意外超出范围）
        # x1: 将p1的x坐标限制在[0, 图像宽度-1]之间
        x1 = min(max(p1.x(), 0), size.width() - 1)
        # y1: 将p1的y坐标限制在[0, 图像高度-1]之间
        y1 = min(max(p1.y(), 0), size.height() - 1)
        # 终点p2的原始坐标（可能在图像外）
        x2, y2 = p2.x(), p2.y()
        # 计算线段(p1→p2)与图像四条边界的所有交点，返回距离p2最近的交点
        # intersectingEdges方法生成所有有效交点，min函数根据距离筛选最近的交点
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        # 获取交点所在的边界线段的两个端点（图像的两个相邻顶点）
        x3, y3 = points[i]  # 边界线段的起点
        x4, y4 = points[(i + 1) % 4]  # 边界线段的终点（取模4确保循环到下一个顶点）

        # 特殊情况处理：如果交点与起点p1重合（说明p1在边界上，且线段p1→p2沿边界方向延伸）
        if (x, y) == (x1, y1):
            # 判断当前边界是垂直线（x3 == x4）还是水平线（y3 == y4）
            if x3 == x4:
                # 垂直线边界（左右边界）：吸附到该垂直线上，y坐标取p2在图像范围内的投影
                # min(max(0, y2), max(y3, y4))确保y坐标在边界线段的y范围内
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:
                # 水平线边界（上下边界）：吸附到该水平线上，x坐标取p2在图像范围内的投影
                # min(max(0, x2), max(x3, x4))确保x坐标在边界线段的x范围内
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        # 普通情况：直接返回计算出的最近交点
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        delta = ev.angleDelta()

        # 处理滚轮逻辑 (缩放/调整笔刷/滚动)
        if QtCore.Qt.ControlModifier == int(mods):
            self.zoomRequest.emit(delta.y(), ev.pos())
        elif QtCore.Qt.AltModifier == int(mods):
            self.zoomBrushRequest.emit(delta.x())
        else:
            self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)

        if self.createMode in ["pen", "eraser"]:
            self.update()

        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current_shape:  # type: ignore[attr-defined]
                self.current_shape = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Escape and self.createMode == "ai_bbox" and self._ai_bbox_dragging:
                self._ai_bbox_dragging = False
                self._ai_bbox_start = None
                self._ai_bbox_end = None
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():  # type: ignore[attr-defined]
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:  # type: ignore[attr-defined]
                self.snapping = False
        elif self.editing():
            # Enter：确认旋转，保留结果并退出旋转模式
            if self._rotationMode and key in (  # type: ignore[attr-defined]
                QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter
            ):
                self.exitRotationMode(restore=False)
                return
            # Esc：撤销本次旋转模式内的所有旋转，恢复原位
            if self._rotationMode and key == QtCore.Qt.Key_Escape:  # type: ignore[attr-defined]
                self.exitRotationMode(restore=True)
                return
            # R 键：切换旋转模式
            if key == QtCore.Qt.Key_R:  # type: ignore[attr-defined]
                self.toggleRotationMode()
                return
            if key == QtCore.Qt.Key_Up:  # type: ignore[attr-defined]
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:  # type: ignore[attr-defined]
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:  # type: ignore[attr-defined]
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:  # type: ignore[attr-defined]
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        # 确保文本不为空，如果为空会抛出异常
        assert text
        # 给最后一个形状设置标签文本
        # self.shapes[-1] 表示形状列表中的最后一个形状（刚刚绘制的形状）
        self.shapes[-1].label = text
        # 给最后一个形状设置标志（如：难例标志、特殊属性等）
        self.shapes[-1].flags = flags
        # 从形状备份堆栈中弹出最近的一次备份
        # 因为已经成功设置了标签，不需要保留设置标签前的备份状态
        self.shapesBackups.pop()
        # 存储当前所有形状的状态，创建新的备份点
        # 用于支持撤销/重做功能
        self.storeShapes()
        # 返回刚刚设置好标签的形状对象，方便调用者进一步操作
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current_shape = self.shapes.pop()
        self.current_shape.setOpen()
        self.current_shape.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current_shape[-1], self.current_shape[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current_shape.points = self.current_shape.points[0:1]
        elif self.createMode == "point":
            self.current_shape = None
        elif self.createMode == "ai_bbox":
            # ai_bbox 取消时彻底清除，不保留任何中间状态
            self.current_shape = None
            self._sam_preview_shape = None
            self._sam_latest_task = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current_shape or self.current_shape.isClosed():
            return
        self.current_shape.popPoint()
        if len(self.current_shape) > 0:
            self.line[0] = self.current_shape[-1]
        else:
            self.current_shape = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, pixmap2=None, clear_shapes=True):
        """
        加载图片到画布，并初始化绘画层

        Args:
            pixmap: 主图像，作为背景显示
            pixmap2: 可选的第二图像层，用于存储画笔/橡皮擦绘制内容
            clear_shapes: 是否清空已有的标注形状
        """
        # 如果提供了第二图像层（如已保存的画笔蒙版）
        if pixmap2:
            self.pixmap = pixmap  # 设置主背景图像
            self.pixmap2 = pixmap2  # 设置第二绘画层
        else:
            # 如果没有提供第二图像层，则创建新的绘画层
            self.pixmap = pixmap  # 设置主背景图像
            self.pixmap2 = pixmap.copy()  # 复制主图像作为绘画层的基础
            # 将整个绘画层填充为透明。这是关键一步，确保背景可以被看见
            # 这样画笔/橡皮擦操作只影响这个透明层，不破坏原图
            self.pixmap2.fill(QtCore.Qt.transparent)

        # 切换图片时清空 SAM 预览和待派发任务
        self._sam_latest_task = None
        self._sam_preview_shape = None

        # 如果当前是AI模式且有AI模型，预计算图像嵌入（用于AI分割）
        if self.createMode in ["ai_polygon", "ai_mask", "ai_bbox"] and self._sam:
            self._compute_and_cache_image_embedding()

        # 如果需要清空形状（如加载新图片时），则重置形状列表
        if clear_shapes:
            self.shapes = []  # 清空所有标注形状

        # 触发界面重绘，显示新加载的图像
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current_shape = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()
    #设置新的光标并保存
    def overrideCursor(self, cursor):
        self._cursor = cursor
        # [核心修改] 在设置新光标前，先恢复上一个覆盖的光标，防止堆栈无限增长
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QApplication.setOverrideCursor(cursor)
    #设置画笔的光标
    def setBrushCursor(self, brush_size, opacity=128):
        """
        方案 A：隐藏系统鼠标，使用 paintEvent 绘制虚拟光标
        """
        # [核心修改] 设置为空白光标（隐藏系统原本的箭头或圆圈）
        self.overrideCursor(QtCore.Qt.BlankCursor)
    #集成光标按钮
    def toggleEditMode(self, enabled):
        """切换编辑模式并设置圆形光标"""
        print(f"toggleEditMode called: enabled={enabled}")

        if enabled:
            brush_size = self.getCurrentBrushSize()
            print(f"Setting circular cursor with size: {brush_size}")
            self.setBrushCursor(brush_size=brush_size, opacity=80)
            self._last_cursor_mode = 'brush'
        else:
            print("Restoring default cursor")
            self.restoreCursor()
            self._cursor = CURSOR_DEFAULT
            self._last_cursor_mode = 'default'
    def getCurrentBrushSize(self):
        """获取当前画笔或橡皮擦的大小"""
        if self.createMode == "pen":
            return self.pen_width
        elif self.createMode == "eraser":
            return self.eraser_width
        else:
            return 20  # 默认大小
    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None  # type: ignore[assignment]
        self.shapesBackups = []
        self.update()


def _update_shape_with_sam(
    shape: Shape,
    createMode: str,
    model_name: str,
    image_embedding: osam.types.ImageEmbedding,
) -> None:
    if createMode not in ["ai_polygon", "ai_mask", "ai_bbox"]:
        raise ValueError(
            f"createMode must be 'ai_polygon' or 'ai_mask', not {createMode}"
        )

    response: osam.types.GenerateResponse = osam.apis.generate(
        osam.types.GenerateRequest(
            model=model_name,
            image_embedding=image_embedding,
            prompt=osam.types.Prompt(
                points=[[point.x(), point.y()] for point in shape.points],
                point_labels=shape.point_labels,
            ),
        )
    )


    if not response.annotations:
        logger.warning("No annotations returned by model {!r}", model_name)
        return

    if createMode == "ai_mask":
        y1: int
        x1: int
        y2: int
        x2: int
        if response.annotations[0].bounding_box is None:
            y1, x1, y2, x2 = imgviz.instances.mask_to_bbox(
                [response.annotations[0].mask]
            )[0].astype(int)
        else:
            y1 = response.annotations[0].bounding_box.ymin
            x1 = response.annotations[0].bounding_box.xmin
            y2 = response.annotations[0].bounding_box.ymax
            x2 = response.annotations[0].bounding_box.xmax
        shape.setShapeRefined(
            shape_type="mask",
            points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
            point_labels=[1, 1],
            mask=response.annotations[0].mask[y1 : y2 + 1, x1 : x2 + 1],
        )
    elif createMode in ("ai_polygon", "ai_bbox"):
        points = polygon_from_mask.compute_polygon_from_mask(
            mask=response.annotations[0].mask
        )
        if len(points) < 2:
            return
        shape.setShapeRefined(
            shape_type="polygon",
            points=[QtCore.QPointF(point[0], point[1]) for point in points],
            point_labels=[1] * len(points),
        )


