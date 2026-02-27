'''
定义了一个用于图形标注（如图像分割、多边形标注等）的 Shape 类，主要用于在图像上绘制和操作各种形状（polygon、多边形，rectangle、矩形，circle、圆，linestrip、折线等）。
该类可用于标注工具（如 LabelMe）中实现图形交互和可视化。
'''
import copy
import math

import numpy as np
import skimage.measure
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
import labelme.utils

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


class Shape(object):
    '''
P_SQUARE, P_ROUND: 点的绘制形状（正方形或圆形）。
MOVE_VERTEX, NEAR_VERTEX: 表示与当前选中点的关系，用于高亮处理。
line_color, fill_color 等：颜色设定（类属性，可被实例覆盖）。
point_size, scale：点的大小和图像缩放比例。
    '''

    # Render handles as squares
    P_SQUARE = 0  # 用数字 0 代表正方形的点

    # Render handles as circles
    P_ROUND = 1  # 用数字 1 代表圆形的点

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0  # 用数字 0 代表鼠标正悬停在准备移动的顶点上

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1  # 用数字 1 代表鼠标只是靠近了某个顶点

    PEN_WIDTH = 5  # 绘制图形线条的默认宽度（像素）

    # --- 定义类变量，作为所有 Shape 对象的默认外观设置 ---
    # 这些变量属于 Shape 类本身，而不是某个具体的图形。
    # The following class variables influence the drawing of all shape objects.
    line_color = None  # 默认线条颜色
    fill_color = None  # 默认填充颜色
    select_line_color = None  # 图形被选中时的线条颜色
    select_fill_color = None  # 图形被选中时的填充颜色
    vertex_fill_color = None  # 顶点（控制点）的填充颜色
    hvertex_fill_color = None  # 被高亮顶点的填充颜色
    point_type = P_ROUND  # 顶点默认绘制成圆形
    point_size = 1  # 顶点的基础大小
    scale = 1.0  # 图形的缩放比例，默认为 1.0 (不缩放)
    '''
    label: 图形标签名。
    shape_type: 图形类型（如 polygon, rectangle, circle）。
    points: 控制点列表，表示形状。
    point_labels: 每个点的标签（1 表示正例点，可用于可视化分类）。
    mask: 若为 mask 形状，使用二值图表示。
    '''
    def __init__(
        self,
        label=None,  # 图形的标签/名字
        line_color=None,  # 这个图形特定的线条颜色。
        shape_type=None,  # 图形的类型，比如 'polygon', 'rectangle'。默认为 None。
        flags=None,  # 一个字典，存储与这个图形相关的标志，比如 {'occluded': True}。默认为 None。
        group_id=None,  # 图形所属的组ID。默认为 None。
        description=None,  # 对这个图形的文字描述。默认为 None。
        mask=None,  # 如果是 mask 类型的图形，这里会存储一个 numpy 数组表示的掩码。默认为 None。
        pen_width=None
    ):
        # 这些变量属于每一个具体的 Shape 对象实例。
        self.label = label
        self.group_id = group_id
        self.points = []  # 初始化一个空列表，用来存储这个图形的所有顶点坐标 (QPointF 对象)
        self.point_labels = []  # 初始化一个空列表，用来存储每个顶点的标签 (比如 AI 模式下的正负样本点)
        self.shape_type = shape_type    # 把传入的 shape_type 存起来 (后面会通过 @property 进行检查)
        self._shape_raw = None  # 备份原始的形状信息 (类型, 点, 点标签)，初始为空
        self._points_raw = []   # (这个变量似乎没有被使用)
        self._shape_type_raw = None     # (这个变量似乎没有被使用)
        self.fill = False  # 这个图形是否需要被填充颜色
        self.selected = False   # 这个图形当前是否被选中
        self.flags = flags  # 把传入的 flags 字典存起来
        self.description = description
        self.other_data = {}  # 初始化一个空字典，可以用来存储其他附加数据
        self.mask = mask  # 把传入的 mask 数组存起来
        self._highlightIndex = None    # 当前哪个顶点被高亮了
        self._highlightMode = self.NEAR_VERTEX  # 当前的高亮模式是“靠近”还是“准备移动”
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),  # 靠近时：放大4倍，圆形
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),  # 准备移动时：放大1.5倍，方形

        }

        self._closed = False  # 这个图形是否是闭合的
        # 存储这一个 Shape 实例自己的宽度
        if pen_width is not None:
            self.pen_width = pen_width
        else:
            self.pen_width = Shape.PEN_WIDTH  # 如果没提供，用当前的类默认值
        # 处理传入的线条颜色
        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.

            # 这会覆盖掉类变量 Shape.line_color 的默认值，只对这一个 Shape 对象生效。
            self.line_color = line_color

    def _scale_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        """根据当前的缩放比例 self.scale 来调整点的坐标"""
        # 返回一个新的 QPointF 对象，其 x 和 y 坐标都乘以了缩放比例
        return QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
    '''
    setShapeRefined / restoreShapeRaw
    支持备份原始图形（类型、点、标签）并进行还原，用于修改后恢复。'''
    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        """
        当 AI 模型对形状进行了优化后，调用此方法来更新形状，并备份原始形状。
        """
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        """
        将形状恢复到 AI 优化之前的状态。
        """
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
        ]:
            raise ValueError("Unexpected shape_type: {}".format(value))
        self._shape_type = value

    def close(self):
        """将图形标记为“已闭合”"""
        self._closed = True

    def get_centroid(self) -> QtCore.QPointF | None:
        """
        计算并返回图形的几何中心点（质心）。
        对于点、线、矩形、圆，使用简化计算。
        对于多边形，使用质心公式。
        对于 mask，返回 None (或边界框中心)。
        """
        # 获取图形顶点的数量
        num_points = len(self.points)

        # --- 处理特殊或简单情况 ---
        if self.shape_type == 'mask':
            # 对于 Mask 类型，可以简单返回其边界框的中心
            # (假设 self.points 存储了左上角和右下角)
            if num_points == 2:
                return (self.points[0] + self.points[1]) / 2.0
            else:
                # 如果点数不对，或者需要更精确的基于掩码本身的质心计算（较复杂），
                # 这里暂时返回 None
                logger.warning(f"无法为 mask '{self.label}' 计算中心点 (点数: {num_points})")
                return None
        elif num_points == 0:
            # 没有顶点，无法计算中心
            return None
        elif num_points == 1 or self.shape_type == 'point':
            # 只有一个点，中心就是该点本身
            return self.points[0]
        elif num_points == 2 and self.shape_type in ['line', 'rectangle', 'circle']:
            # 对于由两个点定义的线、标准矩形、圆，中心是两点的中点
            return (self.points[0] + self.points[1]) / 2.0
        elif self.shape_type == 'rectangle' and num_points == 4:
            # 对于旋转后的矩形（有4个顶点），中心是所有顶点的算术平均值
            # (这在几何上等于其边界框的中心)
            sum_x = sum(p.x() for p in self.points)
            sum_y = sum(p.y() for p in self.points)
            return QtCore.QPointF(sum_x / 4.0, sum_y / 4.0)

        # --- 计算多边形 (polygon) 或 折线 (linestrip) 的质心 ---
        # (注意：折线的质心几何意义不大，但公式可用)

        # 质心计算需要至少3个点才能形成有意义的面积
        # if num_points < 3 and self.shape_type == 'polygon':
        #    logger.warning(f"多边形 '{self.label}' 点数少于3，返回顶点平均值作为中心。")
        #    sum_x = sum(p.x() for p in self.points)
        #    sum_y = sum(p.y() for p in self.points)
        #    return QtCore.QPointF(sum_x / num_points, sum_y / num_points)

        # 使用标准多边形质心公式 (适用于任意简单多边形)
        area = 0.0
        center_x = 0.0
        center_y = 0.0

        # 遍历多边形的每条边 (由点 i 和点 i+1 构成)
        for i in range(num_points):
            p1 = self.points[i]
            # 获取下一个点，如果是最后一个点，则下一个是第一个点 (形成闭环)
            p2 = self.points[(i + 1) % num_points]  # 使用取模运算处理循环

            # 计算叉积 (p1.x * p2.y - p2.x * p1.y)，这是有向面积的两倍
            cross_product = (p1.x() * p2.y()) - (p2.x() * p1.y())
            # 累加面积 (最终需要除以 2)
            area += cross_product

            # 累加质心计算公式的分子部分
            center_x += (p1.x() + p2.x()) * cross_product
            center_y += (p1.y() + p2.y()) * cross_product

        # 计算最终面积
        area *= 0.5

        # 处理面积为零或非常小的情况 (例如，所有点在一条直线上)
        # 使用一个很小的数 (epsilon) 作为阈值来判断面积是否接近零
        if abs(area) < 1e-10:
            logger.warning(f"图形 '{self.label}' (类型: {self.shape_type}) 面积接近零，使用顶点平均值作为中心。")
            # 在这种退化情况下，返回所有顶点的算术平均值作为近似中心
            sum_x = sum(p.x() for p in self.points)
            sum_y = sum(p.y() for p in self.points)
            # 再次检查点数，防止除以零 (虽然理论上前面已处理 num_points == 0)
            if num_points > 0:
                return QtCore.QPointF(sum_x / num_points, sum_y / num_points)
            else:
                return None  # 如果真的没有点，还是返回 None

        # 根据质心公式计算最终的质心坐标
        # 分子除以 (6 * 面积)
        centroid_x = center_x / (6.0 * area)
        centroid_y = center_y / (6.0 * area)

        # 返回计算得到的质心坐标 (QPointF 对象)
        return QtCore.QPointF(centroid_x, centroid_y)


    def rotate(self, angle_degrees, center_point):
        """
        围绕给定的中心点旋转图形指定的角度。
        参数:
            angle_degrees (float): 旋转的角度 (单位：度)。
            center_point (QPointF): 旋转的中心点坐标 (图像坐标系)。
        """
        # 0. 如果是 mask 类型，目前不支持旋转，直接返回或抛出错误
        if self.shape_type == 'mask':
            logger.warning(f"旋转功能暂不支持 mask 类型的图形: {self.label}")
            # 或者 raise NotImplementedError("Rotation for mask shapes is not implemented.")
            return  # 暂时跳过 mask

        # 1. 将角度转换为弧度，因为 math 函数使用弧度
        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)

        # 2. 创建一个新的列表来存储旋转后的点
        rotated_points = []

        # 3. 遍历当前图形的所有顶点
        for point in self.points:
            # a. 将顶点坐标平移，使旋转中心点变为原点 (0,0)
            translated_x = point.x() - center_point.x()
            translated_y = point.y() - center_point.y()

            # b. 应用 2D 旋转矩阵计算旋转后的坐标
            rotated_x = translated_x * cos_angle - translated_y * sin_angle
            rotated_y = translated_x * sin_angle + translated_y * cos_angle

            # c. 将坐标平移回去，恢复到原来的坐标系
            new_x = rotated_x + center_point.x()
            new_y = rotated_y + center_point.y()

            # d. 将计算出的新点添加到列表中
            rotated_points.append(QtCore.QPointF(new_x, new_y))

        # 4. 用旋转后的点列表替换原来的点列表
        self.points = rotated_points

        # 5. 重要：如果图形是矩形或圆形，旋转后它不再是严格意义上的矩形/圆形了
        #    最好将其类型转换为多边形，以避免后续绘制或计算错误
        if self.shape_type in ['rectangle', 'circle']:
            logger.info(f"图形 '{self.label}' 因旋转已从 '{self.shape_type}' 转换为 'polygon'")
            self.shape_type = 'polygon'

    def addPoint(self, point, label=1):
        """向图形的点列表末尾添加一个新的顶点"""
        # 如果点列表不为空，并且新加的点和第一个点相同
        if self.points and point == self.points[0]:
            # 就自动将图形闭合
            self.close()
        else:
            # 否则，就把新点添加到 points 列表的末尾
            self.points.append(point)
            # 同时，把这个点的标签 (默认为 1) 添加到 point_labels 列表的末尾
            self.point_labels.append(label)

    def canAddPoint(self):
        """判断当前图形类型是否允许继续添加点 (只有多边形和折线可以)"""
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        """移除并返回图形的最后一个顶点"""
        if self.points:  # 如果点列表不为空
            if self.point_labels:  # 如果点标签列表也不为空
                self.point_labels.pop()  # 从点标签列表移除最后一个
            return self.points.pop()  # 从点列表移除最后一个，并将其返回
        return None

    def insertPoint(self, i, point, label=1):
        """在指定索引 i 位置插入一个新的顶点"""
        self.points.insert(i, point)  # 在 points 列表的第 i 个位置插入 point
        self.point_labels.insert(i, label)  # 在 point_labels 列表的第 i 个位置插入 label

    def removePoint(self, i):
        """移除指定索引 i 位置的顶点"""
        # 检查当前图形类型是否允许移除点
        if not self.canAddPoint():
            # 如果不允许，记录一条警告日志并返回
            logger.warning(
                "Cannot remove point from: shape_type=%r",
                self.shape_type,
            )
            return
        # 对多边形，检查移除后点数是否会少于3个
        if self.shape_type == "polygon" and len(self.points) <= 3:
            # 如果少于3个，记录警告并返回 (多边形至少需要3个点)
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return
        # 对折线，检查移除后点数是否会少于2个
        if self.shape_type == "linestrip" and len(self.points) <= 2:
            # 如果少于2个，记录警告并返回 (折线至少需要2个点)
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return
        # 如果可以移除，就从两个列表中移除对应索引的元素
        self.points.pop(i)
        self.point_labels.pop(i)

    def isClosed(self):
        """返回图形是否已闭合"""
        return self._closed

    def setOpen(self):
        """将图形标记为“未闭合”"""
        self._closed = False
    '''
    paint(self, painter)
    核心绘图函数：
    设置颜色和画笔。
    如果有 mask，则使用 QImage 显示 mask 区域，利用 skimage.measure.find_contours 获取轮廓路径并绘制。
    否则，根据 shape_type 不同，绘制矩形、圆、多边形、线段等。
    使用 drawVertex 画点（控制点高亮，红色负样本点等）。
    drawVertex(self, path, i)
    绘制第 i 个控制点：
    高亮点使用不同大小、颜色。
    区分圆形和正方形。
    '''
    def paint(self, painter):
        """
        这个函数负责将这个 Shape 对象绘制到画布上。
        参数 painter 是一个 QtGui.QPainter 对象，你可以把它想象成一支“画笔工具”。
        """
        # 如果既没有 mask 也没有点，那这个图形是空的，直接返回，什么也不画
        if self.mask is None and not self.points:
            return
        # --- 设置画笔颜色和宽度 ---
        # 根据图形是否被选中 (self.selected)，选择不同的线条颜色
        color = self.select_line_color if self.selected else self.line_color

        # 创建一支画笔 (QtGui.QPen)
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        # 设置画笔的宽度，要乘以当前的缩放比例 self.scale，这样缩放时线条粗细看起来不变
        # pen.setWidthF(self.PEN_WIDTH * self.scale)  # 原来的
        pen.setWidthF(self.pen_width * self.scale)

        # 把设置好的画笔交给“画师” painter
        painter.setPen(pen)
        # --- 情况一：绘制 Mask 类型的图形 ---
        if self.mask is not None:
            # 1. 创建一个临时的 RGBA 图像数组 (numpy array)，大小和 mask 一样
            image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
            # 2. 根据是否选中，选择填充颜色
            fill_color = (
                self.select_fill_color.getRgb()  # type: ignore[attr-defined]
                if self.selected
                else self.fill_color.getRgb()  # type: ignore[attr-defined]
            )
            # 3. 将 mask 中为 True 的区域，在临时图像上填充上颜色
            image_to_draw[self.mask] = fill_color
            # 4. 将 numpy 数组转换为 QImage 对象
            qimage = QtGui.QImage.fromData(labelme.utils.img_arr_to_data(image_to_draw))
            # 5. 根据当前缩放比例 self.scale 缩放 QImage
            qimage = qimage.scaled(
                qimage.size() * self.scale,
                QtCore.Qt.IgnoreAspectRatio,  # type: ignore[attr-defined]
                QtCore.Qt.SmoothTransformation,  # type: ignore[attr-defined]
            )
            # 6. 使用 painter 将缩放后的 QImage 绘制到画布上，绘制的起始位置是 mask 的左上角坐标 (self.points[0])
            painter.drawImage(self._scale_point(point=self.points[0]), qimage)  # 创建一个路径对象
            # 7. (可选) 绘制 mask 的轮廓线
            line_path = QtGui.QPainterPath()
            # 使用 skimage 找到 mask 的所有轮廓点
            contours = skimage.measure.find_contours(np.pad(self.mask, pad_width=1))
            for contour in contours:
                # 将轮廓点坐标转换到画布坐标系 (加上 mask 左上角的偏移)
                contour += [self.points[0].y(), self.points[0].x()]
                # 从轮廓的第一个点开始
                line_path.moveTo(
                    self._scale_point(QtCore.QPointF(contour[0, 1], contour[0, 0]))
                )
                # 连接轮廓上的所有后续点
                for point in contour[1:]:
                    line_path.lineTo(
                        self._scale_point(QtCore.QPointF(point[1], point[0]))
                    )
            # 使用 painter 绘制计算出的轮廓路径
            painter.drawPath(line_path)

        # --- 情况二：绘制基于点的图形 (多边形、矩形等) ---
        if self.points:
            line_path = QtGui.QPainterPath()  # 创建用于绘制图形线条的路径对象
            vrtx_path = QtGui.QPainterPath()  # 创建用于绘制普通顶点(正样本)的路径对象
            negative_vrtx_path = QtGui.QPainterPath()  # 创建用于绘制负样本顶点的路径对象
            # 根据不同的 shape_type 执行不同的绘制逻辑
            if self.shape_type in ["rectangle", "mask"]:  # 矩形 或 mask (mask也需要画矩形框)
                # 断言检查：点数必须是 1 或 2
                assert len(self.points) in [1, 2]  # 如果有两个点 (已经画完了)
                if len(self.points) == 2:
                    # 根据两个点创建矩形区域
                    rectangle = QtCore.QRectF(
                        self._scale_point(self.points[0]),
                        self._scale_point(self.points[1]),
                    )
                    # 将矩形添加到线条路径
                    line_path.addRect(rectangle)
                if self.shape_type == "rectangle":  # 只有矩形类型才需要画顶点
                    for i in range(len(self.points)):
                        # 调用 drawVertex 方法准备绘制每个顶点
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":  # 圆形
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:  # 如果有两个点 (圆心和边缘点)
                    # 计算半径 (两点之间的距离)
                    raidus = labelme.utils.distance(
                        self._scale_point(self.points[0] - self.points[1])
                    )
                    # 以第一个点为圆心，计算出的距离为半径，添加圆形到线条路径
                    line_path.addEllipse(
                        self._scale_point(self.points[0]), raidus, raidus
                    )
                for i in range(len(self.points)):
                    # 准备绘制顶点 (圆心和边缘控制点)
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":  # 折线 (不闭合)
                # 移动到第一个点
                line_path.moveTo(self._scale_point(self.points[0]))
                # 遍历所有点
                for i, p in enumerate(self.points):
                    # 连接到下一个点
                    line_path.lineTo(self._scale_point(p))
                    # 准备绘制这个顶点
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "points":  # AI模式下的点集
                assert len(self.points) == len(self.point_labels)  # 点和标签数量必须一致
                # 遍历所有点和对应的标签
                for i, point_label in enumerate(self.point_labels):
                    if point_label == 1:  # 如果是正样本点 (label=1)
                        self.drawVertex(vrtx_path, i)  # 准备绘制到普通顶点路径
                    else:   # 如果是负样本点 (label!=1)
                        self.drawVertex(negative_vrtx_path, i)# 准备绘制到负样本顶点路径

            else:  # 其他类型，主要是 polygon (多边形) 和 line (直线)
                # 移动到第一个点
                line_path.moveTo(self._scale_point(self.points[0]))
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                # 遍历所有点
                for i, p in enumerate(self.points):
                    # 连接到下一个点
                    line_path.lineTo(self._scale_point(p))
                    # 准备绘制这个顶点
                    self.drawVertex(vrtx_path, i)
                # 如果图形是闭合的 (比如多边形画完了)
                if self.isClosed():
                    # 从最后一个点连接回第一个点，形成封闭图形
                    line_path.lineTo(self._scale_point(self.points[0]))
            # --- 执行绘制 ---
            painter.drawPath(line_path)  # 绘制图形的线条
            if vrtx_path.length() > 0:  # 如果有普通顶点需要绘制
                painter.drawPath(vrtx_path)  # 绘制顶点的轮廓
                # 填充顶点的颜色
                painter.fillPath(vrtx_path, self._vertex_fill_color)  # type: ignore[has-type]
            # 如果需要填充图形内部 (self.fill is True)，并且不是 mask 类型
            if self.fill and self.mask is None:
                # 根据是否选中选择填充颜色
                color = self.select_fill_color if self.selected else self.fill_color
                # 使用 painter 填充线条路径内部
                painter.fillPath(line_path, color)

            # --- 绘制负样本点 (红色) ---
            # 设置画笔颜色为红色
            pen.setColor(QtGui.QColor(255, 0, 0, 255))
            painter.setPen(pen)
            # 绘制负样本点的轮廓
            painter.drawPath(negative_vrtx_path)
            # 填充负样本点的颜色 (红色)
            painter.fillPath(negative_vrtx_path, QtGui.QColor(255, 0, 0, 255))

    def drawVertex(self, path, i):
        """      绘制顶点
                这是一个辅助函数，负责准备绘制第 i 个顶点到指定的路径对象 (path)。
                它不直接绘制，而是将顶点的形状 (圆形或方形) 添加到 path 中。
                """
        d = self.point_size  # 获取顶点的基础大小
        shape = self.point_type  # 获取顶点的默认形状 (圆形或方形)
        point = self._scale_point(self.points[i])  # 获取第 i 个顶点坐标并进行缩放
        # 检查这个顶点是否是当前被高亮的顶点 (self._highlightIndex)
        if i == self._highlightIndex:
            # 如果是，就从 _highlightSettings 中获取高亮时的大小倍数 (size) 和形状 (shape)
            size, shape = self._highlightSettings[self._highlightMode]
            # 将顶点大小乘以倍数，让它看起来更大
            d *= size  # type: ignore[assignment]
        # 根据是否高亮，选择顶点的填充颜色
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color  # 使用高亮颜色
        else:
            self._vertex_fill_color = self.vertex_fill_color  # 使用普通颜色
        # 根据顶点的形状 (shape)，将对应的几何图形添加到传入的 path 对象中
        if shape == self.P_SQUARE:  # 如果是方形
            # 添加一个以 point 为中心，边长为 d 的正方形
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            # 添加一个以 point 为中心，半径为 d/2 的圆形
             path.addEllipse(point, d / 2, d / 2)
        else:
            # 断言：如果形状不是方形也不是圆形，就报错，说明程序逻辑有误
            assert False, "unsupported vertex shape"

    # --- 交互相关的方法 ---
    def nearestVertex(self, point, epsilon):
        """
                查找离给定点 point 最近的顶点。
                参数:
                    point: 用户鼠标点击或悬停的位置 (图像坐标)。
                    epsilon: 一个小的距离阈值，只有当顶点与 point 的距离小于 epsilon 时才被认为是“靠近”。
                返回:
                    如果找到了靠近的顶点，返回该顶点的索引 (整数)；否则返回 None。
                """
        min_distance = float("inf")
        min_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i, p in enumerate(self.points):
            p = QtCore.QPointF(p.x() * self.scale, p.y() * self.scale)
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        """
                查找离给定点 point 最近的边。
                返回:
                    如果找到了靠近的边，返回这条边结束点的索引 (整数)；否则返回 None。
        """
        min_distance = float("inf")
        post_i = None
        point = QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)
        for i in range(len(self.points)):
            start = self.points[i - 1]
            end = self.points[i]
            start = QtCore.QPointF(start.x() * self.scale, start.y() * self.scale)
            end = QtCore.QPointF(end.x() * self.scale, end.y() * self.scale)
            line = [start, end]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        """
                检查给定的点 point 是否位于这个图形的内部。
        """
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]
        return self.makePath().contains(point)

    def makePath(self):
        """
                根据图形的类型 (self.shape_type) 和顶点列表 (self.points)，
                创建一个 PyQt 的 QPainterPath 对象。
                QPainterPath 是 PyQt 中表示复杂二维形状的标准方式。
                """
        if self.shape_type in ["rectangle", "mask"]:
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                path.addRect(QtCore.QRectF(self.points[0], self.points[1]))
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                raidus = labelme.utils.distance(self.points[0] - self.points[1])
                path.addEllipse(self.points[0], raidus, raidus)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        """
                计算并返回能够完整包围这个图形的最小矩形 (Bounding Rectangle)。
                """
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        """
                将整个图形移动指定的偏移量。
                参数 offset 是一个 QPointF 对象，表示 x 和 y 方向的移动距离。
                """
        # 使用列表推导式，对 self.points 列表中的每一个点 p，都加上偏移量 offset
        # 然后用这个新的点列表替换掉旧的 self.points 列表
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        """
                只移动图形中指定索引 i 的顶点。
                """
        # 直接修改 self.points 列表中第 i 个顶点的值，让它等于原来的值加上偏移量 offset
        self.points[i] = self.points[i] + offset

    # --- 高亮相关的方法 ---
    def highlightVertex(self, i, action):
        """
                设置要高亮显示的顶点及其高亮模式。
                参数:
                    i (int): 要高亮的顶点的索引。
                    action (int): 高亮模式 (MOVE_VERTEX 或 NEAR_VERTEX)。
                """
        """Highlight a vertex appropriately based on the current action

        Args:
            i (int): The vertex index
            action (int): The action
            (see Shape.NEAR_VERTEX and Shape.MOVE_VERTEX)
        """
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        """清除所有顶点的高亮状态"""
        """Clear the highlighted point"""
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
