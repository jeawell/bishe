import copy
import numpy as np
from PyQt5 import QtCore, QtGui


class Brush:


    PEN_WIDTH = 50  # 默认画笔半径（兼容旧逻辑的类属性）
    line_color = None
    point_size = 1
    scale = 1.0

    def __init__(  #参数设置为None表示可选参数和默认值
        self,
        label=None,
        line_color=None,
        brush_type=None,
        flags=None,
        group_id=None,
        description=None,
    ):
        self.label = label
        self.group_id = group_id
        self.last_point = None
        self.point_labels = []
        # 规范：内部使用 brush_type 表示是画笔还是橡皮
        self._brush_type = brush_type or "pen"
        self.flags = flags
        self.description = description
        self.other_data = {}
        self._closed = False

        # 每个 Brush 实例自己的笔宽，默认继承类属性
        self.pen_width = self.PEN_WIDTH

        if line_color is not None:
            self.line_color = line_color

    ''': QtCore.QPointF：类型注解，表示参数应为QtCore.QPointF类型
    -> QtCore.QPointF:：返回类型注解，表示方法返回QtCore.QPointF类型'''
    def _scale_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(point.x() * self.scale, point.y() * self.scale)

    @property
    def brush_type(self):
        return self._brush_type

    @brush_type.setter
    def brush_type(self, value):
        if value is None:
            value = "pen"
        if value not in ['pen', 'eraser']:
            raise ValueError("Unexpected brush_type: {}".format(value))
        self._brush_type = value

    def close(self):
        self._closed = True

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def paint(self, painter, current_point: QtCore.QPointF):
        if self.last_point is None:
            self.last_point = current_point
            return

        distance = (current_point - self.last_point).manhattanLength()
        if distance < 1:
            return

        # 插值点数，步长控制绘制密度
        steps = int(distance / (self.PEN_WIDTH / 2)) + 1
        for i in range(steps + 1):
            t = i / steps
            interp_x = (1 - t) * self.last_point.x() + t * current_point.x()
            interp_y = (1 - t) * self.last_point.y() + t * current_point.y()
            pt = QtCore.QPointF(interp_x, interp_y)
            radius = self.pen_width / 2.0

            if self.brush_type == "eraser":
                # 使用 Source + 透明色，强制把这一块像素变成透明，以便真正“擦除” pixmap2 上的内容
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
                painter.setBrush(QtCore.Qt.transparent)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(pt, radius, radius)
            elif self.brush_type == "pen":
                # 如果没有提供颜色，使用一个安全的默认颜色，避免崩溃
                color_tuple = self.line_color or (255, 0, 0)
                qcolor = QtGui.QColor(*color_tuple)

                pen = QtGui.QPen()
                pen.setWidth(0)
                pen.setColor(qcolor)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(qcolor))
                painter.drawEllipse(pt, radius, radius)  # 绘制椭圆

        self.last_point = current_point

    def copy(self):
        return copy.deepcopy(self)
