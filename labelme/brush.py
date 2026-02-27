import copy
import numpy as np
from PyQt5 import QtCore, QtGui


class Brush:


    PEN_WIDTH = 50
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
        self._brush_type = brush_type
        self.flags = flags
        self.description = description
        self.other_data = {}
        self._closed = False

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
    def shape_type(self, value):
        if value is None:
            value = "pen"
        if value not in ['pen', 'eraser']:
            raise ValueError("Unexpected shape_type: {}".format(value))
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

            if self.shape_type == "eraser":
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
                painter.setBrush(QtCore.Qt.transparent)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(pt, self.PEN_WIDTH / 2, self.PEN_WIDTH / 2)
            elif self.shape_type == "pen":
                pen = QtGui.QPen()
                pen.setWidth(0)
                pen.setColor(QtGui.QColor(*self.line_color))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(*self.line_color)))
                painter.drawEllipse(pt, self.PEN_WIDTH / 2, self.PEN_WIDTH / 2)#绘制椭圆方法

        self.last_point = current_point

    def copy(self):
        return copy.deepcopy(self)
