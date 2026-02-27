from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets


class Penwidget(QtWidgets.QSpinBox):
    def __init__(self, value=20):
        super(Penwidget, self).__init__()
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setRange(1, 300)  # 合理的画笔宽度范围
        self.setValue(value)
        self.setToolTip("Pen Width")
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter) # type: ignore[attr-defined]

    # def minimumSizeHint(self):
    #     height = super(Penwidget, self).minimumSizeHint().height()
    #     fm = QtGui.QFontMetrics(self.font())
    #     width = fm.width(str(self.maximum()))
    #     return QtCore.QSize(width, height)
