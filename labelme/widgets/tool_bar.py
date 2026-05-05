from PyQt5 import QtCore
from PyQt5 import QtWidgets

TOOLBAR_BUTTON_STYLE = """
QToolButton {
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px;
    background-color: transparent;
}
QToolButton:hover {
    background-color: rgba(100, 170, 230, 0.25);
    border: 1px solid rgba(100, 170, 230, 0.5);
}
QToolButton:pressed {
    background-color: rgba(70, 140, 200, 0.40);
    border: 1px solid rgba(70, 140, 200, 0.7);
}
QToolButton:checked {
    background-color: rgba(70, 140, 200, 0.30);
    border: 1px solid rgba(70, 140, 200, 0.6);
}
QToolButton:disabled {
    opacity: 0.4;
}
"""


class ToolBar(QtWidgets.QToolBar):
    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)  # type: ignore[union-attr]
        layout.setContentsMargins(*m)  # type: ignore[union-attr]
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)  # type: ignore[attr-defined]

    def addAction(self, action):  # type: ignore[override]
        if isinstance(action, QtWidgets.QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        btn.setFont(self.font())
        btn.setStyleSheet(TOOLBAR_BUTTON_STYLE)
        self.addWidget(btn)

        # center align
        for i in range(self.layout().count()):  # type: ignore[union-attr]
            if isinstance(self.layout().itemAt(i).widget(), QtWidgets.QToolButton):  # type: ignore[union-attr]
                self.layout().itemAt(i).setAlignment(QtCore.Qt.AlignCenter)  # type: ignore[attr-defined,union-attr]
