import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenuBar, QMenu,
                             QAction, QPlainTextEdit, QStyle, QFileDialog,
                             QMessageBox)


class DemoNotepad(QMainWindow):
    def __init__(self, parent=None):
        super(DemoNotepad, self).__init__(parent)

        # 设置窗口标题
        self.setWindowTitle('实战PyQt5: QAction Demo-记事本')
        # 设置窗口大小
        self.resize(480, 360)

        self.path = None

        self.initUi()

    def initUi(self):
        # 设置一个文本编辑器作为中心小部件
        self.txtEditor = QPlainTextEdit(self)

        self.setCentralWidget(self.txtEditor)  # 设置为中心控件

        self.initMenuBar()

    def initMenuBar(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('文件(&F)')
        editMenu = menuBar.addMenu('编辑(&E)')
        formatMenu = menuBar.addMenu('格式(&O)')
        helpMenu = menuBar.addMenu('帮助(&H)')

        style = QApplication.style()

        # ==== 文件操作部分 ==== #
        # 新建文件
        aFileNew = QAction('新建(&N)', self)
        # 添加一个图标
        aFileNew.setIcon(style.standardIcon(QStyle.SP_FileIcon))
        # 添加快捷键
        aFileNew.setShortcut(Qt.CTRL + Qt.Key_N)
        aFileNew.triggered.connect(self.onFileNew)
        fileMenu.addAction(aFileNew)
        # 打开文件
        aFileOpen = QAction('打开(&O)...', self)
        aFileOpen.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
        aFileOpen.setShortcut(Qt.CTRL + Qt.Key_O)
        aFileOpen.triggered.connect(self.onFileOpen)
        fileMenu.addAction(aFileOpen)

        # 保存
        aFileSave = QAction('保存(&S)', self)
        aFileSave.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        aFileSave.setShortcut(Qt.CTRL + Qt.Key_S)
        aFileSave.triggered.connect(self.onFileSave)
        fileMenu.addAction(aFileSave)
        # 另存为
        aFileSaveAs = QAction('另存为(&A)...', self)
        aFileSaveAs.triggered.connect(self.onFileSaveAs)
        fileMenu.addAction(aFileSaveAs)
        # 间隔线
        fileMenu.addSeparator()
        # 退出菜单
        aFileExit = QAction('退出(&X)', self)
        aFileExit.triggered.connect(self.close)
        fileMenu.addAction(aFileExit)

        # ==== 编辑部分 ==== #
        # 撤销编辑
        aEditUndo = QAction('撤销(&U)', self)
        aEditUndo.setShortcut(Qt.CTRL + Qt.Key_Z)
        aEditUndo.triggered.connect(self.txtEditor.undo)
        editMenu.addAction(aEditUndo)
        # 恢复编辑
        aEditRedo = QAction('恢复(&R)', self)
        aEditRedo.setShortcut(Qt.CTRL + Qt.Key_Y)
        aEditUndo.triggered.connect(self.txtEditor.redo)
        editMenu.addAction(aEditRedo)
        # 间隔线
        editMenu.addSeparator()
        # 剪切操作
        aEditCut = QAction('剪切(&T)', self)
        aEditCut.setShortcut(Qt.CTRL + Qt.Key_X)
        aEditCut.triggered.connect(self.txtEditor.cut)
        editMenu.addAction(aEditCut)
        # 复制操作
        aEditCopy = QAction('复制(&C)', self)
        aEditCopy.setShortcut(Qt.CTRL + Qt.Key_C)
        aEditCopy.triggered.connect(self.txtEditor.copy)
        editMenu.addAction(aEditCopy)
        # 粘贴操作
        aEditPaste = QAction('粘贴(&P)', self)
        aEditPaste.setShortcut(Qt.CTRL + Qt.Key_V)
        aEditPaste.triggered.connect(self.txtEditor.paste)
        editMenu.addAction(aEditPaste)
        # 删除操作
        aEditDel = QAction('删除(&L)', self)
        aEditDel.setShortcut(Qt.Key_Delete)
        aEditDel.triggered.connect(self.onEditDelete)
        editMenu.addAction(aEditDel)
        # 添加分割线
        editMenu.addSeparator()
        # 全选操作
        aEditSelectAll = QAction('全选(&A)', self)
        aEditSelectAll.setShortcut(Qt.CTRL + Qt.Key_A)
        aEditSelectAll.triggered.connect(self.txtEditor.selectAll)
        editMenu.addAction(aEditSelectAll)

        # ==== 格式设置部分 ==== #
        aFmtAutoLine = QAction('自动换行(&W)', self)
        aFmtAutoLine.setCheckable(True)
        aFmtAutoLine.setChecked(True)
        aFmtAutoLine.triggered[bool].connect(self.onFormatAutoLine)
        formatMenu.addAction(aFmtAutoLine)

        # ==== 帮助部分 ==== #
        aHelpAbout = QAction('关于(&A)...', self)
        aHelpAbout.triggered.connect(self.onHelpAbout)
        helpMenu.addAction(aHelpAbout)

    def msgCritical(self, strInfo):
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Critical)
        dlg.setText(strInfo)
        dlg.show()

    def onFileNew(self):
        self.txtEditor.clear()

    def onFileOpen(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开文件', '', '文本文件 (*.txt)')
        if path:
            try:
                with open(path, 'rU') as f:
                    text = f.read()
            except Exception as e:
                self.msgCritical(str(e))
            else:
                self.path = path
                self.txtEditor.setPlainText(text)

    def onFileSave(self):
        if self.path is None:
            return self.onFileSaveAs()
        self._saveToPath(self.path)

    def onFileSaveAs(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存文件', '', '文本文件 (*.txt)')
        if not path:
            return
        self._saveToPath(path)

    def _saveToPath(self, path):
        text = self.txtEdit.toPlainText()
        try:
            with open(path, 'w') as f:
                f.write(text)
        except Exception as e:
            self.msgCritical(str(e))
        else:
            self.path = path

    def onEditDelete(self):
        tc = self.txtEditor.textCursor()
        # tc.select(QtGui.QTextCursor.BlockUnderCursor) 这样删除一行
        tc.removeSelectedText()

    def onFormatAutoLine(self, autoLine):
        if autoLine:
            self.txtEditor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        else:
            self.txtEditor.setLineWrapMode(QPlainTextEdit.NoWrap)

    def onHelpAbout(self):
        QMessageBox.information(self, '实战PyQt5', 'PyQt5实现的文本编辑器演示版')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DemoNotepad()
    window.show()
    sys.exit(app.exec())