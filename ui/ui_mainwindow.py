# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(804, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 70, 397, 481))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.address_text = QtGui.QPlainTextEdit(self.groupBox)
        self.address_text.setGeometry(QtCore.QRect(10, 50, 311, 31))
        self.address_text.setObjectName(_fromUtf8("address_text"))
        self.train_button = QtGui.QPushButton(self.groupBox)
        self.train_button.setGeometry(QtCore.QRect(10, 440, 81, 31))
        self.train_button.setAutoRepeat(False)
        self.train_button.setAutoDefault(False)
        self.train_button.setObjectName(_fromUtf8("train_button"))
        self.kfold_text = QtGui.QPlainTextEdit(self.groupBox)
        self.kfold_text.setGeometry(QtCore.QRect(110, 190, 61, 31))
        self.kfold_text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.kfold_text.setAutoFillBackground(False)
        self.kfold_text.setObjectName(_fromUtf8("kfold_text"))
        self.kfold_error = QtGui.QTextBrowser(self.groupBox)
        self.kfold_error.setGeometry(QtCore.QRect(300, 130, 61, 31))
        self.kfold_error.setObjectName(_fromUtf8("kfold_error"))
        self.oob_error = QtGui.QTextBrowser(self.groupBox)
        self.oob_error.setGeometry(QtCore.QRect(300, 190, 61, 31))
        self.oob_error.setObjectName(_fromUtf8("oob_error"))
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(200, 190, 81, 20))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(230, 130, 51, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.kfold_label = QtGui.QLabel(self.groupBox)
        self.kfold_label.setGeometry(QtCore.QRect(20, 200, 46, 13))
        self.kfold_label.setObjectName(_fromUtf8("kfold_label"))
        self.number_of_trees = QtGui.QPlainTextEdit(self.groupBox)
        self.number_of_trees.setGeometry(QtCore.QRect(110, 130, 61, 31))
        self.number_of_trees.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.number_of_trees.setAutoFillBackground(False)
        self.number_of_trees.setObjectName(_fromUtf8("number_of_trees"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 130, 91, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(20, 30, 51, 16))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.open_button = QtGui.QPushButton(self.groupBox)
        self.open_button.setGeometry(QtCore.QRect(330, 50, 51, 31))
        self.open_button.setObjectName(_fromUtf8("open_button"))
        self.groupBox_3 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(410, 280, 381, 271))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.train_deeplearning_button = QtGui.QPushButton(self.groupBox_3)
        self.train_deeplearning_button.setGeometry(QtCore.QRect(30, 230, 81, 31))
        self.train_deeplearning_button.setObjectName(_fromUtf8("train_deeplearning_button"))
        self.label_5 = QtGui.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(30, 90, 71, 16))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_6 = QtGui.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(30, 130, 47, 13))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.label_7 = QtGui.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(30, 170, 81, 16))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.hidden_layer_text = QtGui.QPlainTextEdit(self.groupBox_3)
        self.hidden_layer_text.setGeometry(QtCore.QRect(110, 80, 81, 31))
        self.hidden_layer_text.setObjectName(_fromUtf8("hidden_layer_text"))
        self.lr_text = QtGui.QPlainTextEdit(self.groupBox_3)
        self.lr_text.setGeometry(QtCore.QRect(110, 160, 81, 31))
        self.lr_text.setObjectName(_fromUtf8("lr_text"))
        self.epochs_text = QtGui.QPlainTextEdit(self.groupBox_3)
        self.epochs_text.setGeometry(QtCore.QRect(110, 120, 81, 31))
        self.epochs_text.setObjectName(_fromUtf8("epochs_text"))
        self.deep_learning_address_text = QtGui.QPlainTextEdit(self.groupBox_3)
        self.deep_learning_address_text.setGeometry(QtCore.QRect(10, 30, 311, 31))
        self.deep_learning_address_text.setObjectName(_fromUtf8("deep_learning_address_text"))
        self.open_button_2 = QtGui.QPushButton(self.groupBox_3)
        self.open_button_2.setGeometry(QtCore.QRect(330, 30, 41, 31))
        self.open_button_2.setObjectName(_fromUtf8("open_button_2"))
        self.groupBox_2 = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(410, 10, 381, 271))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.mfcc_radioButton = QtGui.QRadioButton(self.groupBox_2)
        self.mfcc_radioButton.setGeometry(QtCore.QRect(300, 50, 61, 17))
        self.mfcc_radioButton.setObjectName(_fromUtf8("mfcc_radioButton"))
        self.filterbank_radioButton = QtGui.QRadioButton(self.groupBox_2)
        self.filterbank_radioButton.setGeometry(QtCore.QRect(300, 20, 82, 17))
        self.filterbank_radioButton.setObjectName(_fromUtf8("filterbank_radioButton"))
        self.progressBar = QtGui.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 20, 251, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.wait_label = QtGui.QLabel(self.centralwidget)
        self.wait_label.setGeometry(QtCore.QRect(270, 20, 61, 20))
        self.wait_label.setObjectName(_fromUtf8("wait_label"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 804, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.groupBox.setTitle(_translate("MainWindow", "RandomForest", None))
        self.address_text.setPlainText(_translate("MainWindow", "duetto.csv", None))
        self.train_button.setText(_translate("MainWindow", "Train", None))
        self.kfold_text.setPlainText(_translate("MainWindow", "3", None))
        self.label_3.setText(_translate("MainWindow", "Out of bag Error", None))
        self.label_2.setText(_translate("MainWindow", "Kfold Error", None))
        self.kfold_label.setText(_translate("MainWindow", "K-Fold", None))
        self.number_of_trees.setPlainText(_translate("MainWindow", "100", None))
        self.label.setText(_translate("MainWindow", "Number of Trees", None))
        self.label_4.setText(_translate("MainWindow", "Audio Files", None))
        self.open_button.setText(_translate("MainWindow", "Open", None))
        self.groupBox_3.setTitle(_translate("MainWindow", "Deep Learning", None))
        self.train_deeplearning_button.setText(_translate("MainWindow", "Train", None))
        self.label_5.setText(_translate("MainWindow", "Hidden Layer", None))
        self.label_6.setText(_translate("MainWindow", "Epochs", None))
        self.label_7.setText(_translate("MainWindow", "Learning Rate", None))
        self.hidden_layer_text.setPlainText(_translate("MainWindow", "2000", None))
        self.lr_text.setPlainText(_translate("MainWindow", "0.0001", None))
        self.epochs_text.setPlainText(_translate("MainWindow", "10", None))
        self.deep_learning_address_text.setPlainText(_translate("MainWindow", "mfcc.csv", None))
        self.open_button_2.setText(_translate("MainWindow", "Open", None))
        self.groupBox_2.setTitle(_translate("MainWindow", "Extractor", None))
        self.mfcc_radioButton.setText(_translate("MainWindow", "MFCC", None))
        self.filterbank_radioButton.setText(_translate("MainWindow", "FilterBank", None))
        self.wait_label.setText(_translate("MainWindow", "Wait", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))

