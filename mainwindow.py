__author__ = 'Rey'

import sys
from ui.ui_mainwindow import Ui_MainWindow
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from Classifiers.RandomForest import RandomForest
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from Extractors.Spectral.mfcc import mfcc
from Extractors.RBM import RBM
import pandas as pd


class MainWindow(QMainWindow,Ui_MainWindow):

    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)

        # train_button = QPushButton('train_button')
        # train_button.clicked.connect(self.on_train_button_clicked)

        self.setupUi(self)
        self.database = './GeneratedFeatures/duetto.csv'
        self.progress = 0



        self.wait_label.hide()


    @pyqtSlot()
    def on_train_button_clicked(self):

        self.wait_label.show()
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        self.progress=0

        self.database = self.address_text.toPlainText()
        trees = int(self.number_of_trees.toPlainText())
        try:
            df = pd.read_csv('./GeneratedFeatures/'+self.database)
            Y = pd.factorize(df.Category)[0]
            X = df.drop('Category',axis=1).values

            errors = []
            kfold = KFold(len(Y),int(self.kfold_text.toPlainText()),shuffle=True)

            self.progressBar.setMaximum(kfold.n_folds+1)

            for idx_train,idx_test in kfold:
                rf = RandomForest(n_estimators=trees)
                rf.train(X[idx_train,:],Y[idx_train])
                pred = rf.classify(X[idx_test,:])
                errors.append(accuracy_score(Y[idx_test],pred))

                self.progress += 1
                self.progressBar.setValue(self.progress)



            print(np.mean(errors))
            #oob error

            rf = RandomForest(n_estimators=trees)
            rf.train(X,Y)

            self.progress += 1
            self.progressBar.setValue(self.progress)

            self.kfold_error.setPlainText(str(np.round(np.mean(errors), 3)))
            self.oob_error.setPlainText(str(np.round(rf.oob_score,3)))

            self.wait_label.hide()

        except:
            print('Exception')

    @pyqtSlot()
    def on_open_button_clicked(self):
        files = QFileDialog.getOpenFileNames(self, 'Open File',"./GeneratedFeatures", '(*.csv)')
        self.address_text.setPlainText(files[0].split('\\')[-1])

    @pyqtSlot()
    def on_open_button_2_clicked(self):
        files = QFileDialog.getOpenFileNames(self, 'Open File', "./GeneratedFeatures", '(*.csv)')
        self.deep_learning_address_text.setPlainText(files[0].split('\\')[-1])

    @pyqtSlot()
    def on_train_deeplearning_button_clicked(self):

        self.wait_label.show()

        self.database = self.deep_learning_address_text.toPlainText()
        df = pd.read_csv('.//GeneratedFeatures//'+self.database)

        X = df.drop('Category',axis=1).values

        rbm = RBM(X.shape[1],int(self.hidden_layer_text.toPlainText()),float(self.lr_text.toPlainText()),int(self.epochs_text.toPlainText()),\
                  type_visible_layer='G')
        rbm.fit(X)
        X = rbm.transform(X)

        new_df = pd.DataFrame(data=X)
        new_df['Category'] = df.Category
        new_df.to_csv('.//GeneratedFeatures//rbm_'+self.database)

        self.wait_label.hide()




app = QApplication(sys.argv)
form = MainWindow()
form.show()
app.exec_()


# df = pd.read_csv('./GeneratedFeatures/duetto.csv')
#
# df.drop('Freq',axis=1,inplace=True)
# df.to_csv('./GeneratedFeatures/duetto.csv')
# print(df.head(3))