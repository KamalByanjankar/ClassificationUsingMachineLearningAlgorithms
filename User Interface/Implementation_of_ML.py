import sys
import os
from os.path import basename
from glob import glob
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication, QSizePolicy
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QVBoxLayout, QHeaderView
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import to_categorical
import pickle



class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        #self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
    

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def plot(self,test_y,result,title):
        self.ax.remove()
        self.ax = self.fig.add_subplot(111)
        cm = confusion_matrix(test_y, result)
        sum = np.sum(cm)
        score = accuracy_score(test_y, result)
    
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_CLASS_A = round(precision_score(test_y, result, average='binary',pos_label=0),2)
        precision_CLASS_B = round(precision_score(test_y, result, average='binary',pos_label=1),2)
        recall_CLASS_A = round(recall_score(test_y, result, average='binary',pos_label=0),2)
        recall_CLASS_B = round(recall_score(test_y, result, average='binary',pos_label=1),2)
        print('Precision: Class A',precision_CLASS_A)
        print('Precision: Class B',precision_CLASS_B)
    
        cm_new = np.append(cm[0], recall_CLASS_A)
        cm_new2 = np.append(cm[1], recall_CLASS_B)
        cm_new3 = np.array([precision_CLASS_A, precision_CLASS_B, score])
        cm = np.array([cm_new,cm_new2,cm_new3])
        
#        fig, ax = plt.subplots(figsize=(10,10))
#        ax = self.figure.add_subplot(111)
        sns.heatmap(cm, annot=True, ax = self.ax, linewidths=.5,fmt='g',cmap="Greens");
        
        
        # labels, title and ticks
        self.ax.set_xlabel('Predicted labels');
        self.ax.set_ylabel('True labels'); 
        self.ax.set_title(title); 
        counter = 0
        for i in range(0,2):
            for j in range(0,3):
                percentage = cm[i,j]/sum
                t = self.ax.texts[counter]
                if j == 2:
                    t.set_text(str(cm[i,j]))
                else:
                    t.set_text(str(cm[i,j]) + '\n' + str(round(percentage*100,2)) + " %")
                counter = counter + 1
        labels = ['HUMAN', 'NON HUMAN']
        self.ax.xaxis.set_ticklabels(labels)
        self.ax.yaxis.set_ticklabels(labels);
        self.ax.plot(cm)
        self.draw()


    def plot_1(self, test_y, result, labels, title):
        self.ax.remove()
        self.ax = self.fig.add_subplot(111)
        cm = confusion_matrix(test_y, result)
        sum = np.sum(cm)
        score = accuracy_score(test_y, result)
    
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_CLASS_A = round(precision_score(test_y, result, average='binary',pos_label=labels[0]),2)
        precision_CLASS_B = round(precision_score(test_y, result, average='binary',pos_label=labels[1]),2)
        recall_CLASS_A = round(recall_score(test_y, result, average='binary',pos_label=labels[0]),2)
        recall_CLASS_B = round(recall_score(test_y, result, average='binary',pos_label=labels[1]),2)
        print('Precision: Class A',precision_CLASS_A)
        print('Precision: Class B',precision_CLASS_B)
    
        cm_new = np.append(cm[0], recall_CLASS_A)
        cm_new2 = np.append(cm[1], recall_CLASS_B)
        cm_new3 = np.array([precision_CLASS_A, precision_CLASS_B, score])
        cm = np.array([cm_new,cm_new2,cm_new3])
        
        
        sns.heatmap(cm, annot=True, ax = self.ax, linewidths=.5,fmt='g',cmap="Greens");
        
        
        # labels, title and ticks
        self.ax.set_xlabel('Predicted labels');
        self.ax.set_ylabel('True labels'); 
        self.ax.set_title(title); 
        counter = 0
        for i in range(0,2):
            for j in range(0,3):
                percentage = cm[i,j]/sum
                t = self.ax.texts[counter]
                if j == 2:
                    t.set_text(str(cm[i,j]))
                else:
                    t.set_text(str(cm[i,j]) + '\n' + str(round(percentage*100,2)) + " %")
                counter = counter + 1
                
        self.ax.xaxis.set_ticklabels(labels)
        self.ax.yaxis.set_ticklabels(labels);
        self.ax.plot(cm)
        self.draw()

class Widget(QWidget):
    def __init__(self):
        print('Machine Learning Implementation')
        super().__init__()
        uifile = os.path.join(os.path.dirname(__file__), 'Implementation_of_ML.ui')
        self.ui = loadUi(uifile, self)
        
        w = 5
        h = 4
        
        self.m = PlotCanvas(self, width=w, height=h)
#        self.m.move(500,50)
        
        self.n = PlotCanvas(self, width=w, height=h)
#        self.n.move(950,50) 
        
        self.o = PlotCanvas(self, width=w, height=h)
#        self.o.move(1400,50) 
        
        self.files=[]
        self.required_data=[]
        self.li=[]
        self.li2=[]
        self.frame=[]
        self.frame2=[]
        self.normalized_X_train=[]
        self.normalized_X_test=[]
        self.train_y=[]
        self.test_y=[]
        self.labels=[]
#        self.filename_random_forest=[]
        
        for elem in self.ui.children():
            name = elem.objectName()
            
            if name == 'ML_frame':
                for child_elem in elem.children():
                    child_name = child_elem.objectName()
                    
                    if child_name == 'browse_file':
                        child_elem.clicked.connect(self.load_file)
                    elif child_name == 'table_widget':
                        self.table = child_elem
                    elif child_name == 'browse_file_input':
                        self.browse_file_input = child_elem
                    elif child_name == 'RF_train_btn':
                        child_elem.clicked.connect(self.train_RF)
                    elif child_name == 'KNN_train_btn':                        
                        child_elem.clicked.connect(self.train_KNN)
                    elif child_name == 'CNN_train_btn':
                        child_elem.clicked.connect(self.train_CNN)
                    elif child_name == 'confusion_matrix_btn_RF':
                        child_elem.clicked.connect(self.confusion_matrix_RF)
                    elif child_name == 'confusion_matrix_btn_KNN':
                        child_elem.clicked.connect(self.confusion_matrix_KNN)
                    elif child_name == 'confusion_matrix_btn_CNN':
                        child_elem.clicked.connect(self.confusion_matrix_CNN)
                    elif child_name == 'save_RF_model':
                        child_elem.clicked.connect(self.save_model_random_forest)
                        
            elif name == 'plot_frame':
                for child_elem in elem.children():
                    child_name = child_elem.objectName()
#                    print(child_name,child_elem,child_elem.children()[0].objectName())
                    
                    if child_elem.children()[0].objectName() == 'rf_plot':
                        child_elem.children()[0].addWidget(self.m)                    
                    elif child_elem.children()[0].objectName() == 'knn_plot':
                        child_elem.children()[0].addWidget(self.n)
                    elif child_elem.children()[0].objectName() == 'cnn_plot':
                        child_elem.children()[0].addWidget(self.o)
                        
    def load_file(self):
        files=[]
        self.directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        paths = glob(f'{self.directory}/*')
        self.browse_file_input.setText(self.directory)
        for file in paths:
            if len(paths) > 0:
                for p in paths:
                    files += glob(f'{p}/*.csv')
                if not files:
                    files = paths
                
                self.files.append(file)
        self.load_files_in_table(files)
        
    def load_files_in_table(self, files):
        self.table.setRowCount(len(files))
        for row, item in enumerate(files):
            filename = item.split('/')[-1]
            if filename.split('\\')[0]:
                filename = filename.split('\\')[0]
            self.table.setItem(row, 0, QTableWidgetItem(filename))
            self.table.setItem(row, 1, QTableWidgetItem(basename(item)))
              
    def show_data(self):
        for file in self.files:
            data = pd.read_csv(file, index_col=None, header=0)
            self.li.append(data)
        
        frame = pd.concat(self.li, axis=0, ignore_index=True)
        print("Total data", frame.shape)
        
        car = frame.loc[frame['type'] == 'CAR'].iloc[:, 4:]
        human = frame.loc[frame['type'] == 'HUMAN'].iloc[:, 4:]
        pillar = frame.loc[frame['type'] == 'PILLAR'].iloc[:, 4:]
        wall = frame.loc[frame['type'] == 'WALL'].iloc[:, 4:]
        
#        print(car.shape)
#        print(human.shape)
#        print(pillar.shape)
#        print(wall.shape)
        
        human_label = ['HUMAN']*human.shape[0]
        non_human_label = ['NON_HUMAN']*(car.shape[0] + wall.shape[0] + pillar.shape[0])
        label = human_label + non_human_label
        print("Label", np.array(label).shape)
        data = human.values.tolist() + car.values.tolist() + wall.values.tolist() + pillar.values.tolist()
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(data, label, test_size=0.33, random_state=42)
        
        print('X train data:', np.array(self.train_X).shape)
        print('X test data:', np.array(self.test_X).shape)
        print('y train data:', np.array(self.train_y).shape)
        print('y test data:', np.array(self.test_y).shape)
        
        
        self.normalized_X_train = preprocessing.normalize(self.train_X)
        self.normalized_X_test = preprocessing.normalize(self.test_X)
        
        
    def CNN_show_data(self):
        for file in self.files:
            data = pd.read_csv(file, index_col=None, header=0)
            self.li2.append(data)
        
        frame2 = pd.concat(self.li2, axis=0, ignore_index=True)
        print("hello", frame2.shape)
        
        car = frame2.loc[frame2['type'] == 'CAR'].iloc[:, 4:]
        human = frame2.loc[frame2['type'] == 'HUMAN'].iloc[:, 4:]
        pillar = frame2.loc[frame2['type'] == 'PILLAR'].iloc[:, 4:]
        wall = frame2.loc[frame2['type'] == 'WALL'].iloc[:, 4:]
        
#        print(car.shape)
#        print(human.shape)
#        print(pillar.shape)
#        print(wall.shape)
        
        human_label = ['HUMAN']*human.shape[0]
        non_human_label = ['NON_HUMAN']*(car.shape[0] + wall.shape[0] + pillar.shape[0])
        label = human_label + non_human_label
        print("Label", np.array(label).shape)
        data = human.values.tolist() + car.values.tolist() + wall.values.tolist() + pillar.values.tolist()
        
        from sklearn.preprocessing import LabelBinarizer, LabelEncoder
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(label)
        train_X, self.test_X, train_y, self.y_test = train_test_split(data, label_encoded, test_size=0.33, random_state=42)
        
        print('X train data:', np.array(train_X).shape)
        print('X test data:', np.array(self.test_X).shape)
        print('y train data:', np.array(train_y).shape)
        print('y test data:', np.array(self.y_test).shape)
        
        
        normalized_X_train = preprocessing.normalize(train_X)
        normalized_X_test = preprocessing.normalize(self.test_X)
        
        train_X = np.array(normalized_X_train)[:,6:31].reshape(-1,5,5,1)
        self.test_X =  np.array(normalized_X_test)[:,6:31].reshape(-1,5,5,1)
        classes = np.unique(train_y)
        nClasses = len(classes)
        print('Total number of outputs : ', nClasses)
        print('Output classes : ', classes)
        
        train_X = train_X.astype('float32')
        self.test_X = self.test_X.astype('float32')
        print(train_X.shape, self.test_X.shape, train_y.shape, self.y_test.shape)

        # Change the labels from categorical to one-hot encoding
        train_Y_one_hot = keras.utils.to_categorical(train_y)
        test_Y_one_hot = keras.utils.to_categorical(self.y_test)
        
        # Display the change for category label using one-hot encoding
        print('Original label:', train_y[0])
        print('After conversion to one-hot:', train_Y_one_hot[0])
        
        print(train_Y_one_hot.shape)
        print(test_Y_one_hot.shape)
        
        train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.3, random_state=13)
        train_X.shape,valid_X.shape,train_label.shape,valid_label.shape
        
        epochs = 50
        num_classes = 2
        
        self.model = Sequential()
        
        self.model.add(Conv2D(128, kernel_size=(2, 2),input_shape=(5,5,1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2),padding='same'))
        
        self.model.add(Flatten())
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.train_dropout = self.model.fit(train_X, train_label,epochs=epochs,validation_data=(valid_X, valid_label))
        
        print("CNN Training Complete")
        
    def train_RF(self):
        print("Training Random Forest ...")
        self.show_data()
        
        self.clf = RandomForestClassifier(n_estimators=100)
        self.clf.fit(self.normalized_X_train, self.train_y)
        
        print("Random Forest Training complete")
        
        
        
    def RFMatrix(self):
#        print("Loading Random Forest Model")
#        filename_random_forest = './models/random_forest.sav'
#        loaded_model = pickle.load(open(filename_random_forest, 'rb'))
#        model_result = loaded_model.score(self.normalized_X_test, self.test_y)
#        print(model_result)

        result = self.clf.predict(self.normalized_X_test)
        cm = confusion_matrix(self.test_y, result)
        print(cm)
        print("Accuracy: ",accuracy_score(self.test_y, result))
        labels = ['HUMAN', 'NON_HUMAN']
        title = ('Random Forest Confusion Matrix')
        self.m.plot_1(self.test_y, result, labels, title)
        self.li=[]
        
        
    def train_KNN(self):
        print("Training KNN ...")
#        self.show_data()
        self.knn = KNeighborsClassifier(n_neighbors=2)
        self.knn.fit(self.normalized_X_train, self.train_y)
        
        print("KNN Training complete")

        
        
    def KNNMatrix(self):
        result = self.knn.predict(self.normalized_X_test)        
        cm = confusion_matrix(self.test_y, result)
        print(cm)
        print("Accuracy: ",accuracy_score(self.test_y, result))
        labels = ['HUMAN', 'NON_HUMAN']
        title = ('KNN Confusion Matrix')
        self.n.plot_1(self.test_y, result, labels, title)
        
    def train_CNN(self):
        print("Training CNN ...")
        self.CNN_show_data()

        
    def CNNMatrix(self):
        predicted_classes = self.model.predict(self.test_X)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
        print("ZZZ", predicted_classes.shape, self.y_test.shape)
        title = ('CNN Confusion Matrix')
        self.o.plot(self.y_test, predicted_classes,title)
        
    
    def confusion_matrix_RF(self):
        self.RFMatrix()
        
        
    def confusion_matrix_KNN(self):
        self.KNNMatrix()
        
    def confusion_matrix_CNN(self):
        self.CNNMatrix()
        
#    def save_model_knn(self):
#        filename = './models/KNN.sav'
#        pickle.dump(self.knn, open(filename, 'wb'))
#        
#    def save_model_random_forest(self):
#        print("Saving Random Forest Model")
#        filename_random_forest = './models/random_forest.sav'
#        pickle.dump(self.clf, open(filename_random_forest, 'wb'))
#        print("Random Forest Model Saved")
#    
#    def save_model_cnn(self):
#        filename = './models/CNN.sav'
#        pickle.dump(self.model, open(filename, 'wb'))
    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())