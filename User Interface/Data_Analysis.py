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
from random import randrange


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        #self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def plot(self, data_list,title, randNum):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.set_title(title)
        ax.plot(data_list[randNum])
        self.draw()     
        
    def fft_plot(self, freq, data, title):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.set_xlim([30000, 50000])
        ax.set_title(title)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('FFT')
        ax.plot(freq, data)
        self.draw() 

class Widget(QWidget):
    def __init__(self):
        print("Hello!! Welcome to the Data Analysis")
        
        super().__init__()
        uifile = os.path.join(os.path.dirname(__file__), 'Data_Analysis.ui')
        self.ui = loadUi(uifile, self)
        
        w = 6
        h = 4
        
        self.m = PlotCanvas(self, width=w, height=h)
        self.m.move(560,50)
        
        
        self.n = PlotCanvas(self, width=w, height=h)
        self.n.move(1180,50) 
        
        self.o = PlotCanvas(self, width=w, height=h)
        self.o.move(880,500) 
        
        self.files=[]
        self.required_data_without_offset=[]
        self.required_echo_list = []
        self.required_fft_data_frame = []
        self.signal_set= []
        
        for elem in self.ui.children():
            name = elem.objectName()
                
            if name == 'echo_frame':
                for child_elem in elem.children():
                    child_name = child_elem.objectName()
                    
                    if child_name == 'browse_file':
                        child_elem.clicked.connect(self.load_file)
                    elif child_name == 'table_widget':
                        self.table = child_elem
                    elif child_name == 'browse_file_input':
                        self.browse_file_input = child_elem
                    elif child_name == 'show_plot_btn':
                        child_elem.clicked.connect(self.plot_graph)
#                    elif child_name == 'show_echo_btn':
#                        child_elem.clicked.connect(self.plot_echo)
#                    elif child_name == 'save_echo_btn': 
#                        child_elem.clicked.connect(self.save_echo)
                    elif child_name == 'plot_fft_btn':
                        child_elem.clicked.connect(self.plot_fft)
                    elif child_name == 'save_feature_btn':
                        child_elem.clicked.connect(self.save_features)
                        
    
    def load_file(self):
        files = []
        self.directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        paths = glob(f'{self.directory}/*')
        self.browse_file_input.setText(self.directory)
        #self.text_echo_filename.setText(self.directory)
        if len(paths) > 0:
            for p in paths:
                files += glob(f'{p}/*.csv')
            if not files:
                files = paths
            #for i in range(len(paths)):
            self.files.append(paths[0])
        self.load_files_in_table(files)

        
    def load_files_in_table(self, files):
        self.table.setRowCount(len(files))
        for row, item in enumerate(files):
            filename = item.split('/')[-1]
            if filename.split('\\')[0]:
                filename =  filename.split('\\')[0]
            self.table.setItem(row, 0, QTableWidgetItem(filename))
            self.table.setItem(row, 1, QTableWidgetItem(basename(item)))
            
        
    def plot_graph(self):
        for file in self.files:
            print(file)
            df = pd.read_csv(file, skiprows=[0], header=None)
            required_data = df.iloc[:, 9:]
            data_without_offset = required_data.sub(required_data.mean(axis=1), axis=0).values
            print(data_without_offset.shape)
            self.randNum = randrange(len(df))
            print(self.randNum)
            title = ('Original Plot', self.randNum)
            self.m.plot(data_without_offset, title, self.randNum)
            self.required_data_without_offset.append(data_without_offset)
            self.plot_echo();
#            self.files = []
            
    def plot_echo(self):
        for i, data in enumerate(self.required_data_without_offset):
            required_echos = self.get_echos(data)
#            print(np.array(required_echos).shape)
            title= ('Echo Plot', self.randNum)
            self.n.plot(required_echos, title, self.randNum)
            self.required_echo_list.append(required_echos)
#            self.required_data_without_offset = []
            
    def get_echos(self, filtered_values):
        NOISE_SIZE = 0
        ECHO_SIZE = 2048
        STARTING_POINT = 256
        
        all_echo_range = [] 
        for index, data in enumerate(filtered_values):
            chopped_data = data[NOISE_SIZE:]
            max_point_distance = self.peak_value(chopped_data)
            if max_point_distance:
                cutting_distance = max_point_distance - STARTING_POINT
                if cutting_distance > 0:
                    echo_range = chopped_data[cutting_distance:]
                    echo_range = echo_range[:ECHO_SIZE]
                    all_echo_range.append(echo_range)
        return all_echo_range
    
    def peak_value(self, data):
        THRESHOLD = 0.15
        max_point_distance = 0
        peakData = 0
        max_point_distance = np.array(data).argmax()
        peakData = np.array(data).max()
        if peakData > THRESHOLD:
            return max_point_distance
        else: 
            return None
        
#    def save_echo(self):
#        echo_set = []
#        for i, data in enumerate(self.required_echo_list):
#            echo_set = echo_set + data
#        data = pd.DataFrame(echo_set)
#        #filename = self.text_echo_filename.text()
#        data.to_csv(self.directory+'_overall.csv', header=False, index=False)
#        print(data.shape)
        
        
    def plot_fft(self):
        print("Plot fft button clicked")
        for i, data in enumerate(self.required_echo_list):
            fft_data_frame = DataFrame(data)
#            print(fft_data_frame.shape)
            self.required_fft_data_frame.append(fft_data_frame)
#            self.required_echo_list = []
        
        fs= 1.14e6
        row = fft_data_frame.values[self.randNum]
        fft_data = fft(row)/row.size
        freq = fftfreq(row.size, d=1/fs)
        
        print(row.shape)
        fft_data = np.abs(fft_data)
        
        cut_high_signal = (fft_data).copy()
        cut_high_signal[(freq > 50000)] = 0
        cut_high_signal[(freq < 30000)] = 0
        signal_without_0 = list(filter(lambda a: a != 0, cut_high_signal))
        
        print(np.array(signal_without_0).shape)
        self.signal_set.append(np.abs(signal_without_0))
        title = ('FFT Plot', self.randNum)
        self.o.fft_plot(freq, cut_high_signal, title)
        return self.signal_set
    
#    Todo: Save features
    def save_features(self):
        print('Save features button clicked')
        for i, data in enumerate(self.signal_set):
            
            data = pd.DataFrame(data)
            data.to_csv(self.directory+'_overall.csv', header=False, index=False)
            print(data.shape)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())