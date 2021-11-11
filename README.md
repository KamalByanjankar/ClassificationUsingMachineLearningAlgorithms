# ClassificationUsingMachineLearningAlgorithms

# Requirements
The packages required are:
  •	Numpy
  •	pandas
  •	matplotlib
  •	glob
  •	sklearn
  •	PyQt5

# Data Analysis
After loading the data, the offset voltage is removed from each data frame, i.e. substracting the mean of each data frame with the data itself. It is done to received the plot 
that aligns with x-axis. After removing the offset voltage, the echo or reflected signal is taken into consideration in which the highest peak is determined along with its index 
from which the echo signal is calculated and it is saved in a file which is further used to calculate features. It is done by applying FFT with the frequency range of 30KHz to 
50KHz, since the operating frequency of an ultrasonic sensor is 40KHz.

# Applying ML
After extracting the features, we train the ML with the features and then we classify the objects. Random Forest classifier, KNN classifier and CNN is used to classify the 
objects.
