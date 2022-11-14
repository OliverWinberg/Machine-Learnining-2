from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
import xlrd
import sklearn.linear_model as lm
from toolbox_02450 import rlr_validate
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
from sklearn import model_selection
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot


#Load data
doc = xlrd.open_workbook('../Raisins/Raisin_Dataset.xls').sheet_by_index(0)
# doc = xlrd.open_workbook(r"C:\Users\Johannes\iCloudDrive\Desktop\Machine Learning\Raisin_Dataset.xls").sheet_by_index(0)

attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=8)
classLabels = doc.col_values(7,1,901)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

#Extract data to matrix X
X = np.empty((900,7))
for i in range(7):
    X[:,i] = np.array(doc.col_values(i,1,901)).T  
    
X = stats.zscore(X)

#Extract class values
y = np.array([classDict[value] for value in classLabels])

#Extract dimensions of data
N = len(y)
M = len(attributeNames)-1
C = len(classNames)

# K-fold crossvalidation
K = 10                 # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

for train_index, test_index in CV.split(X,y):
    errors = []
    CV2 = model_selection.KFold(K, shuffle=True)
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index] 
    for train_index2, test_index2 in CV2.split(X_train,y_train):
        # extract training and test set for current CV fold
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]
        if np.mean(y_train2) > 0.5:
            y_est = np.ones(y_test2.shape)
        else:
            y_est = np.zeros(y_test2.shape)
        e = 0    
        for i in range(0, len(y_test2)):
            if y_est[i] != y_test2[i]:
                e += 1
                
        error_rate = e/len(y_test2)
        errors.append(error_rate)
        
    print(min(errors))
        