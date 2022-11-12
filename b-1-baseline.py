# exercise 8.1.1

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
from sklearn.dummy import DummyRegressor
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

# doc = xlrd.open_workbook('Raisin_Dataset.xls').sheet_by_index(0)
doc = xlrd.open_workbook(r"C:\Users\Johannes\iCloudDrive\Desktop\Machine Learning\Raisin_Dataset.xls").sheet_by_index(0)

attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=8)

classLabels = doc.col_values(7,1,901) 


# Preallocate memory, then extract data to matrix X
X = np.empty((900,1))
X[:,0] = np.array(doc.col_values(0,1,901)).T

# Normalize data
# X = stats.zscore(X)


idx = attributeNames.index('Area')
y = X[:,idx]
attributeNames = np.delete(attributeNames,idx)
N, M = X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
# lambdas = np.power(10.,np.arange(0,10,0.01))
lambdas = np.power(10.,range(-10,5))


# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    errors = []
    CV2 = model_selection.KFold(K, shuffle=True)
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10    
    for train_index2, test_index2 in CV2.split(X_train,y_train):
        # extract training and test set for current CV fold
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]
        X_test2 = stats.zscore(X_test2)
        X_train2 = stats.zscore(X_train2)
        y_train2 = stats.zscore(y_train2)
        y_test2 = stats.zscore(y_test2)
        ymean = np.mean(y_train2)
        dummy_regr = DummyRegressor(strategy="mean")
        dummy_regr.fit(X_train2, y_train2)
        yhat = dummy_regr.predict(X_test2)
        sum = 0
        for i in range(yhat.size):
            sum += (y_train2[i]-yhat[i])**2
        error = sum/yhat.size
        errors.append(error)


    print(min(errors))

