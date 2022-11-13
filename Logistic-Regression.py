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

lambda_interval = np.logspace(-8, 2, 10)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))


for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    
    CV2 = model_selection.KFold(K, shuffle=True)
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test  = X[test_index,:]
    y_test  = y[test_index]
    
    min_errors = np.zeros(K)
    opt_lambdas = np.zeros(K)
    
    for (k2, (train_index2, test_index2)) in enumerate(CV2.split(X_train,y_train)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]
        
        
        for l in range(0, lambda_interval.size):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[l] )
            mdl.fit(X_train2, y_train2)
            
        
    
            y_train_est = mdl.predict(X_train2).T
            y_test_est = mdl.predict(X_test2).T
            
            train_error_rate[l] = np.sum(y_train_est != y_train2) / len(y_train2)
            test_error_rate[l] = np.sum(y_test_est != y_test2) / len(y_test2)
        
            w_est = mdl.coef_[0] 
            coefficient_norm[l] = np.sqrt(np.sum(w_est**2))
            
        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        min_errors[k2]=min_error
        opt_lambdas[k2]=opt_lambda
    print("min error:" + str(np.min(min_errors)))
    print("optimal lamda:" + str(opt_lambdas[np.argmin(min_errors)]))
    

# plt.figure(figsize=(8,8))
# #plt.plot(np.log10(lambda_interval), train_error_rate*100)
# #plt.plot(np.log10(lambda_interval), test_error_rate*100)
# #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
# plt.semilogx(lambda_interval, train_error_rate*100)
# plt.semilogx(lambda_interval, test_error_rate*100)
# plt.semilogx(opt_lambda, min_error*100, 'o')
# plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.ylabel('Error rate (%)')
# plt.title('Classification error')
# plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
# plt.ylim([0, 15])
# plt.grid()
# plt.show()    

# plt.figure(figsize=(8,8))
# plt.semilogx(lambda_interval, coefficient_norm,'k')
# plt.ylabel('L2 Norm')
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.title('Parameter vector L2 norm')
# plt.grid()
# plt.show()    
