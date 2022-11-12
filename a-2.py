
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
import xlrd
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

#doc = xlrd.open_workbook('../Raisins/Raisin_Dataset.xls').sheet_by_index(0)
doc = xlrd.open_workbook(r"C:\Users\Johannes\iCloudDrive\Desktop\Machine Learning\Raisin_Dataset.xls").sheet_by_index(0)

attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=7)

classLabels = doc.col_values(7,1,901) 


# Preallocate memory, then extract data to matrix X
X = np.empty((900,7))
for i in range(7):
    X[:,i] = np.array(doc.col_values(i,1,901)).T

# Transforming the data
for i in range(7):
    X[:,i] = X[:,i] - np.mean(X[:,i])
    X[:,i] = X[:,i]*(1/np.std(X[:,i],0))

idx = attributeNames.index('Area')
y = X[:,idx]
X = np.delete(X,idx,1)
N, M = X.shape

lambdas = np.power(10.,range(-10,5))
K = 10   
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, K)

print(np.log10(opt_lambda))

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
show()