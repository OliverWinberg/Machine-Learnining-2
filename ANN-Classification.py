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
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary



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

n_hidden_units = 2 # number of hidden units in the signle hidden layer
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
loss_fn = torch.nn.BCELoss()
max_iter = 10000

    
    
# K-fold crossvalidation
K = 10                 # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

lambda_interval = np.logspace(-8, 2, 10)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))


for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    
    CV2 = model_selection.KFold(K, shuffle=True)
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index]).unsqueeze(1)
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
 
    for (k2, (train_index2, test_index2)) in enumerate(CV2.split(X_train,y_train)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train2 = torch.Tensor(X[train_index2,:])
        y_train2 = torch.Tensor(y[train_index2])
        X_test2 = torch.Tensor(X[test_index2,:])
        y_test2 = torch.Tensor(y[test_index2])
        
        
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=3,
                                                           max_iter=max_iter)
        
    

        print('\n\tBest loss: {}\n'.format(final_loss))
         
        # Determine estimated class labels for test set
        y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
        y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        y_test = y_test.type(dtype=torch.uint8)
        # Determine errors and error rate
        e = (y_test_est != y_test)
        error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    

