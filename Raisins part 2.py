# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:41:04 2022

@author: olive
"""

import numpy as np

import xlrd

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot, subplot, hist

import sklearn.linear_model as lm
from sklearn import model_selection, tree
doc = xlrd.open_workbook('Raisin_Dataset.xls').sheet_by_index(0)


attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=7)

classLabels = doc.col_values(7,1,901) 
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[value] for value in classLabels])


# Preallocate memory, then extract data to matrix X
X = np.empty((900,7))
for i in range(7):
    X[:,i] = np.array(doc.col_values(i,1,901)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

X = X - np.ones((N,1))*X.mean(0)
X = X*(1/np.std(X,0))

##Linear regression
area_idx = attributeNames.index('Area')
y = X[:,area_idx]

X_cols = list(range(0,area_idx)) + list(range(area_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict area content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Area content (true)'); ylabel('Area content (estimated)');
subplot(2,1,2)
hist(residual,40)

show()


model = lm.LogisticRegression()
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_besni_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new type of wine), as in exercise 5.1.7
x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31]).reshape(1,-1)
# Evaluate the probability of x being a white wine (class=0) 
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being a Besni: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_besni_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_besni_prob[class1_ids], '.r')
xlabel('Data object (raisins sample)'); ylabel('Predicted prob. of class Besni');
legend(['Besni', 'Kecimen'])


show()


#Cross validation


# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

test_proportion = 0.5
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

# Initialize variables
Error_train = np.empty((len(tc),1))
Error_test = np.empty((len(tc),1))

for i, t in enumerate(tc):
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
    dtc = dtc.fit(X_train,y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = np.asarray(dtc.predict(X_test),dtype=int)
    y_est_train = np.asarray(dtc.predict(X_train), dtype=int)
    misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
    misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
    Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train
    
f = figure()
plot(tc, Error_train*100)
plot(tc, Error_test*100)
xlabel('Model complexity (max tree depth)')
ylabel('Error (%)')
legend(['Error_train','Error_test'])
    
show()    

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1

    
f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()
