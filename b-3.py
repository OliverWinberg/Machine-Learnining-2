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
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
import numpy as np, scipy.stats as st


alpha = 0.05
zANN = np.array([0.000056, 0.00017, 0.000027, 0.00024, 0.000068,0.00015, 0.00019, 0.00023, 0.00016, 0.000048]) * 810
zLin = np.array([0.0037, 0.0020, 0.0016, 0.0017, 0.0018, 0.0025, 0.0066, 0.0021, 0.0029, 0.0036]) * 810
zBase = np.array([0.52, 0.54, 0.52, 0.51, 0.58,0.53, 0.56, 0.52, 0.56, 0.51]) * 810

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis

print(zANN)
zANNBASE = zANN - zBase
zANNLIN = zANN - zLin
zLINBASE = zLin - zBase

CI1 = st.t.interval(1-alpha, len(zANNBASE)-1, loc=np.mean(zANNBASE), scale=st.sem(zANNBASE))  # Confidence interval
p1 = 2*st.t.cdf( -np.abs( np.mean(zANNBASE) )/st.sem(zANNBASE), df=len(zANNBASE)-1)  # p-value

CI2 = st.t.interval(1-alpha, len(zANNLIN)-1, loc=np.mean(zANNLIN), scale=st.sem(zANNLIN))  # Confidence interval
p2 = 2*st.t.cdf( -np.abs( np.mean(zANNLIN) )/st.sem(zANNLIN), df=len(zANNLIN)-1)  # p-value

CI3 = st.t.interval(1-alpha, len(zLINBASE)-1, loc=np.mean(zLINBASE), scale=st.sem(zLINBASE))  # Confidence interval
p3 = 2*st.t.cdf( -np.abs( np.mean(zLINBASE) )/st.sem(zLINBASE), df=len(zLINBASE)-1)  # p-value

print(CI1)
print(CI2)
print(CI3)

print(p1)
print(p2)
print(p3)