import sklearn.tree
import sklearn.linear_model
import numpy as np
from toolbox_02450 import *
from scipy.io import loadmat

zANN = np.array([0.42, 0.46, 0.39, 0.49, 0.49, 0.43, 0.47,0.47,0.47,0.49])
zLOG = np.array([0.099, 0.049, 0.037, 0.37, 0.062, 0.074, 0.062,0.074,0.074,0.062])
zBASE = np.array([0.38, 0.37, 0.37, 0.35, 0.36, 0.37, 0.39,0.36,0.38,0.39])
alpha = 0.05

z = zANN - zBASE
CI= st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

print(p)
print(CI)
z = zANN - zLOG
CI= st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

print(p)
print(CI)
z = zLOG - zBASE
CI= st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

print(p)
print(CI)