import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# read csv file
csv = pd.read_csv('data.csv')

# extract attributeNames
attributeNames = list(csv.columns.values)
attributeNames = attributeNames[:-1]

# turn classnames to Number, Dropout = 0, Enrolled = 1, Graduate = 2
classLabels = csv.values[:,-1]
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(3)))
y = np.asarray([classDict[value] for value in classLabels])

# extract excel data to matrix X
X = np.array(csv.iloc[:,:-1])

# extract 80% data uesd for training
rows, cols = X.shape

#  apply one-out-of-K / one-hot
K = y.max()+1
onehot = np.zeros((len(X[:,0]),K))
onehot[np.arange(y.size),y] = 1

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.95

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
