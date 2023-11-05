#%%
from sklearn.preprocessing import StandardScaler, LabelBinarizer 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from toolbox_02450 import rlr_validate
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.


data = pd.read_csv('dropout_data.csv', delimiter=';')
# data.columns = [col.replace('\t', '') for col in data.columns]
data_no_target = data.drop(columns=['Target']) # delete target column
data_matrix = data_no_target.values  # data matrix
target = data['Target']
attributeNames = data_no_target.columns.tolist()
classNames = target.unique().tolist()
class_num = len(classNames)
class_name_mapping = {className: index for index, className in enumerate(classNames)}
target_to_num = [class_name_mapping[className] for className in target] # drop out = 0, graduate = 1, enrolled = 2
one_hot_encoded = LabelBinarizer().fit_transform(target_to_num)

# revise partial data according to appendix of reference
for col_index in [0,1,3,5,6,7,8,9,10,11,12]:
        current_column = data_matrix[:, col_index]
        unique_values = np.unique(current_column)
        unique_values.sort()
        value_to_rank = {value: rank +1 for rank, value in enumerate(unique_values)}
        data_matrix[:, col_index] = np.vectorize(value_to_rank.get)(current_column)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_matrix) # mean 0, standard deviation 1
#############
# apply PCA #
#############
U, S, Vt = np.linalg.svd(pd.DataFrame(normalized_data), full_matrices=False)
n_components = 25
pca_data = np.dot(U[:, :n_components], np.diag(S[:n_components]))
scaler = StandardScaler()
pca_data = scaler.fit_transform(pca_data) # mean 0, standard deviation 1

#%%
#Load data from matlab file
X = pca_data
y = np.array(target_to_num)
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
#lambdas = np.power(10.,range(-5,10))
lambdas = np.logspace(-8, 8, 100)
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, K)

# Display the results for the last cross-validation fold
figure(K, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

plt.savefig("generalization error.png", dpi=500)  # 如果要保存图
show()