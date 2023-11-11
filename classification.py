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
import torch
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary, rlr_validate_mse, rlr_validate_nmo, mcnemar, ttest_twomodels
from scipy import stats
import statsmodels.stats.contingency_tables as tbl
from collections import Counter


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False  # display “ - ”


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
X = normalized_data # if use pca_data, remember change some parameters in ANN #
y = np.array(target_to_num)

##################
# classification #
##################

#%%
N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = 37
K = 5
CV = model_selection.KFold(K, shuffle=True)

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

thetahat_al_c = []
CI_al_c = []
p_al_c = []

thetahat_ab_c = []
CI_ab_c = []
p_ab_c = []

thetahat_lb_c = []
CI_lb_c = []
p_lb_c = []

for train_index, test_index in CV.split(X,y):
    print('\n ~~~Outer Crossvalidation Fold: {0}/{1}'.format(k+1,K))
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 5
    lambdas = np.logspace(-8, 8, 100)
    
    # receive output
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, baseline_nmo, best_units_num, min_error_ann = rlr_validate_nmo(X_train, y_train, lambdas, internal_cross_validation)
    
    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
    X_train_ann = torch.Tensor(X_train)
    y_train_ann = torch.Tensor(y_train)
    X_test_ann = torch.Tensor(X_test)
    y_test_ann = torch.Tensor(y_test)
    y_train_ann = y_train_ann.view(-1).long()
    y_test_ann = y_test_ann.view(-1).long()
    
    # ANN
    model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(37, best_units_num), #M features to H hiden units
                                torch.nn.LeakyReLU(),                            #torch.nn.ReLU(),torch.nn.Tanh()
                                torch.nn.Linear(best_units_num, 3), # H hidden units to 1 output neuron
                                torch.nn.Softmax(dim=1) # final tranfer function
                                )
    loss_fn = torch.nn.CrossEntropyLoss()
    max_iter = 500 #200 700 50 500

    net, final_loss, learning_curve = train_neural_net(model,
                                                    loss_fn,
                                                    X=X_train_ann,
                                                    y=y_train_ann,
                                                    n_replicates=1, # 3 1 10 3
                                                    max_iter=max_iter)
    
    # Determine estimated class labels for test set
    y_softmax = net(X_test_ann) # activation of final note, i.e. prediction of network            y_test_ann = y_test_ann.type(dtype=torch.uint8)
    y_test_est_ann = (torch.max(y_softmax, dim=1)[1])  # select the label with max possibility
    
    # Logistic Regression
    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda, max_iter=1000)
    mdl.fit(X_train, y_train)
    y_test_est = mdl.predict(X_test).T

    
    # Baseline
    element_counts = Counter(y_train)
    most_common_element, most_common_count = element_counts.most_common(1)[0] # find the most common element
    y_test_est_baseline = np.full(len(y_test), most_common_element)
    
    # calculate mcnemar of each inner fold
    thetahat_al, CI_al, p_al = mcnemar(y_test, y_test_est, y_test_est_ann.data.numpy(), alpha=0.1)
    thetahat_ab, CI_ab, p_ab = mcnemar(y_test, y_test_est_ann.data.numpy(), y_test_est_baseline, alpha=0.1)
    thetahat_lb, CI_lb, p_lb = mcnemar(y_test, y_test_est, y_test_est_baseline,alpha=0.1)

    thetahat_al_c.append(thetahat_al)
    CI_al_c.append(CI_al)
    p_al_c.append(p_al)

    thetahat_ab_c.append(thetahat_ab)
    CI_ab_c.append(CI_ab)
    p_ab_c.append(p_ab)

    thetahat_lb_c.append(thetahat_lb)
    CI_lb_c.append(CI_lb)
    p_lb_c.append(p_lb)
    
    # evaluation
    print('\n',
        'Optimal Hidden units: {0}'.format(np.mean(best_units_num)), '\n',
        'ANN error: {0}'.format(np.mean(min_error_ann)), '\n',
        'Optimal λ: {0}'.format(np.log10(opt_lambda)), '\n',
        'Logistic Regression error: {0}'.format(opt_val_err), '\n',
        'Baseline error: {0}'.format(np.mean(baseline_nmo)), '\n',
        #'Test error without: {0}'.format(Error_test.mean()),'\n',
        #'Test error: {0}'.format(Error_test_rlr.mean()), '\n',        
        )
    k+=1

#############################
# classification_evaluation #
#############################
#%%
# show ANN vs Logistic Regression
lower_bounds_al = [row[0] for row in CI_al_c]
upper_bounds_al = [row[1] for row in CI_al_c]
column_means_al = [sum(row) / len(row) for row in CI_al_c]
center_line1_al = np.mean(np.concatenate([lower_bounds_al,upper_bounds_al]))
x = np.arange(1,K+1)
plt.figure(figsize=(8, 3))
for x, start, end in zip(x, lower_bounds_al, upper_bounds_al):
    plt.bar(x, height=end-start, bottom=start, width = 0.2)
    
plt.scatter(np.arange(1, k+1), np.array(column_means_al), color='black', marker='o', label='Mean Value')
plt.axhline(center_line1_al, color='red', linestyle='--', label='mean')
plt.xlabel('K-fold')
plt.ylabel('Confidence interval')
plt.title('ANN vs Logistic Regression')
plt.show()
plt.figure(figsize=(8, 3))
plt.plot(np.arange(1, k+1), p_al_c, marker='o', linestyle='-', color = 'red')
plt.xlabel('K-fold')
plt.ylabel('p Value')
plt.title('ANN vs Logistic Regression')
plt.show()

# show ANN vs Baseline
lower_bounds_ab = [row[0] for row in CI_ab_c]
upper_bounds_ab = [row[1] for row in CI_ab_c]
column_means_ab = [sum(row) / len(row) for row in CI_ab_c]
center_line1_ab = np.mean(np.concatenate([lower_bounds_ab,upper_bounds_ab]))
x = np.arange(1,K+1)
plt.figure(figsize=(8, 3))
for x, start, end in zip(x, lower_bounds_ab, upper_bounds_ab):
    plt.bar(x, height=end-start, bottom=start, width = 0.2)
    
plt.scatter(np.arange(1, k+1), np.array(column_means_ab), color='black', marker='o', label='Mean Value')
plt.axhline(center_line1_ab, color='red', linestyle='--', label='mean')
plt.xlabel('K-fold')
plt.ylabel('Confidence interval')
plt.title('ANN vs Baseline')
plt.show()
plt.figure(figsize=(8, 3))
plt.plot(np.arange(1, k+1), p_ab_c, marker='o', linestyle='-', color = 'red')
plt.xlabel('K-fold')
plt.ylabel('p Value')
plt.title('ANN vs Baseline')
plt.show()

# show Logistic Regression vs Baseline
lower_bounds_lb = [row[0] for row in CI_ab_c]
upper_bounds_lb = [row[1] for row in CI_ab_c]
column_means_lb = [sum(row) / len(row) for row in CI_ab_c]
center_line1_lb = np.mean(np.concatenate([lower_bounds_lb,upper_bounds_lb]))
x = np.arange(1,K+1)
plt.figure(figsize=(8, 3))
for x, start, end in zip(x, lower_bounds_lb, upper_bounds_lb):
    plt.bar(x, height=end-start, bottom=start, width = 0.2)
    
plt.scatter(np.arange(1, k+1), np.array(column_means_lb), color='black', marker='o', label='Mean Value')
plt.axhline(center_line1_lb, color='red', linestyle='--', label='mean')
plt.xlabel('K-fold')
plt.ylabel('Confidence interval')
plt.title('Logistic Regression vs Baseline')
plt.show()
plt.figure(figsize=(8, 3))
plt.plot(np.arange(1, k+1), p_lb_c, marker='o', linestyle='-', color = 'red')
plt.xlabel('K-fold')
plt.ylabel('p Value')
plt.title('Logistic Regression vs Baseline')
plt.show()