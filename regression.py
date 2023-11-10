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
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary, rlr_validate_mse, rlr_validate_nmo
from scipy import stats
import statsmodels.stats.contingency_tables as tbl


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

################
# regression_a #
################

#%%
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10

lambdas = np.logspace(-8, 8, 100)
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, K)

# Display the results for the last cross-validation fold
figure(K, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
legend(attributeNames[1:], loc='best')

subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

plt.savefig("generalization error.png", dpi=50)  # save image, and set dpi
show()

###############
# regression_b#
###############

#%%
N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = 37
K = 10
CV = model_selection.KFold(K, shuffle=True)
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
model_ann_performance = []
model_linear_regression_performance = []
model_baseline_performance = []
k=0
for train_index, test_index in CV.split(X,y):
    print('\n ***Outer Crossvalidation fold: {0}/{1}'.format(k+1,K))
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    lambdas = np.logspace(-8, 8, 100)

    # receive output
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, baseline_mse, best_units_num, min_error_ann = rlr_validate_mse(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    # Compute mean squared error without regularization
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    print('Optimal Hidden units: {0}'.format(np.mean(best_units_num)), '\n',
        'ANN error: {0}'.format(np.mean(min_error_ann)), '\n',
        'Optimal λ: {0}'.format(np.log10(opt_lambda)), '\n',
        'Linear Regression error: {0}'.format(opt_val_err), '\n',
        'Baseline error: {0}'.format(np.mean(baseline_mse)), '\n',
        'Test error without: {0}'.format(Error_test.mean()),'\n',
        'Test error: {0}'.format(Error_test_rlr.mean()), '\n',        
        )
    model_ann_performance.append(min_error_ann)
    model_linear_regression_performance.append(opt_val_err)
    model_baseline_performance.append(baseline_mse)
    k+=1

ann_performance = np.array(model_ann_performance)
linear_regression_performance = np.array(model_linear_regression_performance)
baseline_performance = np.array(model_baseline_performance)

###########################
# regression_b_evaluation #
###########################
#%%
# get t and p value
t_stat_ann_vs_lr, p_value_ann_vs_lr = stats.ttest_ind(ann_performance, linear_regression_performance)
t_stat_ann_vs_baseline, p_value_ann_vs_baseline = stats.ttest_ind(ann_performance, baseline_performance)
t_stat_lr_vs_baseline, p_value_lr_vs_baseline = stats.ttest_ind(linear_regression_performance, baseline_performance)

if p_value_ann_vs_lr < 0.05:
    print("Significant differences in performance between ANN and Linear Regression")
else:
    print("No significant difference in performance between ANN and Linear Regression")
print(f"t-statistic ANN vs LR: {t_stat_ann_vs_lr}, p-value: {p_value_ann_vs_lr}")

if p_value_ann_vs_baseline < 0.05:
    print("Significant differences in performance between ANN and Baseline")
else:
    print("No significant difference in performance between ANN and Baseline")
print(f"t-statistic ANN vs Baseline: {t_stat_ann_vs_baseline}, p-value: {p_value_ann_vs_baseline}")

if p_value_lr_vs_baseline < 0.05:
    print("Significant differences in performance between Linear Regression and Baseline")
else:
    print("No significant difference in performance between Linear Regression and Baseline")
print(f"t-statistic LR vs Baseline: {t_stat_lr_vs_baseline}, p-value: {p_value_lr_vs_baseline}")

models = ['ANN', 'Linear Regression', 'Baseline']
p_values = [p_value_ann_vs_lr, p_value_ann_vs_baseline, p_value_lr_vs_baseline]
plt.bar(models, p_values, color=['blue', 'orange', 'red'])
plt.xlabel('Models')
plt.ylabel('p-value')
plt.title('p-value for Model Comparisons')
plt.show()

# calculate confidence interval
def confidence_interval(data, alpha=0.05):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    n = len(data)
    z = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = z * (std_dev / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return (lower_bound, upper_bound)

ann_ci = confidence_interval(ann_performance)
linear_regression_ci = confidence_interval(linear_regression_performance)
baseline_ci = confidence_interval(baseline_performance)

print("ANN Confidence Interval:\n", ann_ci)
print("Linear Regression Confidence Interval:\n", linear_regression_ci)
print("Baseline Confidence Interval:\n", baseline_ci)

models = ['ANN', 'Linear Regression', 'Baseline']
mean_performance = [np.mean(ann_performance), np.mean(linear_regression_performance), np.mean(baseline_performance)]
conf_intervals = [ann_ci, linear_regression_ci, baseline_ci]
performance_data = [ann_performance, linear_regression_performance, baseline_performance]
x_pos = np.arange(len(models))
bar_width = 0.3

plt.figure(figsize=(10, 6))

for i, model in enumerate(models):
    lower_bound, upper_bound = conf_intervals[i]
    y = mean_performance[i]
    plt.bar(x_pos[i], y, bar_width)
    plt.errorbar(x_pos[i], y, yerr=[[y - lower_bound], [upper_bound - y]], fmt='o',color = 'black', capsize=5)

plt.xticks(x_pos, models)
plt.xlabel('Models')
plt.ylabel('Performance')
plt.title('Confidence Intervals for Different Models')

plt.show()

##################
# classification #
##################

#%%
N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = 37
K = 10
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
model_ann_performance = []
model_linear_regression_performance = []
model_baseline_performance = []
k=0
for train_index, test_index in CV.split(X,y):
    print('\n ***Outer Crossvalidation Fold: {0}/{1}'.format(k+1,K))
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    lambdas = np.logspace(-8, 8, 100)
    
    # receive output
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, baseline_nmo, best_units_num, min_error_ann = rlr_validate_nmo(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    # Compute mean squared error without regularization
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    print(
        'Optimal Hidden units: {0}'.format(np.mean(best_units_num)), '\n',
        'ANN error: {0}'.format(np.mean(min_error_ann)), '\n',
        'Optimal λ: {0}'.format(np.log10(opt_lambda)), '\n',
        'Logistic Regression error: {0}'.format(opt_val_err), '\n',
        'Baseline error: {0}'.format(np.mean(baseline_nmo)), '\n',
        #'Test error without: {0}'.format(Error_test.mean()),'\n',
        #'Test error: {0}'.format(Error_test_rlr.mean()), '\n',        
        )
    model_ann_performance.append(min_error_ann)
    model_linear_regression_performance.append(opt_val_err)
    model_baseline_performance.append(baseline_nmo)
    k+=1

ann_performance = np.array(model_ann_performance)
linear_regression_performance = np.array(model_linear_regression_performance)
baseline_performance = np.array(model_baseline_performance)
#############################
# classification_evaluation #
#############################
#%%
# get t and p value

# if p_value_ann_vs_lr < 0.05:
#     print("Significant differences in performance between ANN and Linear Regression")
# else:
#     print("No significant difference in performance between ANN and Linear Regression")
# print(f"t-statistic ANN vs LR: {t_stat_ann_vs_lr}, p-value: {p_value_ann_vs_lr}")

# if p_value_ann_vs_baseline < 0.05:
#     print("Significant differences in performance between ANN and Baseline")
# else:
#     print("No significant difference in performance between ANN and Baseline")
# print(f"t-statistic ANN vs Baseline: {t_stat_ann_vs_baseline}, p-value: {p_value_ann_vs_baseline}")

# if p_value_lr_vs_baseline < 0.05:
#     print("Significant differences in performance between Linear Regression and Baseline")
# else:
#     print("No significant difference in performance between Linear Regression and Baseline")
# print(f"t-statistic LR vs Baseline: {t_stat_lr_vs_baseline}, p-value: {p_value_lr_vs_baseline}")

# models = ['ANN', 'Linear Regression', 'Baseline']
# p_values = [p_value_ann_vs_lr, p_value_ann_vs_baseline, p_value_lr_vs_baseline]
# plt.bar(models, p_values, color=['blue', 'orange', 'red'])
# plt.xlabel('Models')
# plt.ylabel('p-value')
# plt.title('p-value for Model Comparisons')
# plt.show()

# # calculate confidence interval
# def confidence_interval(data, alpha=0.05):
#     mean = np.mean(data)
#     std_dev = np.std(data, ddof=1)
#     n = len(data)
#     z = stats.t.ppf(1 - alpha / 2, n - 1)
#     margin_of_error = z * (std_dev / np.sqrt(n))
#     lower_bound = mean - margin_of_error
#     upper_bound = mean + margin_of_error
#     return (lower_bound, upper_bound)

# ann_ci = confidence_interval(ann_performance)
# linear_regression_ci = confidence_interval(linear_regression_performance)
# baseline_ci = confidence_interval(baseline_performance)

# print("ANN Confidence Interval:\n", ann_ci)
# print("Linear Regression Confidence Interval:\n", linear_regression_ci)
# print("Baseline Confidence Interval:\n", baseline_ci)

# models = ['ANN', 'Linear Regression', 'Baseline']
# mean_performance = [np.mean(ann_performance), np.mean(linear_regression_performance), np.mean(baseline_performance)]
# conf_intervals = [ann_ci, linear_regression_ci, baseline_ci]
# performance_data = [ann_performance, linear_regression_performance, baseline_performance]
# x_pos = np.arange(len(models))
# bar_width = 0.3

# plt.figure(figsize=(10, 6))

# for i, model in enumerate(models):
#     lower_bound, upper_bound = conf_intervals[i]
#     y = mean_performance[i]
#     plt.bar(x_pos[i], y, bar_width)
#     plt.errorbar(x_pos[i], y, yerr=[[y - lower_bound], [upper_bound - y]], fmt='o',color = 'black', capsize=5)

# plt.xticks(x_pos, models)
# plt.xlabel('Models')
# plt.ylabel('Performance')
# plt.title('Confidence Intervals for Different Models')
# plt.legend()
# plt.show()