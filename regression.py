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
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary, rlr_validate_mse
from scipy import stats


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

plt.savefig("generalization error.png", dpi=50)  # save image, and set dpi
show()
#%%
################################
# regression_a_test # use ridge#
################################
# lambda_values = np.logspace(-8, 8, 100)

# train_errors = []
# validation_errors = []

# for alpha in lambda_values:
#     ridge = Ridge(alpha=alpha)
    
#     train_scores = cross_val_score(ridge, X, y, cv=10, scoring='neg_mean_squared_error')
#     train_errors.append(np.mean(-train_scores))
    
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     ridge.fit(X_train, y_train)
#     val_pred = ridge.predict(X_val)
#     validation_error = np.mean((val_pred - y_val) ** 2)
#     validation_errors.append(validation_error)

# plt.figure(figsize=(10, 6))
# plt.plot(lambda_values, train_errors, label='Train Error', marker='o')
# plt.plot(lambda_values, validation_errors, label='Validation Error', marker='o')
# plt.xscale('log')
# plt.xlabel('Lambda (log scale)')
# plt.ylabel('Mean Squared Error')
# plt.title('Ridge Regression - Generalization Error vs. Lambda')
# plt.legend()
# plt.grid()
# plt.show()

################
# regression_b #
################

#%%
N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.logspace(-8, 8, 100)

# Initialize variables
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
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    min_errors_ann = []  # 用于存储每次循环的最小错误率
    best_h_ann = []  # 用于存储每次循环的对应的k值
    errors_ann = [] 
    
    # ANN #
    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensors
    H = np.arange(3,13,1)

    for h in H:
        X_train_ann = torch.Tensor(X[train_index,:] )
        y_train_ann = torch.Tensor(y[train_index] )
        X_test_ann = torch.Tensor(X[test_index,:] )
        y_test_ann = torch.Tensor(y[test_index] )
        y_train_ann = y_train_ann.view(-1, 1)
        y_test_ann = y_test_ann.view(-1, 1)

        # Define the model structure
        n_hidden_units = h*2+3 # number of hidden units in the signle hidden layer
        # The lambda-syntax defines an anonymous function, which is used here to 
        # make it easy to make new networks within each cross validation fold
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(37, n_hidden_units), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.Sigmoid(),                            #torch.nn.ReLU(),torch.nn.Tanh()
                            torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                            torch.nn.Sigmoid() # final tranfer function
                            )
        # Since we're training a neural network for binary classification, we use a 
        # binary cross entropy loss (see the help(train_neural_net) for more on
        # the loss_fn input to the function)
        loss_fn = torch.nn.BCELoss()
        # Train for a maximum of 10000 steps, or until convergence (see help for the 
        # function train_neural_net() for more on the tolerance/convergence))
        max_iter = 10000
        
        #####print('Training model of type:\n{}\n'.format(str(model())))
        # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
        # and see how the network is trained (search for 'def train_neural_net',
        # which is the place the function below is defined)
        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train_ann,
                                                        y=y_train_ann,
                                                        n_replicates=3,
                                                        max_iter=max_iter)
        
        # print('\n\tBest loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set
        y_sigmoid = net(X_test_ann) # activation of final note, i.e. prediction of network
        y_test_est_ann = (y_sigmoid > 0.5).type(dtype=torch.uint8) # threshold output of sigmoidal function
        y_test_ann = y_test_ann.type(dtype=torch.uint8)
        # Determine errors and error rate
        # e = (y_test_est_ann != y_test_ann)
        error_rate_ann = np.square(y_test_ann.numpy() - y_test_est_ann.numpy()).sum(axis=0)/len(y_test_ann)
        # error_rate_ann = (sum(e).type(torch.float)/len(y_test_ann)).data.numpy()
        errors_ann.append(error_rate_ann) # store error rate for current CV fold 
    min_error_ann = min(errors_ann)
    min_error_index = errors_ann.index(min_error_ann)
    best_h_ann = H[min_error_index]
        

    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # baseline error calculation
    # baseline_error=np.empty((10,1))
    # baseline_error = np.append(baseline_error,baseline_mse)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    #Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    #Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    print(f'Out fold: {k}','\n' ,'optimal error: {0}'.format(opt_val_err), '\n',
        'optimal λ: {0}'.format(np.log10(opt_lambda)), '\n',
        'Test error without: {0}'.format(Error_test.mean()),'\n',
        'Test error: {0}'.format(Error_test_rlr.mean()), '\n',
        )

    #print('Linear regression without feature selection:')
    #print('- Test error:     {0}'.format(Error_test.mean()))
    #print('Regularized linear regression:')
    #print('- Test error:     {0}'.format(Error_test_rlr.mean()))

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
#print('Linear regression without feature selection:')
# print('- Training error: {0}'.format(Error_train.mean()))
#print('- Test error:     {0}'.format(Error_test.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
#print('Regularized linear regression:')
# print('- Training error: {0}'.format(Error_train_rlr.mean()))
#print('- Test error:     {0}'.format(Error_test_rlr.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')

#############################
# regression_b_finished_here#
#############################

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
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
        

    # ANN #
    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensors
    # Values of lambda
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
#%%

t_stat_ann_vs_lr, p_value_ann_vs_lr = stats.ttest_ind(ann_performance, linear_regression_performance)
t_stat_ann_vs_baseline, p_value_ann_vs_baseline = stats.ttest_ind(ann_performance, baseline_performance)
t_stat_lr_vs_baseline, p_value_lr_vs_baseline = stats.ttest_ind(linear_regression_performance, baseline_performance)

if p_value_ann_vs_lr < 0.05:  # 选择显著性水平（通常为0.05）
    print("Significant differences in performance between ANN and Linear Regression")
else:
    print("No significant difference in performance between ANN and Linear Regression")

if p_value_ann_vs_baseline < 0.05:  # 选择显著性水平（通常为0.05）
    print("Significant differences in performance between ANN and Baseline")
else:
    print("No significant difference in performance between ANN and Baseline")

if p_value_lr_vs_baseline < 0.05:  # 选择显著性水平（通常为0.05）
    print("Significant differences in performance between Linear Regression and Baseline")
else:
    print("No significant difference in performance between Linear Regression and Baseline")

print(f"t-statistic ANN vs LR: {t_stat_ann_vs_lr}, p-value: {p_value_ann_vs_lr}")
print(f"t-statistic ANN vs Baseline: {t_stat_ann_vs_baseline}, p-value: {p_value_ann_vs_baseline}")
print(f"t-statistic LR vs Baseline: {t_stat_lr_vs_baseline}, p-value: {p_value_lr_vs_baseline}")
models = ['ANN', 'Linear Regression', 'Baseline']
p_values = [p_value_ann_vs_lr, p_value_ann_vs_baseline, p_value_lr_vs_baseline]
plt.bar(models, p_values, color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('p-value')
plt.title('p-value for Model Comparisons')
plt.show()

# 函数来计算置信区间
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

print("ANN Confidence Interval:", ann_ci)
print("Linear Regression Confidence Interval:", linear_regression_ci)
print("Baseline Confidence Interval:", baseline_ci)


models = ['ANN', 'Linear Regression', 'Baseline']
mean_performance = [np.mean(ann_performance), np.mean(linear_regression_performance), np.mean(baseline_performance)]
conf_intervals = [ann_ci, linear_regression_ci, baseline_ci]
performance_data = [ann_performance, linear_regression_performance, baseline_performance]

cis = [confidence_interval(data) for data in performance_data]

lower_bounds = [ci[0] for ci in cis]
upper_bounds = [ci[1] for ci in cis]

plt.bar(models, [np.mean(data) for data in performance_data], yerr=[(upper - lower) / 2 for upper, lower in zip(upper_bounds, lower_bounds)], capsize=5)
plt.ylabel("Mean Performance")
plt.title("Model Performance with Confidence Intervals")
plt.show()