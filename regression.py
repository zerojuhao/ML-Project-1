#%%
def jls_extract_def(sklearn, preprocessing, numpy, matplotlib, pyplot, pandas, scipy, linalg, linear_model, model_selection, impute, metrics, toolbox_02450, pylab, io):
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
    return plt, pd, LabelBinarizer, np, StandardScaler, rlr_validate, figure, subplot, semilogx, xlabel, ylabel, grid, legend, title, loglog, show, Ridge, cross_val_score, train_test_split, model_selection, lm


plt, pd, LabelBinarizer, np, StandardScaler, rlr_validate, figure, subplot, semilogx, xlabel, ylabel, grid, legend, title, loglog, show, Ridge, cross_val_score, train_test_split, model_selection, lm = jls_extract_def(sklearn, preprocessing, numpy, matplotlib, pyplot, pandas, scipy, linalg, linear_model, model_selection, impute, metrics, toolbox_02450, pylab, io)


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

X = normalized_data

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

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda ,baseline_mse= rlr_validate(X, y, lambdas, K)


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


plt.savefig("generalization error.png", dpi=50)  # save image

show()

#%%
################################
# regression_a_test # use ridge#
################################

lambda_values = np.logspace(-8, 8, 100)


train_errors = []

validation_errors = []


for alpha in lambda_values:

    ridge = Ridge(alpha=alpha)
    

    train_scores = cross_val_score(ridge, X, y, cv=10, scoring='neg_mean_squared_error')

    train_errors.append(np.mean(-train_scores))
    

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    

    ridge.fit(X_train, y_train)

    val_pred = ridge.predict(X_val)

    validation_error = np.mean((val_pred - y_val) ** 2)

    validation_errors.append(validation_error)


plt.figure(figsize=(10, 6))

plt.plot(lambda_values, train_errors, label='Train Error', marker='o')

plt.plot(lambda_values, validation_errors, label='Validation Error', marker='o')

plt.xscale('log')

plt.xlabel('Lambda (log scale)')

plt.ylabel('Mean Squared Error')

plt.title('Ridge Regression - Generalization Error vs. Lambda')
plt.legend()
plt.grid()

plt.show()

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


    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, baseline_mse = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # baseline error calculation

    baseline_error=np.empty((K,1))

    baseline_error = np.append(baseline_error,baseline_mse)


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

    print(f'Out fold: {k}','\n' ,'optimal error: {0}'.format(opt_val_err), '\n','optimal λ: {0}'.format(np.log10(opt_lambda)), '\n','Test error without: {0}'.format(Error_test.mean()),'\n','Test error: {0}'.format(Error_test_rlr.mean()), '\n','Baseline error: {0}'.format(np.mean(baseline_error)))


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