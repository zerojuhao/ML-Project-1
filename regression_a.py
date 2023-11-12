#%%
from sklearn.preprocessing import StandardScaler, LabelBinarizer 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from toolbox_02450 import rlr_validate
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)

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
for col_index in [0,1,3,5,7,8,9,10,11]:
        current_column = data_matrix[:, col_index]
        unique_values = np.unique(current_column)
        unique_values.sort()
        value_to_rank = {value: rank +1 for rank, value in enumerate(unique_values)}
        data_matrix[:, col_index] = np.vectorize(value_to_rank.get)(current_column)

X = data_matrix
y = X[:, 6]
X = np.delete(X, 6, axis=1)
attributeNames.remove('Previous qualification (grade)')
attributeNames = [u'Offset']+attributeNames
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

colormap = plt.get_cmap('tab20', len(attributeNames) - 1)
markers = ['o', 's', 'D', 'v', '<', '>', 'p']
line_styles = ['-', '--', '-.', ':']

figure(figsize=(12,11))
for i in range(len(attributeNames) - 2):
    plt.semilogx(lambdas, mean_w_vs_lambda.T[:, i+1], color=colormap(i),
                 linestyle=line_styles[i % len(line_styles)],
                 marker=markers[i % len(markers)], label=attributeNames[i+1], markersize=5)

xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
legend(attributeNames[1:], loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0,0,1,1])
plt.savefig("coefficient_values_error.pdf", bbox_inches='tight')
show()

figure(figsize=(12,8))
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

plt.savefig("generalization_error.pdf", bbox_inches='tight')
show()
