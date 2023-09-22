import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import get_sta_info
from pandas.plotting import table
import plot_sta
import one_hot

# read csv file
csv = pd.read_csv('dropout_data.csv', delimiter=';')
target = csv['Target']
attributeNames = list(csv.columns.values)
attributeNames = attributeNames[:-1]
classLabels = csv.values[:,-1]
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(3)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])
X = np.array(csv.iloc[:,:-1])
rows, cols = X.shape

# apply one hot coding
y_one_hot = one_hot.one_out_of_k_encoding(y)

# get basic statistics
X_basic_info = get_sta_info.basic_info(X)
rounded_X_basic_info = np.round(X_basic_info,3) # set number like 0.000
basic_sta_info = pd.DataFrame(rounded_X_basic_info.T, index = attributeNames, columns = ['Mean','Median','Dispersion','Min','Max'])
basic_sta_info = basic_sta_info.applymap(lambda x: f'{x:.3f}'.rstrip('0').rstrip('.') if isinstance(x, (float, np.float64)) else x)

# divide data into groups
demographic_data = basic_sta_info.iloc[[0,7,13,19,20]]
socioeconomics_data = basic_sta_info.iloc[[9,8,11,10,14,15,16,18]]
macroeconomics_data = basic_sta_info.iloc[[33,34,35]]
academic_data_enrollment = basic_sta_info.iloc[[1,2,3,4,5]]
academic_data_1st = basic_sta_info.iloc[[21,22,23,24,25,26]]
academic_data_2st = basic_sta_info.iloc[[27,28,29,30,31,32]]
academic_data_target = pd.DataFrame(get_sta_info.basic_info(y).T ,index = ["Target"], columns = ['Mean','Median','Dispersion','Min','Max'])
academic_data_target.iloc[0,2] = "Graduate"
academic_data_target = academic_data_target.drop(columns = "Mean")

# plot tables
plot_sta.plot_table(academic_data_target)




N, M = X.shape  # N = Number of data objects, M = Number of attributes

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)  # Centering the data
Y = Y * (1 / np.std(Y, 0))  # Standardize the data

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
V = Vh.T  # The principal directions
pca1_coefficients = V[:, 0]
pca2_coefficients = V[:, 1]

# Transforming the data to the principal component space
Z = Y @ V

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9
# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()


def get_colors(color_count: int, preferred_colors=None):
    if preferred_colors is None or len(preferred_colors) < color_count:
        num_random_colors = color_count - len(preferred_colors) if preferred_colors else color_count
        random_colors = [np.random.rand(3) for _ in range(num_random_colors)]
        if preferred_colors:
            return preferred_colors + random_colors
        else:
            return random_colors
    else:
        return preferred_colors


plt.figure()
for label, color in zip(classNames, get_colors(len(classNames),
                                               preferred_colors=['red', 'green',
                                                                 'blue'])):
    plt.scatter(Z[target == label, 0],
                Z[target == label, 1],
                label=label, color=color, alpha=0.5)
# plt.title('Data transformed onto PCA1 and PCA2')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.grid(True)

# 

fig, ax = plt.subplots(2, 1, figsize=(16, 12))
ax[0].bar(attributeNames, pca1_coefficients, color='blue', alpha=0.7)
ax[0].set_title('PCA1 Coefficients')
ax[0].set_ylabel('Coefficient')
ax[0].tick_params(axis='x', rotation=45)  # axis-x rotate 45
for label in ax[0].get_xticklabels():
    label.set_ha('right')

ax[1].bar(attributeNames, pca2_coefficients, color='red', alpha=0.7)
ax[1].set_title('PCA2 Coefficients')
ax[1].set_ylabel('Coefficient')
ax[1].tick_params(axis='x', rotation=45)
for label in ax[1].get_xticklabels():
    label.set_ha('right')

plt.tight_layout()
plt.show()