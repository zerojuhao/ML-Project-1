from sklearn.preprocessing import StandardScaler, LabelBinarizer 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
#N, M = normalized_data.shape  # N = Number of data objects, M = Number of attributes
#Y = normalized_data - np.ones((N, 1)) * normalized_data.mean(0)  # Centering the data
#Y = Y * (1 / np.std(Y, 0))  # Standardize the data

#U, S, Vh = svd(Y, full_matrices=False)  # Performing SVD
#V = Vh.T  # The principal directions
#pca1_coefficients = V[:, 0]
#pca2_coefficients = V[:, 1]

# 使用SVD分解数据矩阵
U, S, Vt = np.linalg.svd(pd.DataFrame(normalized_data), full_matrices=False)

# 选择要保留的主成分的数量
n_components = 22

# 使用前n_components个奇异值和相应的左奇异向量来进行数据变换
reduced_data = np.dot(U[:, :n_components], np.diag(S[:n_components]))


# 定义一组不同的正则化参数 λ 的值
alphas = np.logspace(-20, 20, 100)

# 初始化一个列表来存储每个 λ 对应的交叉验证得分
cross_val_scores = []
# 划分训练集和测试集
#X_train, X_test, y_train, y_test = train_test_split(reduced_data, one_hot_encoded, test_size=0.2, random_state=42)



# 使用 K = 10 倍交叉验证来估计泛化误差
for alpha in alphas:
    # 创建 Ridge 回归模型，指定正则化参数
    model = Ridge(alpha=alpha)
    # 计算 K = 10 倍交叉验证的平均得分
    scores = cross_val_score(model, reduced_data, one_hot_encoded, cv=10)
    cross_val_scores.append(np.mean(scores))

# 绘制 λ 作为函数的估计泛化误差图
plt.plot(alphas, cross_val_scores)
plt.xscale('log')
plt.xlabel('λ (log scale)')
plt.ylabel('Cross-Validation Score')
plt.title('Estimated Generalization Error as a Function of λ')
plt.show()