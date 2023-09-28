import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import get_sta_info # calculate mean median min max of basic statistics
import plot_sta # plot basic statistics table
import one_hot # apply one-out-of-k to traget
import seaborn as sns


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
df = np.array(csv.iloc[:,:-1])
rows, cols = df.shape

# apply one hot coding
y_one_hot = one_hot.one_out_of_k_encoding(y)

###############################
# plot basic statistics table #
###############################

# get basic statistics
df_basic_info = get_sta_info.basic_info(df)
rounded_df_basic_info = np.round(df_basic_info,3) # set number like 0.000
basic_sta_info = pd.DataFrame(rounded_df_basic_info.T, index = attributeNames, columns = ['Mean','Median','Dispersion','Min','Max'])
basic_sta_info = basic_sta_info.applymap(lambda x: f'{x:.3f}'.rstrip('0').rstrip('.') if isinstance(x, (float, np.float64)) else x)

# divide data into groups
demographic_data = basic_sta_info.iloc[[0,7,13,17,19,20]]
socioeconomics_data = basic_sta_info.iloc[[9,8,11,10,14,15,16,18]]
macroeconomics_data = basic_sta_info.iloc[[33,34,35]]
academic_data_enrollment = basic_sta_info.iloc[[1,2,3,4,5]]
academic_data_1st = basic_sta_info.iloc[[21,22,23,24,25,26]]
academic_data_2st = basic_sta_info.iloc[[27,28,29,30,31,32]]
academic_data_target = pd.DataFrame(get_sta_info.basic_info(y).T ,index = ["Target"], columns = ['Mean','Median','Dispersion','Min','Max'])
academic_data_target.iloc[0,2] = "Graduate"
academic_data_target = academic_data_target.drop(columns = "Mean")

# plot tables
plot_sta.plot_table(demographic_data)
# plot_sta.plot_table(socioeconomics_data)
# plot_sta.plot_table(macroeconomics_data)
# plot_sta.plot_table(academic_data_enrollment)
# plot_sta.plot_table(academic_data_1st)
# plot_sta.plot_table(academic_data_2st)
# plot_sta.plot_table(academic_data_target)


#############
# apply PCA #
#############

N, M = df.shape  # N = Number of data objects, M = Number of attributes
# Subtract mean value from data
Y = df - np.ones((N,1))*df.mean(0)  # Centering the data
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

##########################
# Plot variance explained#
##########################
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()

#######################################
# Data transformed onto PCA1 and PCA2 #
#######################################

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

###########################
# plot Correlation Heatmap#
###########################
csv_file_path = 'dropout_data.csv'
df = pd.read_csv('dropout_data.csv', sep=";")
df = df.drop('Target', axis=1)
df = df.drop('Previous qualification (grade)', axis=1)

new_order = [#demographic_data#
             'Marital status', 'Nacionality', 'Displaced','Gender','Age at enrollment','International',
             #socioeconomics_data#
             "Mother's qualification","Father's qualification","Mother's occupation","Father's occupation",
             'Educational special needs','Debtor','Tuition fees up to date','Scholarship holder',
             #macroeconomics_data#
             'Unemployment rate','Inflation rate','GDP',
             #academic_data_enrollment#
             'Application mode','Application order','Course','Daytime/evening attendance','Previous qualification',
             #academic_data_1st#
             'Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)',
             'Curricular units 1st sem (approved)','Curricular units 1st sem (grade)','Curricular units 1st sem (without evaluations)',
             #academic_data_2st#
             'Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)',
             'Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)','Curricular units 2nd sem (without evaluations)'
             ]
sorted_df = df[new_order]



for field in sorted_df:
    sorted_df[field] = sorted_df[field].astype("category").cat.codes

# Correlation heatmap for numerical features without annotations
corr_matrix = sorted_df.corr()

# Define a smaller figure size
plt.figure(figsize=(9, 8))

# Create the heatmap with the desired color range and smaller font size
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm',linewidths=0.1,linecolor='white', cbar_kws={"shrink": 0.8}, square=True, cbar=True, vmin=-1, vmax=1, annot_kws={"size": 6})

# Create custom labels and position them
plt.xticks(
    ticks=range(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    rotation=45,  # Rotate x-axis labels if needed
    ha='right',  # Horizontal alignment
    fontsize=6  # Font size for x-axis labels
)

plt.yticks(
    ticks=range(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    rotation=0,  # Rotate y-axis labels if needed
    va='center',  # Vertical alignment
    fontsize=6  # Font size for y-axis labels
)

plt.title('Correlation Heatmap')
plt.tight_layout()  # Ensure everything fits within the frame
plt.show()

###########################
# plot Correlation table  #
###########################

# set threshold
threshold = 0.7

# create an empty dataframe to store  Feature\Collinearity with\Pearson
high_corr_df = pd.DataFrame(columns=['Feature', 'Collinearity with', 'Correlation'])

# find the correlation > 0.7
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        if (abs(corr_matrix.iloc[i, j]) > threshold) and (i!=j) :
            label1 = corr_matrix.index[i]
            label2 = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            # Check if Feature are repeated
            duplicate = False
            for _, row in high_corr_df.iterrows():
                if (row['Feature'] == label1 and row['Collinearity with'] == label2) or (row['Feature'] == label2 and row['Collinearity with'] == label1):
                    duplicate = True
                    break
            
            # If the Feature are not repeated, they will be stored in the dataframe.
            if not duplicate:   
                high_corr_df = pd.concat([high_corr_df, pd.DataFrame({'Feature': [label1], 'Collinearity with': [label2], 'Correlation': [correlation]})], ignore_index=True)

high_corr_df = np.round(high_corr_df,4) # set number like 0.0000

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=high_corr_df.values, colLabels=high_corr_df.columns, cellLoc='center', loc='center', colColours=['#f3f3f3']*high_corr_df.shape[1])
plt.show()

###########################
# plot distribution image #
###########################

# Separate numerical and categorical columns
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = df.select_dtypes(include=['object']).columns.tolist()

# Create a 2-column figure for the overlapping ridge plots with reduced vertical gap
plt.figure(figsize=(12, 6))

# Create overlapping ridge plots for numerical features in the left column
for i, feature in enumerate(num_features[:len(num_features)//2]):
    ax = plt.subplot(len(num_features)//2, 2, i * 2 + 1)
    sns.histplot(df[feature], color='blue', element='step', bins=30, kde=False)
    plt.xlabel('')
    plt.ylabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(df[feature].min(), df[feature].max())  # Set x-axis limits

    # Add feature names on the left side
    plt.text(1.02, 0.5, f'{feature}', rotation=0, fontsize=8, ha='left', va='center', transform=ax.transAxes)

# Create overlapping ridge plots for numerical features in the right column
for i, feature in enumerate(num_features[len(num_features)//2:]):
    ax = plt.subplot(len(num_features)//2, 2, i * 2 + 2)
    sns.histplot(df[feature], color='blue', element='step', bins=30, kde=False)
    plt.xlabel('')
    plt.ylabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim(df[feature].min(), df[feature].max())  # Set x-axis limits

    # Add feature names on the left side
    plt.text(1.02, 0.5, f'{feature}', rotation=0, fontsize=8, ha='left', va='center', transform=ax.transAxes)

# Reduce the vertical gap between histograms
plt.tight_layout(h_pad=0)
plt.show()


