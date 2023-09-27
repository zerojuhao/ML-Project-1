import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dropout_data.csv', sep=";")

#One-out-of-K ransformation
df['Dropout'] = df['Target'] == 'Dropout'
df['Graduate'] = df['Target'] == 'Graduate'
df['Enrolled'] = df['Target'] == 'Enrolled'
#
df['Dropout'] = df['Dropout'].apply(lambda x: int(x))
df['Graduate'] = df['Graduate'].apply(lambda x: int(x))
df['Enrolled'] = df['Enrolled'].apply(lambda x: int(x))
#
df = df.drop('Target', axis=1)

# Countplot for the target variable
plt.figure(figsize=(8, 6))
#sns.countplot(x='Target', data=df)
plt.title('Target Variable Distribution')
plt.show()


# Correlation heatmap for numerical features without annotations
corr_matrix = df.corr()

# Define a smaller figure size
plt.figure(figsize=(9, 8))

# Create the heatmap with the desired color range and smaller font size
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar_kws={"shrink": 0.8}, square=True, cbar=True, vmin=-1, vmax=1, annot_kws={"size": 6})

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







