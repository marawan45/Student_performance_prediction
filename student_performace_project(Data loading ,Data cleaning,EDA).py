import os
from joblib import dump
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt   
import seaborn as sns             
import plotly.express as px       
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df=pd.read_csv("C://Users//maraw//Downloads//my work space//student_performance.csv")

print(df.head())

print(df.info())

print(df.describe())    

print(df.isnull().sum())

print(df.duplicated().sum())

print(df.nunique())

df.drop_duplicates(inplace=True)

# outlier detection and removal using IQR method (before splitting)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before_count = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    after_count = df.shape[0]
    print(f"Removed {before_count - after_count} outliers from column '{col}' using IQR.")

# only 150,000 rows 100,000 for training, 50,000 for testing
df = df.sample(n=150000, random_state=42) if len(df) > 150000 else df
df = df.reset_index(drop=True)
train_df = df.iloc[:100000]
test_df = df.iloc[100000:150000]

train_df2=train_df.copy()
test_df2=test_df.copy()

# Data Cleaning and Preprocessing
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')


train_df.loc[:, num_cols] = num_imputer.fit_transform(train_df[num_cols])
test_df.loc[:, num_cols] = num_imputer.transform(test_df[num_cols])

train_df.loc[:, cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
test_df.loc[:, cat_cols] = cat_imputer.transform(test_df[cat_cols])



# encode categorical variables
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df.loc[:, col] = le.fit_transform(train_df[col])
    test_df.loc[:, col] = le.transform(test_df[col])
    encoders[col] = le



train_df.loc[:, num_cols] = train_df[num_cols].astype(float)
test_df.loc[:, num_cols] = test_df[num_cols].astype(float)

# for the regression task
train_df2=train_df.copy()
test_df2=test_df.copy()

# Feature scaling  for the classification(only features)
save_dir = "prepared_data"
scaler = StandardScaler()
# For classification
feature_cols = ['weekly_self_study_hours', 'attendance_percentage', 'class_participation']
train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])
dump(scaler, os.path.join(save_dir, "standard_scaler.pkl"))
print("Saved StandardScaler to prepared_data/standard_scaler.pkl")



# Feature scaling  for the regression(only features)
feature_cols1 = ['weekly_self_study_hours']
scaler_reg = StandardScaler()
train_df2.loc[:, feature_cols1] = scaler_reg.fit_transform(train_df2[feature_cols1])
test_df2.loc[:, feature_cols1] = scaler_reg.transform(test_df2[feature_cols1])
dump(scaler_reg, os.path.join(save_dir, "standard_scaler_reg.pkl"))
print("Saved regression StandardScaler to prepared_data/standard_scaler_reg.pkl")


#plot correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Classification task: Predict grade (A–F) using correct feature columns
print("Columns in train_df just before feature selection:", train_df.columns.tolist())
feature_cols = ['weekly_self_study_hours', 'attendance_percentage', 'class_participation']
target_column = 'grade'
X_train = train_df[feature_cols]
y_train = train_df[target_column]
X_test = test_df[feature_cols]
y_test = test_df[target_column]


print("Prepared X_train shape:", X_train.shape)
print("Prepared X_test shape:", X_test.shape)
print("Grade value counts (train):\n", y_train.value_counts())

# --- EDA Visualizations for Classification Task ---
plt.figure(figsize=(7,4))
# Bar plot: Grade distribution
y_train.value_counts().plot(kind='bar', color='skyblue')
plt.title('Grade Distribution (Bar Plot)')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()

# Pie chart: Grade distribution
plt.figure(figsize=(6,6))
y_train.value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Grade Distribution (Pie Chart)')
plt.ylabel('')
plt.show()

# Line plot: Average feature values by grade
avg_features_by_grade = X_train.groupby(y_train).mean()
plt.figure(figsize=(8,5))
for col in feature_cols:
    plt.plot(avg_features_by_grade.index, avg_features_by_grade[col], marker='o', label=col)
plt.title('Average Feature Values by Grade (Line Plot)')
plt.xlabel('Grade')
plt.ylabel('Average Value')
plt.legend()
plt.show()





# Grade distribution
plt.figure(figsize=(7,4))
sns.countplot(x=y_train, order=sorted(y_train.unique()))
plt.title('Grade Distribution (Train)')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()

#Boxplots for features by grade
for col in feature_cols:
    plt.figure(figsize=(7,4))
    sns.boxplot(x=y_train, y=X_train[col])
    plt.title(f'{col} by Grade')
    plt.xlabel('Grade')
    plt.ylabel(col)
    plt.show()

#Pairplot for feature relationships colored by grade
eda_df = X_train.copy()
eda_df['grade'] = y_train.values
sns.pairplot(eda_df, hue='grade', diag_kind='kde', plot_kws={'alpha':0.5})
plt.suptitle('Feature Relationships by Grade', y=1.02)
plt.show()

save_dir = "prepared_data"
os.makedirs(save_dir, exist_ok=True)
X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)
print(f"Saved X_train, y_train, X_test, y_test to folder: {save_dir}")


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# linear regression task: Predict total score (0-100)

feature_cols1 = ['weekly_self_study_hours']
target_column1 = 'total_score'
X_train_reg = train_df2[feature_cols1]
y_train_reg = train_df2[target_column1]
X_test_reg = test_df2[feature_cols1]
y_test_reg = test_df2[target_column1]


# --- EDA Visualizations for Regression Task ---
plt.figure(figsize=(7,4))
# Bar plot: Binned total score distribution
score_bins = pd.cut(y_train_reg, bins=10)
score_bin_counts = score_bins.value_counts().sort_index()
score_bin_counts.plot(kind='bar', color='salmon')
plt.title('Total Score Distribution (Bar Plot)')
plt.xlabel('Total Score (Binned)')
plt.ylabel('Count')
plt.show()

#Correlation of Features to Total Score
print("Correlation of Each Feature to Total Score")
correlations = train_df2.corr(numeric_only=True)['total_score'].sort_values(ascending=False)
print(correlations)



# Line plot: Weekly self-study hours vs total score (mean)
mean_score_by_study_hours = pd.DataFrame({'weekly_self_study_hours': X_train_reg['weekly_self_study_hours'], 'total_score': y_train_reg}).groupby('weekly_self_study_hours').mean()
plt.figure(figsize=(8,5))
plt.plot(mean_score_by_study_hours.index, mean_score_by_study_hours['total_score'], marker='o', color='green')
plt.title('Weekly Self Study Hours vs Mean Total Score (Line Plot)')
plt.xlabel('Weekly Self Study Hours')
plt.ylabel('Mean Total Score')
plt.show()


# save regression train/test splits
X_train_reg.to_csv(os.path.join(save_dir, "X_train_reg.csv"), index=False)
y_train_reg.to_csv(os.path.join(save_dir, "y_train_reg.csv"), index=False)
X_test_reg.to_csv(os.path.join(save_dir, "X_test_reg.csv"), index=False)
y_test_reg.to_csv(os.path.join(save_dir, "y_test_reg.csv"), index=False)
print(f"Saved X_train_reg, y_train_reg, X_test_reg, y_test_reg to folder: {save_dir}")


print(f"Prepared X_train_reg shape:", X_train_reg.shape)
print(f"Prepared X_test_reg shape:", X_test_reg.shape)