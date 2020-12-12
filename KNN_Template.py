# =============================================================================
# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email us: arislaza@csd.auth.gr, ipierros@csd.auth.gr
# =============================================================================

# import the KNeighborsClassifier
# if you want to do the hard task, also import the KNNImputer
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

random.seed = 42
np.random.seed(666)



# Import the titanic dataset
# Decide which features you want to use (some of them are useless, ie PassengerId).
#
# Feature 'Sex': because this is categorical instead of numerical, KNN can't deal with it, so drop it
# Note: another solution is to use one-hot-encoding, but it's out of the scope for this exercise.
#
# Feature 'Age': because this column contains missing values, KNN can't deal with it, so drop it
# If you want to do the harder task, don't drop it.
#
# =============================================================================
# load DataFrame
titanic = pd.read_csv('titanic.csv')
print(titanic.head())
# print(titanic.dtypes)
print('# of examples: ', len(titanic))

# categorical to binary(integer)
print(titanic['Sex'].unique())
titanic["Sex"].replace({"male": 1, "female": 0}, inplace=True)

print(titanic.dtypes)

# drop unnecessary features
cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = titanic.drop(cols, axis=1)

print(df.dtypes)

# replace infinity values with nan (if exist)
# df = df.replace([np.inf, -np.inf], np.nan)

# count nan values per feature
print(df.isna().sum())

# drop 2 rows that include NaN value in this feature
df_new = df.dropna(subset=['Embarked'])

# One Hot Encoding of categorical features
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df_new[['Embarked']]).toarray(), columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'])
df_new = df_new.join(enc_df)
df_new = df_new.drop(['Embarked'], axis=1)

# df without imputer (just drop 'Age' column)
# df_2 = df_new.dropna()
df_2 = df_new.drop(['Age'], axis=1)
print('you are here')
# print(df_2.isna().sum())
df_2 = df_2.dropna(subset=['Embarked_C', 'Embarked_Q', 'Embarked_S'])

X = df_new.drop(['Survived'], axis=1)
y = df_new['Survived'].values

X_2 = df_2.drop(['Survived'], axis=1)
y_2 = df_2['Survived'].values

# Normalize feature values using MinMaxScaler
# Fit the scaler using only the train data
# Transform both train and test data.
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.25)

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# Do the following only if you want to do the hard task.
#
# Perform imputation for completing the missing data for the feature
# 'Age' using KNNImputer. 
# As always, fit on train, transform on train and test.
#
# Note: KNNImputer also has a n_neighbors parameter. Use n_neighbors=3.
# =============================================================================

imputer = KNNImputer(n_neighbors=3, weights='uniform', metric='nan_euclidean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# min-max scaling
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_2 = scaler.fit_transform(X_train_2)
X_test_2 = scaler.transform(X_test_2)

# Create your KNeighborsClassifier models for different combinations of parameters
# Train the model and predict. Save the performance metrics.
# =============================================================================

knn = KNeighborsClassifier(n_neighbors=7, weights='uniform', p=2)
knn.fit(X_train, y_train)

y_predicted = knn.predict(X_test)

print('Accuracy: %.2f' % metrics.accuracy_score(y_test, y_predicted))
print('Recall: %.2f' % metrics.recall_score(y_test, y_predicted))
print('Precision: %.2f' % metrics.precision_score(y_test, y_predicted))
print('F1: %.2f' % metrics.f1_score(y_test, y_predicted))
print()

# Plot the F1 performance results for any combination οf parameter values of your choice.
# If you want to do the hard task, also plot the F1 results with/without imputation (in the same figure)
# =============================================================================

f1_impute = []
max_f1 = 0
optimal_k = 0
for k in range(1, 200):
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=2)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_predicted)
    if f1_score > max_f1:
        max_f1 = f1_score
        optimal_k = k
    # print(f1_score)
    f1_impute.append(f1_score)

print('Max F1: %.2f' % max_f1)
print('Optimal k: ', optimal_k)


f1_no_impute = []
for k in range(1, 200):
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=2)
    knn.fit(X_train_2, y_train_2)
    y_predicted_2 = knn.predict(X_test_2)
    f1_score = metrics.f1_score(y_test_2, y_predicted_2)
    # print(f1_score)
    f1_no_impute.append(f1_score)

plt.title('k-Nearest Neighbors (Weights = uniform, Metric = F1, p = 2)')
plt.plot(f1_impute, label='with impute')
plt.plot(f1_no_impute, label='without impute')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1')
plt.show()

