import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, recall_score, accuracy_score, precision_score, f1_score, roc_curve, roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import KFold


def pca(x_train, x_test,  n):
    pca = PCA(n_components=n)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca


def minmax(x):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled


def get_metrics(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    return accuracy, recall, precision, f1, fpr, tpr, auc


htru = pd.read_csv('HTRU_2.csv')
htru.columns = ['Profile_mean', 'Profile_stdev', 'Profile_skewness', 'Profile_kurtosis',
                'DM_mean', 'DM_stdev', 'DM_skewness', 'DM_kurtosis', 'class']

# print(htru.head())

x = htru.drop(['class'], axis=1)
y = htru['class']

# train-test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=42)

rf = RandomForestClassifier(random_state=42)

rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

acc, rec, pre, f1, fpr1, tpr1, auc1 = get_metrics(y_test, y_pred)

print('Accuracy: %.2f' % acc)
print('Recall: %.2f' % rec)
print('Precision: %.2f' % pre)
print('F1: %.2f' % f1)
print("\n")


# Apply Principal Component Analysis (PCA)
x_train, x_test = pca(x_train, x_test, 4)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

acc, rec, pre, f1, fpr2, tpr2, auc2 = get_metrics(y_test, y_pred)

print('Accuracy with PCA: %.2f' % acc)
print('Recall with PCA: %.2f' % rec)
print('Precision with PCA: %.2f' % pre)
print('F1 with PCA: %.2f' % f1)
print("\n")


# plot Area Under Curve
plt.plot(fpr1, tpr1, linestyle='--', color='blue', label='No PCA: (AUC = %0.2f)' % auc1)
plt.plot(fpr2, tpr2, linestyle='--', color='orange', label='With PCA: (AUC = %0.2f)' % auc2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve Plot')
plt.legend(loc='lower right')
plt.show()
# =========================== FEATURE IMPORTANCE =================================
# correlation of every features with class column
fi_series = htru.drop("class", axis=1).apply(lambda x: x.corr(htru['class']))
fi_dict = fi_series.to_dict()
fi_dict = sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)
fi_dict = dict(fi_dict)
print("Features Correlation: ")
sorted_features_list = list(fi_dict.keys())
print(sorted_features_list)
print("\n")

model = RandomForestClassifier(random_state=42)
model.fit(x, y)

importances = model.feature_importances_
importances_dict = dict(zip(htru.columns, importances))
importances_dict = sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)
importances_dict = dict(importances_dict)
print("Random Forest Feature Importance: ")
sorted_features_list = list(importances_dict.keys())
print(sorted_features_list)
print("\n")

# =========================== FEATURE SELECTION =================================
k_best_features = sorted_features_list[0:4]

print('The 4 most important features: ', k_best_features)
print("\n")


k_best_features.append('class')

# k_worst_features = ['DM_kurtosis', 'Profile_stdev', 'DM_skewness', 'class']

htru_fs = htru[k_best_features]
# print(htru_fs)

x = htru_fs.drop(['class'], axis=1)
y = htru_fs['class']

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=42)

model_fs = RandomForestClassifier(random_state=42)

model_fs.fit(x_train, y_train)
y_pred = model_fs.predict(x_test)

acc, rec, pre, f1, fpr, tpr, auc = get_metrics(y_test, y_pred)

print('Accuracy after Feature Selection: %.2f' % acc)
print('Recall after Feature Selection: %.2f' % rec)
print('Precision after Feature Selection:: %.2f' % pre)
print('F1 after Feature Selection: %.2f' % f1)

plt.plot(fpr1, tpr1, linestyle='--', color='blue', label='No Feature Selection: (AUC = %0.2f)' % auc1)
plt.plot(fpr, tpr, linestyle='--', color='red', label='With Feature Selection: (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve Plot')
plt.legend(loc='lower right')
plt.show()


