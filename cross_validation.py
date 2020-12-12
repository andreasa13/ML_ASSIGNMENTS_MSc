from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import statistics as st

# load dataset
X, y = datasets.load_breast_cancer(return_X_y=True)

# define the classifier
clf = DecisionTreeClassifier(criterion='gini', random_state=42)

# k-fold equivalent to Leave-One-Out
kf = KFold(n_splits=10)
kf.get_n_splits(X)

list_of_acc = []
for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc_score = accuracy_score(y_pred, y_test)
    list_of_acc.append(acc_score)

print("List of Accuracy scores: ", list_of_acc)
print("Mean Accuracy score: %.3f" % st.mean(list_of_acc))

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# fit
clf.fit(X_train, y_train)
# predict
y_pred = clf.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)

tn = cm[0][0]
fn = cm[1][0]
tp = cm[1][1]
fp = cm[0][1]

print("True Positive = {}".format(tp))
print("True Negative = {}".format(tn))
print("False Positive = {}".format(fp))
print("False Negative = {}".format(fn))
