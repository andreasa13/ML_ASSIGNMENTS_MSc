from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


# load train and test dataset
# remove headers, footers and quotes to avoid overfitting
train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True)
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True)

X_train = train_data.data
y_train = train_data.target

X_test = test_data.data
y_test = test_data.target

# print(train_data.target_names)

# Convert text to vectors with TF-IDF vectorizer
tf_idf = TfidfVectorizer()
X_train = tf_idf.fit_transform(X_train)
X_test = tf_idf.transform(X_test)

# naive bayes classifier
clf = MultinomialNB(alpha=0.01).fit(X_train, y_train)
predicted = clf.predict(X_test)

# evaluation metrics
accuracy = metrics.accuracy_score(y_test, predicted)
recall = metrics.recall_score(y_test, predicted, average='macro')
precision = metrics.precision_score(y_test, predicted, average='macro')
F1 = metrics.f1_score(y_test, predicted, average='macro')

# print rounded metrics
print('Accuracy: %.2f' % accuracy)
print('Recall: %.2f' % recall)
print('Precision: %.2f' % precision)
print('F1: %.2f' % F1)

# confusion matrix
cm = confusion_matrix(y_test, predicted)

# heatmap
labels = train_data.target_names
plt.figure(figsize=(10, 10))
plt.title('Multinomial NB - Confusion Matrix (a = 0.01) ' + '[Prec: %.2f' % precision + ', Rec: %.2f' % recall + ', F1: %.2f' % F1 + ']')
sn.heatmap(cm, annot=True, cmap='Reds', fmt='g', xticklabels=labels, yticklabels=labels)
plt.show()

# parameter 'a' effect to F1 score - plot

# list_of_f1_scores = []
# for a in np.arange(0.01, 1.0, 0.01):
#     clf = MultinomialNB(alpha=a).fit(X_train, y_train)
#     predicted = clf.predict(X_test)
#     f1 = metrics.f1_score(y_test, predicted, average='macro')
#     list_of_f1_scores.append(f1)
#
# x = np.arange(0.01, 1.0, 0.01)
# plt.plot(x, list_of_f1_scores, linestyle='-', color='b')
# plt.show()
