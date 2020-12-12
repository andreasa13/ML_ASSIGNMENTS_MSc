# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, ensemble, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================



# Load breastCancer data
# =============================================================================

# ADD COMMAND TO LOAD DATA HERE
breastCancer = datasets.load_breast_cancer()

# =============================================================================



# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

# Split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# This proportion can be changed using the 'test_size' or 'train_size' parameter.
# Alsao, passing an (arbitrary) value to the parameter 'random_state' "freezes" the splitting procedure 
# so that each run of the script always produces the same results (highly recommended).
# Apart from the train_test_function, this parameter is present in many routines and should be
# used whenever possible.

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)


# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================

# ADD COMMAND TO CREATE RANDOM FOREST CLASSIFIER MODEL HERE
model = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42)

# =============================================================================


# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN YOUR MODEL HERE
model.fit(x_train, y_train)

# =============================================================================




# Ok, now let's predict the output for the test set
# =============================================================================

# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = model.predict(x_test)

# =============================================================================



# Time to measure scores. We will compare predicted output (from input of second subset, i.e. x_test)
# with the real output (output of second subset, i.e. y_test).
# You can call 'accuracy_score', 'recall_score', 'precision_score', 'f1_score' or any other available metric
# from the 'sklearn.metrics' library.
# The 'average' parameter is used while measuring metric scores to perform a type of averaging on the data.
# One of the following can be used for this example, but it is recommended that 'macro' is used (for now):
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
#             This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
# =============================================================================



# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print('Accuracy: %.2f' % accuracy_score(y_test, y_predicted))
print('Recall: %.2f' % recall_score(y_test, y_predicted, average='macro'))
print('Precision: %.2f' % precision_score(y_test, y_predicted, average='macro'))
print('F1: %.2f' % f1_score(y_test, y_predicted, average='macro'))

# =============================================================================



# A Random Forest has been trained now, but let's train more models, 
# with different number of estimators each, and plot performance in terms of
# the difference metrics. In other words, we need to make 'n'(e.g. 200) models,
# evaluate them on the aforementioned metrics, and plot 4 performance figures
# (one for each metric).
# In essence, the same pipeline as previously will be followed.
# =============================================================================

# After finishing the above plots, try doing the same thing on the train data
# Hint: you can plot on the same figure in order to add a second line.
# Change the line color to distinguish performance metrics on train/test data
# In the end, you should have 4 figures (one for each metric)
# And each figure should have 2 lines (one for train data and one for test data)



# CREATE MODELS AND PLOTS HERE

# create an empty DataFrame to save results for each value of 'n_estimators' parameter
df = pd.DataFrame(columns=['n_estimators', 'accuracy_train', 'accuracy_test', 'recall_train', 'recall_test',
                  'precision_train', 'precision_test', 'f1_train', 'f1_test'])

for n in range(1, 201):

    model = RandomForestClassifier(n_estimators=n, random_state=42)

    # train model
    model.fit(x_train, y_train)

    # predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # accuracy metric
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # recall metric
    recall_train = recall_score(y_train, y_train_pred, average='macro')
    recall_test = recall_score(y_test, y_test_pred, average='macro')

    # precision metric
    precision_train = precision_score(y_train, y_train_pred, average='macro')
    precision_test = precision_score(y_test, y_test_pred, average='macro')

    # F1 score
    f1_train = metrics.f1_score(y_train, y_train_pred, average='macro')
    f1_test = metrics.f1_score(y_test, y_test_pred, average='macro')

    # create a list for current model evaluation metrics
    data = [n, accuracy_train, accuracy_test, recall_train, recall_test, precision_train, precision_test, f1_train, f1_test]
    # append the list as a new row in the DataFrame
    df = df.append(pd.Series(data, index=df.columns), ignore_index=True)

#print(df)

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# ----------------- ACCURACY LEARNING CURVES ------------------
axs[0, 0].grid()

axs[0, 0].set_title('Accuracy Learning Curves')

# set y-axes limits
axs[0, 0].set_ylim((0.85, 1.01))

# set axes labels
axs[0, 0].set_xlabel("n_estimators")
axs[0, 0].set_ylabel("accuracy")

# draw learning curves
axs[0, 0].plot(df['n_estimators'], df['accuracy_train'], '-', color="b", label="Train")
axs[0, 0].plot(df['n_estimators'], df['accuracy_test'], '-', color="r", label="Test")

# set legend
axs[0, 0].legend(loc="best")

# ----------------- RECALL LEARNING CURVES ------------------
axs[0, 1].grid()

axs[0, 1].set_title('Recall Learning Curves')

# set y-axes limits
axs[0, 1].set_ylim((0.85, 1.01))

# set axes labels
axs[0, 1].set_xlabel("n_estimators")
axs[0, 1].set_ylabel("recall")

# draw learning curves
axs[0, 1].plot(df['n_estimators'], df['recall_train'], '-', color="b", label="Train")
axs[0, 1].plot(df['n_estimators'], df['recall_test'], '-', color="r", label="Test")

axs[0, 1].legend(loc="best")

# ----------------- PRECISION LEARNING CURVES ------------------
axs[1, 0].grid()

axs[1, 0].set_title('Precision Learning Curves')

# set y-axes limits
axs[1, 0].set_ylim((0.85, 1.01))

# set axes labels
axs[1, 0].set_xlabel("n_estimators")
axs[1, 0].set_ylabel("precision")

# draw learning curves
axs[1, 0].plot(df['n_estimators'], df['precision_train'], '-', color="b", label="Train")
axs[1, 0].plot(df['n_estimators'], df['precision_test'], '-', color="r", label="Test")

axs[1, 0].legend(loc="best")

# ----------------- F1 LEARNING CURVES ------------------
# plt.figure()
axs[1, 1].grid()
axs[1, 1].set_title('F1 Learning Curves')

# set y-axes limits
axs[1, 1].set_ylim((0.85, 1.01))

# set axes labels
axs[1, 1].set_xlabel("n_estimators")
axs[1, 1].set_ylabel("F1")

# draw curves
axs[1, 1].plot(df['n_estimators'], df['f1_train'], '-', color="b", label="Train")
axs[1, 1].plot(df['n_estimators'], df['f1_test'], '-', color="r", label="Test")

axs[1, 1].legend(loc="best")

fig.show()
fig.savefig('metrics_plots.png')
# =============================================================================