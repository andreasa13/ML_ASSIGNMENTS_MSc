# =============================================================================
# HOMEWORK 3 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# For this project, the only thing that we will need to import is the "Orange" library.
# However, before importing it, you must first install the library into Python.
# Read the instructions on how to do that (it might be a bit trickier than usual!)
# =============================================================================


# IMPORT LIBRARY HERE (trivial but necessary...)
import Orange
from Orange.classification import CN2Learner, CN2UnorderedLearner
from Orange.data import Table
from Orange.evaluation import testing
from Orange.classification.rules import LaplaceAccuracyEvaluator, EntropyEvaluator
import pandas as pd
# =============================================================================


# Load 'wine' dataset
# =============================================================================


# ADD COMMAND TO LOAD TRAIN AND TEST DATA HERE
wineData = Table('wine.csv')
print("# of examples: ", len(wineData))

# =============================================================================


# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
# =============================================================================


# ADD COMMAND TO DEFINE LEARNER HERE
learner = CN2Learner()
#learner = CN2UnorderedLearner()


# =============================================================================


# At this step we shall configure the parameters of our learner.
# We can set the evaluator/heuristic ('Entropy', 'Laplace' or 'WRAcc'),
# 'beam_width' (in the range of 3-10), 'min_covered_examples' (start from 7-8 and make your way up),
# and 'max_rule_length' (usual values are in the range of 2-5).
# They are located deep inside the 'learner', within the 'rule_finder' class.
# Note: for the evaluator, set it using one of the Evaluator classes in classification.rules
# =============================================================================

# an exhaustive implementation to find optimal parameters based on F1 score

# learner.rule_finder.quality_evaluator = EntropyEvaluator()
# df = pd.DataFrame(columns=['beam', 'rl', 'mr', 'f1'])
# for b in range(3, 11):
#     for rl in range(7, 52, 5):
#         for mr in range(2, 6):

#             learner.rule_finder.search_algorithm.beam_width = b
#             learner.rule_finder.general_validator.min_covered_examples = rl
#             learner.rule_finder.general_validator.max_rule_length = mr
#
#             cv = Orange.evaluation.CrossValidation()
#             results = cv(wineData, [learner])
#
#             f1 = Orange.evaluation.scoring.F1(results, average='macro')[0]
#             data = [b, rl, mr, f1]
#
#             df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
#
# print(df)
# df.to_excel('gridsearch.xlsx', index=False)

# # ADD COMMANDS TO CONFIGURE THE LEARNER HERE

learner.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()
#learner.rule_finder.quality_evaluator = EntropyEvaluator()
learner.rule_finder.search_algorithm.beam_width = 4
learner.rule_finder.general_validator.min_covered_examples = 42
learner.rule_finder.general_validator.max_rule_length = 2

# =============================================================================


# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets,
# then train the model, and produce results.
# So, simply initialize the CrossValidation() object from the 'testing' library
# and call it with input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.

cv = Orange.evaluation.CrossValidation()
results = cv(wineData, [learner])


# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
# =============================================================================


# # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("Accuracy: {:.3f}".format(Orange.evaluation.scoring.CA(results)[0]))
print("Precision: {:.3f}".format(Orange.evaluation.scoring.Precision(results, average='macro')[0]))
print("Recall: {:.3f}".format(Orange.evaluation.scoring.Recall(results, average='macro')[0]))
print("F1: {:.3f}".format(Orange.evaluation.scoring.F1(results, average='macro')[0]))

# =============================================================================


# Ok, now let's train our learner manually to see how it can classify our data
# using rules.You just want to feed it some data- nothing else.
# =============================================================================

# ADD COMMAND TO TRAIN THE LEARNER HERE
classifier = learner(wineData)

# =============================================================================


# Now we can print the derived rules. To do that, we need to iterate through
# the 'rule_list' of our classifier.
for rule in classifier.rule_list:
    print(rule)
