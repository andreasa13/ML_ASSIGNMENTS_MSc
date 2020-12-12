from scipy.stats import friedmanchisquare
import pandas as pd


data = pd.read_csv('algo_performance.csv')
# print(data)

sample1 = data['C4.5'].tolist()
sample2 = data['1-NN'].tolist()
sample3 = data['NaiveBayes'].tolist()
sample4 = data['Kernel'].tolist()
sample5 = data['CN2'].tolist()

stat, p = friedmanchisquare(sample1, sample2, sample3, sample4, sample5)
print(p)

# alpha = 0.01
alpha = 0.05
# alpha = 0.1

if p > alpha:
    print('H0 is True')
else:
    print('H0 is False')
