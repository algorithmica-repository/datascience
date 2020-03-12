##chi-square goodness of fit test
#it tests whether the distribution of sample categorical data matches an 
#expected distribution
import numpy as np
from scipy import stats
import pandas as pd
import os

n_trails = 120
n_outcomes = 6
result = np.random.randint(1, n_outcomes+1, n_trails)

outcomes, observed = np.unique(result, return_counts=True)
for (o,f) in zip(outcomes, observed):
    print(o,f)
expected = np.array(n_outcomes * [n_trails/n_outcomes], dtype=np.int64)

stats.chisquare(f_obs = observed, f_exp = expected)   

observed = [15,29,18,19,20,19]
observed = [20,20,20,20,20,20]
observed = [30,10,20,20,20,20]
observed = [10,30,30,10, 10,30]

#chi-square independence test
#The chi-squared test of independence tests whether two categorical variables 
#are independent
dir = 'F:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())

observed = pd.crosstab(titanic_train.Sex, titanic_train.Survived)
stats.chi2_contingency(observed = observed)   

observed = pd.crosstab(titanic_train.Pclass, titanic_train.Survived)
stats.chi2_contingency(observed = observed)   
