##chi-square goodness of fit test
#it tests whether the distribution of sample categorical data matches an 
#expected distribution
import numpy as np
from scipy import stats
import pandas as pd
import os

dir = 'F:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())

#one sample t-test
#A one-sample t-test checks whether a sample mean differs from the population mean.
fare_sample = titanic_train[['Fare']].sample(frac=0.6)
stats.ttest_1samp(a = fare_sample, popmean = titanic_train[['Fare']].mean()) 

#two sample t-test
#A two-sample t-test investigates whether the means of two independent data samples 
#differ from one another.
fare_by_non_survived = titanic_train.Fare[titanic_train.Survived==0]
fare_by_survived = titanic_train.Fare[titanic_train.Survived==1]
stats.ttest_ind(a = fare_by_non_survived,
                b = fare_by_survived,
                equal_var=False)

#paired t-test
#testing differences between samples of the same group at different points in time.
#a hospital might want to test whether a weight-loss drug works 
#by checking the weights of the same group patients before and after treatment. 
#A paired t-test lets you check whether the means of samples from the same group differ.
before= stats.norm.rvs(scale=30, loc=250, size=100)
after = before + stats.norm.rvs(scale=5, loc=-1.25, size=100)
weight_df = pd.DataFrame({"weight_before":before,
                          "weight_after":after,
                          "weight_change":after-before})
weight_df.describe() 
stats.ttest_rel(a = before, b = after)

#anova test
#The one-way ANOVA tests whether the mean of some numeric variable differs 
#across the levels of one categorical variable(do any of the group means differ from one another?)
fare_by_class1 = titanic_train.Fare[titanic_train.Pclass==1]
fare_by_class2 = titanic_train.Fare[titanic_train.Pclass==2]
fare_by_class3 = titanic_train.Fare[titanic_train.Pclass==3]

stats.f_oneway(fare_by_class1, fare_by_class2, fare_by_class3)
