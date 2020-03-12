##chi-square goodness of fit test
#it tests whether the distribution of sample categorical data matches an 
#expected distribution
from scipy import stats
import pandas as pd
import os

dir = 'F:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())

#anova test
#The one-way ANOVA tests whether the mean of some numeric variable differs 
#across the levels of one categorical variable(do any of the group means differ from one another?)
fare_by_class1 = titanic_train.Fare[titanic_train.Pclass==1]
fare_by_class2 = titanic_train.Fare[titanic_train.Pclass==2]
fare_by_class3 = titanic_train.Fare[titanic_train.Pclass==3]

stats.f_oneway(fare_by_class1, fare_by_class2, fare_by_class3)
