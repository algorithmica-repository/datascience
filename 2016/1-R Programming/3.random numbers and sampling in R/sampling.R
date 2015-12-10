#Get 5 decimal random numbers between 0 and 1
runif(5)

#Get 10 decimal random numbers between 1 and 5.5
runif(10,1,5.5)

# Get and integer sample of size 10 between 1 and 20 without replacement
sample(1:20,10,F)

# Get and integer sample of size 10 between 1 and 20 with replacement
sample(1:20,10,T)

# Understanding the need of setting seed value of random generator
set.seed(100)
sample(1:20,5)

set.seed(100)
sample(1:20,5)

setwd("E:/data analytics/kaggle/titanic/data")
titanic_train = read.table("train.csv", TRUE, ",")
# Extracting 10-size random sample from data frame
titanic_train[sample(1:nrow(titanic_train),10),]

