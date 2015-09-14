library(caret)
setwd("E:/data analytics/kaggle/titanic/data")

titanic = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(titanic)
str(titanic)
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Survived = as.factor(titanic$Survived)

plot(titanic$SibSp, titanic$Parch)
cov(titanic$SibSp, titanic$Parch)
cor(titanic$SibSp, titanic$Parch)

plot(titanic$Fare, titanic$Parch)
cov(titanic$Fare, titanic$Parch)
cor(titanic$Fare, titanic$Parch)

chisq.test(titanic$Pclass, titanic$Survived)
chisq.test(titanic$Embarked, titanic$Survived)
chisq.test(titanic$Survived,titanic$Name)
chisq.test(titanic$Pclass, titanic$Embarked)

setwd("E:/data analytics/datasets")

winedata = read.csv("wine.data", header = TRUE)
dim(winedata)
str(winedata)
head(winedata)

pairs(~X2.8+X3.06,data=winedata)
cor_matrix = cor(winedata)

//Find the variables which have very less variance
nearZvar = nearZeroVar(winedata, saveMetrics = TRUE)
nearZvar

//Find the variables which are highly correlated
corr_matrix = abs(cor(winedata))
diag(corr_matrix) = 0
correlated_col = findCorrelation(corr_matrix, verbose = FALSE , cutoff = .60)
correlated_col

//Remove the variables which have 95% NAs
threshold_val = 0.95 * dim(trainSet)[1]
include_cols = !apply(trainSet, 2, function(y) sum(is.na(y)) > threshold_val)
trainSet = trainSet[, include_cols]



