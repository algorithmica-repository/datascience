library(caret)
library(RANN)
setwd("E:/data analytics/kaggle/titanic/data")

titanic = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(titanic)
str(titanic)
summary(titanic)

preObj = preProcess(titanic[,c("Age", "Pclass")], method=c("medianImpute"))
predict(preObj, titanic[,c("Age", "Pclass")])

preObj = preProcess(titanic[,c("Age", "Pclass")], method=c("knnImpute"))
predict(preObj, titanic[,c("Age", "Pclass")])
