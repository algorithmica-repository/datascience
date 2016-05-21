library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\digit-recognizer")
digit_train = read.csv("train.csv")
dim(digit_train)
str(digit_train)
digit_train$label = as.factor(digit_train$label)

digit_train1 = digit_train[,-1]
dim(digit_train1)
var_obj = nearZeroVar(digit_train1,saveMetrics=TRUE, allowParallel = T)
digit_train1 = digit_train1[,var_obj$zeroVar==FALSE]
dim(digit_train1)

pca_obj = preProcess(digit_train1,method = c("pca"))
pca_obj
pca_obj$rotation
digit_train2 = predict(pca_obj,digit_train1)
dim(digit_train2)