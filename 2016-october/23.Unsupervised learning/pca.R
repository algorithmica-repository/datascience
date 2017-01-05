library(caret)
setwd("D:\\digit_recognizer")

digit_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(digit_train)
str(digit_train)

digit_train1 = digit_train[,-1]
dim(digit_train1)

nzv_obj = nearZeroVar(digit_train1, saveMetrics = T)
digit_train2 = digit_train1[,nzv_obj$zeroVar==F]
dim(digit_train2)

preobj = preProcess(digit_train2, method=c("pca"), thresh = 0.5)
preobj$rotation
digit_train3 = predict(preobj, digit_train2)
dim(digit_train3)
str(digit_train3)
head(digit_train3)
