library(caret)

# toy example to understand zero and near-zero variance
v1 = rep(10, 10)
v2 = c(rep(20,5), rep(30,5))
df = data.frame(v1, v2)

nzv_obj = nearZeroVar(df, saveMetrics = T)
df1 = df[,nzv_obj$zeroVar==F]

# Applying the idea to real data set
setwd("D:\\digit_recognizer")

digit_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(digit_train)
str(digit_train)

nzv_obj = nearZeroVar(digit_train, saveMetrics = T)
digit_train1 = digit_train[,nzv_obj$zeroVar==F]
dim(digit_train1)
digit_train2 = digit_train[,nzv_obj$nzv==F]
dim(digit_train2)

