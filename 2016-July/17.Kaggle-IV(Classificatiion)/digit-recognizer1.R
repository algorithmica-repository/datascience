library(caret)
library(doParallel)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\digit-recognizer")
digit_train = read.csv("train.csv")
dim(digit_train)
str(digit_train)
digit_train$label = as.factor(digit_train$label)


digit_test = read.csv("test.csv")
dim(digit_test)

digit_test$label = sample(0:9, nrow(digit_test),replace = T)
digit_test$ImageId = 1:nrow(digit_test)
submission = digit_test[,c("ImageId","label")]
write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
