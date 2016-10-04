library(caret)
library(doParallel)

#register cluster for parallel processing
cl = makeCluster(detectCores())
registerDoParallel(cl)

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

tr_ctrl1 = trainControl(method = "cv", verboseIter = T) 
tn_grid = data.frame(.k=c(4,5))
model_knn = train(x = digit_train2, digit_train$label, 
                 method = "knn",
                 trControl = tr_ctrl1, tuneGrid = tn_grid)
model_knn

# model_rf = train(x = digit_train2, digit_train$label, 
#                   method = "rf",
#                   trControl = tr_ctrl)
# model_rf

stopCluster(cl)

digit_test = read.csv("test.csv")
dim(digit_test)

digit_test1 = digit_test[,var_obj$zeroVar==FALSE]
dim(digit_test1)
digit_test2 = predict(pca_obj,digit_test1)
dim(digit_test2)

digit_test2$label = predict(model_knn, digit_test2)
digit_test2$ImageId = 1:nrow(digit_test2)
submission = digit_test2[,c("ImageId","label")]
write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
