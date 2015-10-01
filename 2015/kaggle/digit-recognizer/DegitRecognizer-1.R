library(caret)

setwd("E:/data analytics/kaggle/digit-recognizer")
train_data = read.csv("train.csv")

dim(train_data)
str(train_data)
train_data$label = as.factor(train_data$label)
train_data1=train_data[,-1]

near_zero_var = nearZeroVar(train_data1,saveMetrics=TRUE)
which(near_zero_var$nzv==TRUE)
train_data1 = train_data1[,near_zero_var$nzv==FALSE]
dim(train_data1)

pca_obj = preProcess(train_data1,method = c("center", "pca"))

pcmp = predict(pca_obj,train_data1)
dim(pcmp)
class(pcmp)

ctrl = trainControl(method = "cv", # Use cross-validation
                    number = 10) # Use 10 folds for cross-validation

model_knn = train(pcmp,train_data$label, 
                 method = "nb",
                 trControl = ctrl)
model_knn
summary(model_knn)


test_data = read.csv("test.csv")
dim(test_data)
str(test_data)

test_data = test_data[,near_zero_var$nzv==FALSE]
dim(test_data)

test_data = predict(pca_obj,test_data)

test_data$label = predict(model_knn, newdata = test_data)
test_data$ImageId = 1:nrow(test_data)

submission = test_data[,c("ImageId","label")]


write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
