library(caret)
library(randomForest)

setwd("E:/data analytics/kaggle/digit-recognizer")
train_data = read.csv("train.csv")

dim(train_data)
str(train_data)
train_data$label = as.factor(train_data$label)
train_data1=train_data[,-1]

variance_data = nearZeroVar(train_data1,saveMetrics=TRUE)
which(variance_data$nzv==TRUE)
variance_data_rm = train_data1[,variance_data$nzv==FALSE]
dim(variance_data_rm)


correlationMatrix <- abs(cor(variance_data_rm))
dim(correlationMatrix)

highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.95, verbose = TRUE)
reduced_cor_data <-variance_data_rm[,-highlyCorrelated]

dim(reduced_cor_data)


pp_nor_pca = preProcess(variance_data_rm,method = c("center", "pca"),thresh=0.9)

predict_pca = predict(pp_nor_pca,variance_data_rm)
dim(predict_pca)
class(predict_pca)

ctrl = trainControl(method = "cv", # Use cross-validation
                    number = 10) # Use 10 folds for cross-validation

# Train the model using a "random forest" algorithm
model_knn = train(predict_pca,train_data$label, 
                 method = "knn",
                 trControl = ctrl)
summary(model_knn)


test_data = read.csv("test.csv")
dim(test_data)
str(test_data)

variance_data_rm_t = test_data[,-variance_data]
length(variance_data_rm_t)
dim(variance_data_rm_t)

t1 = as.matrix(variance_data_rm_t)
t2=  as.matrix(pp_nor_pca$rotation)
dim(t1)
dim(t2)

new_t3 = t1%*%t2;
dim(new_t3)


new_t3$label = predict(model_knn, newdata = new_t3)
new_t3$ImageId = 1:nrow(test_data)

submission = new_t3[,c("ImageId","label")]


write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
