library(caret)
library(randomForest)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\digit-recognizer")
digits = read.csv("train.csv")
dim(digits)
str(digits)
digits$label = as.factor(digits$label)
digits1=digits[,-1]
summary(digits1)

nzv_info = nearZeroVar(digits1,saveMetrics=TRUE)
digits1 = digits1[,nzv_info$nzv==FALSE]
dim(digits1)


cormat = abs(cor(digits1))
dim(cormat)

highlyCorrelatedFeatures = findCorrelation(cormat, cutoff=0.95, verbose = FALSE)
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
