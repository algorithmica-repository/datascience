library(caret)
library(randomForest)
library(foreach) 
library(doParallel)
library(ROCR)

#register cluster for parallel processing
cl = makeCluster(4)
registerDoParallel(cl)

setwd("C:\\Users\\Thimma Reddy\\machine learning\\kaggle")
set.seed(100)

#Read the data
traindata = read.csv("train.csv",header = TRUE, stringsAsFactors = FALSE)
dim(traindata)
target = as.factor(traindata[,ncol(traindata)])
traindata = traindata[,-c(1,ncol(traindata))]
dim(traindata)

#Removing zerovariance data
zvdata = nearZeroVar(traindata,saveMetrics = TRUE)
nzvnames = names(traindata)[which(zvdata$zeroVar==FALSE)]
traindata1 = traindata[,nzvnames]
dim(traindata1)

#Build random forest model in parallel
model_rf = foreach(ntree=rep(25,8), .combine=combine, .multicombine=TRUE,
                  .packages="randomForest") %dopar% {
                    randomForest(traindata1, target,ntree=ntree,
                          strata=target,
                          do.trace=TRUE, importance=TRUE, forest=TRUE,
                          replace=TRUE, classwt=c(0.5,0.5))
                  }
stopCluster(cl)

#find the important variables from random forest model
model_rf
importance(model_rf)
varImpPlot(model_rf)

#Find the right cutoff from oob data
oob.votes = predict(model_rf, traindata1, type = 'prob')
pred = prediction(oob.votes[,2],target)
roc.perf = performance(pred, "tpr","fpr")
auc = performance(pred, "auc");
auc
plot(roc.perf)
#str(roc.perf)
df = data.frame(cut = roc.perf@alpha.values[[1]], fpr = roc.perf@x.values[[1]], tpr = roc.perf@y.values[[1]])
cutoff = 0.39 

#Read and prepare testdata
testdata = read.csv("test.csv",header = TRUE, stringsAsFactors = FALSE)
dim(testdata)
testdata1 = testdata[,-1]
testdata1 = testdata1[,nzvnames]
dim(testdata1)

# cross checking same features exist in both train and testdata
#names(testdata1) == names(traindata1)

#Do the predictions for test data and make it ready for submission
oob.votes = predict(model_rf, testdata1, type = 'prob')
target1 = numeric(nrow(testdata1))

#target1[which(oob.votes[,2]<cutoff)] = 0
target1[which(oob.votes[,2]>=cutoff)] = 1

submission = data.frame(testdata[,1],target1)
names(submission) = c("ID","TARGET")
write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")


