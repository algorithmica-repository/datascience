library(caret)
library(randomForest)
library(ggplot2)
library(outliers)

setwd("E:/data analytics/kaggle/springleaf")

//Read the dataset from train file
trainSet = read.csv("train.csv", header = TRUE, na.strings=c("NA","","-1","-99999"))
dim(trainSet)
str(trainSet)
head(trainSet)

set.seed(42)
trainSet = trainSet[sample(nrow(trainSet),20000),]
dim(trainSet)

//Remove the variables which have 95% NAs
threshold_val = 0.95 * dim(trainSet)[1]
include_cols = !apply(trainSet, 2, function(y) sum(is.na(y)) > threshold_val)
trainSet = trainSet[, include_cols]

//Remove the variables which have very less variance
nearZvar = nearZeroVar(trainSet, saveMetrics = TRUE)
trainSet = trainSet[ ,nearZvar$nzv==FALSE]

//Remove the variables which are highly correlated
corr_matrix = abs(cor(trainSet[,-dim(trainSet)[2]]))
diag(corr_matrix) = 0
correlated_col = findCorrelation(corr_matrix, verbose = FALSE , cutoff = .95)
trainSet = trainSet[, -c(correlated_col)]





# Set a random seed 


#model tuning strategy
ctrl = trainControl(method = "cv", # Use cross-validation
                    number = 10) # Use 10 folds for cross-validation

model_dt = train(Survived ~ Pclass + Sex + Embarked + Fare, 
                 data = trainSet, 
                 method = "rpart",
                 trControl = ctrl)
model_dt$finalModel
model_dt

model_dt = train(Survived ~ Pclass + Sex + Embarked + Fare, 
                 data = trainSet, 
                 method = "rpart",
                 trControl = ctrl,
                 tuneLength=10)
model_dt$finalModel
model_dt

grid=data.frame(.cp=c(0,0.1,0.6))
model_dt = train(Survived ~ Pclass + Sex + Embarked + Fare, 
                 data = trainSet, 
                 method = "rpart",
                 trControl = ctrl,
                 tuneGrid = grid)
model_dt$finalModel
model_dt

model_dt = train(Survived ~ Pclass + Sex + Age + Embarked + SibSp + Fare, 
                 data = trainSet, control=rpart.control(minsplit=2), 
                 method = "rpart",
                 trControl = ctrl)
model_dt$finalModel
model_dt

model_dt = train(Survived ~ Pclass + Sex + Age + Embarked + SibSp + Fare, 
                 data = trainSet, 
                 method = "J48",
                 trControl = ctrl)
model_dt$finalModel


testSet = read.table("test.csv", sep = ",", header = TRUE)
dim(testSet)
str(testSet)
head(testSet)
testSet$Pclass = factor(testSet$Pclass)
summary(testSet)
testSet$Fare = ifelse(is.na(testSet$Fare), mean(testSet$Fare, na.rm = TRUE), testSet$Fare)


testSet$Survived = predict(model_dt, newdata = testSet)

submission = testSet[,c("PassengerId", "Survived")]

write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")

library()
search()
ls("package:base")
