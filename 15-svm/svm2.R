library(caret)
library(e1071)
library(ggplot2)
library(Amelia)
library(kernlab)
library(ISLR)

data(Auto)

str(Auto)
dim(Auto)
head(Auto)
summary(Auto)

Auto$y = NA
Auto$y[Auto$mpg > median(Auto$mpg)] = 1
Auto$y[Auto$mpg <= median(Auto$mpg)] = 0
Auto$y = as.factor(Auto$y)


missmap(Auto, main="Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

# Set a random seed 
set.seed(42)

#model tuning strategy
ctrl = trainControl(method = "cv", # Use cross-validation
                    number = 10) # Use 10 folds for cross-validation

preProc_opt = c("knnImpute", "center", "scale")

Lmodel = train(y ~ ., preProc=preProc_opt,
                 data = Auto, 
                 method = "svmLinear",
                 trControl = ctrl, tuneLength=5)
Lmodel
Lmodel$finalModel

plot(Lmodel)

Pmodel = train(y ~ ., data = Auto, preProc=preProc_opt,
                   method = "svmPoly",
                   trControl = ctrl, tuneLength=5)
Pmodel

Rmodel = train(y ~ ., preProc=preProc_opt,
                   data = Auto, 
                   method = "svmRadial",
                   trControl = ctrl, tuneLength=5)
Rmodel
plot(Rmodel)

resamps = resamples(list(Linear = Lmodel, Poly = Pmodel, Radial = Rmodel))
summary(resamps)
bwplot(resamps, metric = "Accuracy")
densityplot(resamps, metric = "Accuracy")





testSet = read.table("test.csv", sep = ",", header = TRUE)
dim(testSet)
str(testSet)
head(testSet)
testSet$Pclass = factor(testSet$Pclass)
summary(testSet)
testSet$Fare = ifelse(is.na(testSet$Fare), mean(testSet$Fare, na.rm = TRUE), testSet$Fare)


testSet$Survived = predict(model_logit, newdata = testSet)

submission = testSet[,c("PassengerId", "Survived")]

write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
