library(rpart)
library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_total = read.csv("train.csv")
dim(titanic_total)
str(titanic_total)
titanic_total$Pclass = as.factor(titanic_total$Pclass)
titanic_total$Name = as.character(titanic_total$Name)
titanic_total$Survived = as.factor(titanic_total$Survived)

set.seed(100)
train_ind = sample(1:nrow(titanic_total), 600)
titanic_train = titanic_total[train_ind,]
titanic_test = titanic_total[-train_ind,]

dim(titanic_train)
dim(titanic_test)

tr_ctrl0 = trainControl(method="none")
tr_grid = data.frame(.cp=0.02)
model0 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl0, tuneGrid = tr_grid)
pred = predict(model0, titanic_test)
confusionMatrix(pred, titanic_test$Survived)

iterations = 25

tr_ctrl1 = trainControl(method="cv", number = iterations)
model1 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl1, tuneGrid = tr_grid)
cv = as.vector(model1$resample$Accuracy)
var_cv = var(cv)
bias_cv = 0.8247 - 0.7854377

tr_ctrl2 = trainControl(method="LGOCV",p = 0.7,number = iterations)
model2 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl2, tuneGrid = tr_grid)
rep_holdout = as.vector(model2$resample$Accuracy)
var_rep_holdout = var(rep_holdout)
bias_rep_holdout = 0.8247 - 0.7890503

tr_ctrl3 = trainControl(method="boot",number = iterations)
model3 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl3, tuneGrid = tr_grid)
boot = as.vector(model3$resample$Accuracy)
var_boot = var(boot)
bias_boot = 0.8247 - 0.7886627

tr_ctrl4 = trainControl(method="repeatedcv", number = 10, repeats = iterations)
model4 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl4, tuneGrid = tr_grid)
repcv = as.vector(model4$resample$Accuracy)
var_repcv = var(repcv)
bias_repcv = 0.8247 - 0.7893922

df = data.frame(resampling = c("cv","rep_holdout","bootstrap","repeatedcv"), bias = c(bias_cv,bias_rep_holdout,bias_boot,bias_repcv), variance = c(var_cv,var_rep_holdout,var_boot,var_repcv))
