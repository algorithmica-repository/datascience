library(caret)
library(randomForest)
library(ggplot2)
library(Amelia)

setwd("E:/data analytics/kaggle/titanic/data")

trainSet = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(trainSet)
str(trainSet)
head(trainSet)
trainSet$Survived = factor(trainSet$Survived)
trainSet$Pclass = factor(trainSet$Pclass)
summary(trainSet)

missmap(trainSet, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

table(trainSet$Survived)
ggplot(trainSet, aes(x = Survived)) + geom_bar()

#Comparing Survived and passenger class using table and histograms
summary(trainSet$Pclass)
xtabs(~Survived + Pclass, data=trainSet)
ggplot(trainSet, aes(x = Survived, fill = Pclass)) + geom_bar()

#Comparing Survived and Sex using table and histograms
summary(trainSet$Sex)
xtabs(~Survived + Sex, data=trainSet)
ggplot(trainSet, aes(x = Survived, fill = Sex)) + geom_bar()


#Comparing Survived and Embarked using table and histograms
summary(trainSet$Embarked)
xtabs(~Survived + Embarked, data=trainSet)
ggplot(trainSet, aes(x = Survived, fill = Embarked)) + geom_bar()

# Comparing Age and Survived: The boxplots are very similar between Age
# for survivors and those who died. 
xtabs(~Survived + Age, data=trainSet)
ggplot(trainSet, aes(x = Survived, y = Age)) + geom_boxplot() 
# Also, there are lots of NA's. Exclude this variable
summary(trainSet$Age)

# Comparing Survived and Fare: The boxplots are much different between 
# fare for survivors and those who died.
ggplot(trainSet, aes(x = Survived, y = Fare)) + geom_boxplot() 
# Also, there are no NA's. Include this variable.
summary(trainSet$Fare)

# Comparing Survived and Parch
ggplot(trainSet, aes(x = Survived, y = Parch)) + geom_boxplot() 
summary(trainSet$Parch)

# Set a random seed 
set.seed(42)

#model tuning strategy
ctrl = trainControl(method = "cv", # Use cross-validation
                    number = 10) # Use 10 folds for cross-validation

# Train the model using a "random forest" algorithm
model_rf = train(Survived ~ Pclass + Sex + Age + Embarked + SibSp + Fare, 
                 data = trainSet, 
                 method = "rpart",
                 trControl = ctrl)
model_rf

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
