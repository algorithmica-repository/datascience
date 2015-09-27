library(caret)
library(randomForest)
library(ggplot2)
library(Amelia)
library(Hmisc)

setwd("E:/data analytics/kaggle/titanic/data")

readData = function(path.name, file.name, column.types, missing.types) {
  read.csv(paste(path.name, file.name, sep="/"), header=TRUE, 
            colClasses=column.types,
            na.strings=missing.types )
}

changeNames = function(name) {
  master_vector = grep("Master\\.",name)
  miss_vector = grep("Miss\\.", name)
  mrs_vector = grep("Mrs\\.", name)
  mr_vector = grep("Mr\\.", name)
  dr_vector = grep("Dr\\.", name)
  ms_vector = grep("Ms\\.", name)
  
  for(i in master_vector) {
    name[i] = "Master"
  }
  for(i in miss_vector) {
    name[i] = "Miss"
  }
  for(i in mrs_vector) {
    name[i] = "Mrs"
  }
  for(i in mr_vector) {
    name[i] = "Mr"
  }
  for(i in dr_vector) {
    name[i] = "Dr"
  } 
  for(i in ms_vector) {
    name[i] = "Mrs"
  }
  return (name);
}


imputeMean = function(impute.var, filter.var, var.levels) {
  for (v in var.levels) {
    impute.var[ which( filter.var == v)] = 
      impute(impute.var[which( filter.var == v)],mean)
  }
  return (impute.var)
}

imputeEmbarked = function(impute.var) {
  impute.var[which(is.na(impute.var))] = 'S'
  return (impute.var)
}

imputeFare = function(impute.var) {
  impute.var = ifelse(is.na(impute.var), mean(impute.var, na.rm = TRUE), impute.var)
  return (impute.var)
}
 
titanic.path = getwd()
train.data.file = "train.csv"
missing.types = c("NA", "")
train.column.types = c('integer',   # PassengerId
                        'factor',    # Survived 
                        'factor',    # Pclass
                        'character', # Name
                        'factor',    # Sex
                        'numeric',   # Age
                        'integer',   # SibSp
                        'integer',   # Parch
                        'character', # Ticket
                        'numeric',   # Fare
                        'character', # Cabin
                        'factor'     # Embarked
)
trainSet= readData(titanic.path, train.data.file, 
                      train.column.types, missing.types)

missmap(trainSet, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

trainSet$Name = changeNames(trainSet$Name)
names.na.train = c("Dr", "Master", "Mrs", "Miss", "Mr")
trainSet$Age = imputeMean(trainSet$Age, trainSet$Name, names.na.train)
trainSet$Embarked = imputeEmbarked(trainSet$Embarked)
trainSet$Fare = imputeFare(trainSet$Fare)

missmap(trainSet, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

dim(trainSet)
str(trainSet)
head(trainSet)
summary(trainSet)


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
model_rf = train(Survived ~ Pclass + Sex + Age + Embarked + SibSp, 
              data = trainSet, 
              method = "rf",
              trControl = ctrl)
model_rf

# Train the model using a "logistic regression" algorithm
model_logit = train(Survived ~ Pclass + Sex + Age + SibSp + Sex * Pclass, 
                    data = trainSet, 
                    method = "glm", family=binomial(link='logit'),
                    trControl = ctrl)
model_logit


test.data.file = "test.csv"
test.column.types = train.column.types[-2]

testSet = readData(titanic.path, test.data.file, 
                     test.column.types, missing.types)
dim(testSet)
str(testSet)
head(testSet)
summary(testSet)

testSet$Name = changeNames(testSet$Name)
testSet$Age = imputeMean(testSet$Age, testSet$Name, 
                             names.na.train)
testSet$Embarked = imputeEmbarked(testSet$Embarked)

testSet$Survived = predict(model_rf, newdata = testSet)

submission = testSet[,c("PassengerId", "Survived")]

write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
