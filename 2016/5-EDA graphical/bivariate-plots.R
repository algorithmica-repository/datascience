library(ggplot2)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic = read.table("train.csv", header = TRUE, sep= ",",na.strings=c("NA",""))
dim(titanic)

# converting the types of features
titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Family = titanic$SibSp + titanic$Parch + 1

# standardization of features
titanic_n = titanic[sapply(titanic, is.numeric)]
preObj = preProcess(titanic_n, method=c("center","scale"))
titanic_std = predict(preObj,titanic_n)

# Comparing Survived and passenger class using table and histograms
xtabs(~Survived + Pclass, data=titanic)
ggplot(titanic, aes(x = Survived, fill = Pclass)) + geom_bar()

# Comparing Survived and Sex using table and histograms
xtabs(~Survived + Sex, data=titanic)
ggplot(titanic, aes(x = Survived, fill = Sex)) + geom_bar()

# Comparing Survived and Embarked using table and histograms
xtabs(~Survived + Embarked, data=titanic)
ggplot(titanic, aes(x = Survived, fill = Embarked)) + geom_bar()

# Comparing Age and Survived using boxplots 
ggplot(titanic, aes(x = Survived, y = Age)) + geom_boxplot() 
summary(titanic$Age)

# Comparing Survived and Fare using boxplots 
ggplot(titanic, aes(x = Survived, y = Fare)) + geom_boxplot() 

# Comparing Survived and Family using boxplots
ggplot(titanic, aes(x = Survived, y = Family)) + geom_boxplot()

# Comparing Sibsp and Fare using scatterplot
cor(titanic_std$SibSp, titanic_std$Fare)
ggplot(titanic_std, aes(x = SibSp, y = Fare)) + geom_point()

# Comparing Parch and Fare using scatterplot
cor(titanic_std$Parch, titanic_std$Fare)
ggplot(titanic_std, aes(x = Parch, y = Fare)) + geom_point()

# Comparing Family and Parch using scatterplot
cor(titanic_std$Parch, titanic_std$Family)
ggplot(titanic_std, aes(x = Parch, y = Family)) + geom_point()
