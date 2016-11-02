library(rpart)
library(caret)
library(ggplot2)

setwd("C:/Users/Algorithmica/Downloads")
titanic_train = read.csv("train.csv", na.strings = c("NA",""))
class(titanic_train)
dim(titanic_train)
str(titanic_train)
#preparation of data
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)

#explore single categorical variable
summary(titanic_train$Survived)
X11()
ggplot(titanic_train) + geom_bar(aes(x=Survived))

#explore single continuous variable
summary(titanic_train$Fare)
X11()
ggplot(titanic_train) + geom_histogram(aes(x=Fare))
ggplot(titanic_train) + geom_boxplot(aes(x=factor(0),y=Fare)) + coord_flip()


#explore multi-variate relationship among categorical variables
xtabs(~Sex + Survived, titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Sex, fill=Survived) )

xtabs(~Pclass + Survived, titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Pclass, fill=Survived) )

xtabs(~ Pclass + Survived + Sex, titanic_train)
X11()
ggplot(titanic_train) + geom_bar(aes(x=Pclass, fill=Survived) ) + facet_grid(Sex ~ .)
ggplot(titanic_train) + geom_bar(aes(x=Sex, fill=Survived) ) + facet_grid(Pclass ~ .)

xtabs(~ Embarked + Survived, titanic_train)
xtabs(~ Embarked + Survived + Sex, titanic_train)

xtabs(~ Embarked + Survived +  Pclass + Sex, titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Embarked, fill=Survived) ) + facet_grid(Sex ~ Pclass)

# explore relation betweeen categorical and continuous data
X11()
ggplot(titanic_train) + geom_boxplot(aes(x = Survived, y = Fare))
ggplot(titanic_train) + geom_histogram(aes(x = Survived, y = Fare))
ggplot(titanic_train) + geom_histogram(aes(x = Fare)) + facet_grid(Survived ~ .)
ggplot(titanic_train) + geom_histogram(aes(x = Fare)) + facet_grid(Survived ~ Sex)

# explore relation between continuous variables
ggplot(titanic_train) + geom_point(aes(x = Fare, y = Age))

tata = c(100,150,70,50,200,500)
ms = c(100,120,150,150,160,180)
mad(tata, center = mean(tata))
mad(ms, center = mean(ms))

sd(tata)
sd(ms)

cat_2015 = c(100,99,98,98,97,96)
cat_2016 = c(80,79,79,78,78,77)
cat_2014 = c(140,139,139,139,139,138)

cat_2015_z = (cat_2015 - mean(cat_2015)) / sd(cat_2015)
cat_2016_z = (cat_2016 - mean(cat_2016)) / sd(cat_2016)

cat_scores_z = data.frame(cat_2015, cat_2015_z, cat_2016, cat_2016_z)

cat_scores = data.frame(cat_2014, cat_2015, cat_2016)
?preProcess
#pre process computes required quantities for transformations
preobj = preProcess(cat_scores, method=c("center", "scale"))
preobj$mean
preobj$std
#predict method applies the transformation on data
cat_scores_z = predict(preobj, cat_scores)
