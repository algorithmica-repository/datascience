library(ggplot2)
setwd("D:/kaggle/titanic/data/")
titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

##univariate EDA
#categorical variables
X11()
xtabs(~Survived,titanic_train)
summary(titanic_train$Survived)
ggplot(titanic_train) + geom_bar(aes(x=Survived))

summary(titanic_train$Sex)
ggplot(titanic_train) + geom_bar(aes(x=Sex))

summary(titanic_train$Pclass)
ggplot(titanic_train) + geom_bar(aes(x=Pclass))

#numerical variables
summary(titanic_train$Fare)
ggplot(titanic_train) + geom_histogram(aes(x=Fare),fill = "white", colour = "black")
ggplot(titanic_train) + geom_boxplot(aes(x=factor(0),y=Fare)) + coord_flip()
ggplot(titanic_train) + geom_density(aes(x=Fare))

summary(titanic_train$Age)
ggplot(titanic_train) + geom_histogram(aes(x=Age),fill = "white", colour = "black")
ggplot(titanic_train) + geom_boxplot(aes(x=factor(0),y=Age)) + coord_flip()

##bivariate EDA
#C-C relationships
X11()
xtabs(~Survived+Sex,titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Sex, fill=Survived) )

xtabs(~Survived+Pclass,titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Pclass, fill=Survived) )

xtabs(~Survived+Embarked,titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Embarked, fill=Survived) )

#N-C relationships
ggplot(titanic_train) + geom_boxplot(aes(x = Survived, y = Age))
ggplot(titanic_train) + geom_histogram(aes(x = Age),fill = "white", colour = "black") + facet_grid(Survived ~ .)

ggplot(titanic_train) + geom_boxplot(aes(x = Survived, y = Fare))
ggplot(titanic_train) + geom_histogram(aes(x = Fare),fill = "white", colour = "black") + facet_grid(Survived ~ .)

##multivariate EDA
X11()
xtabs(~Survived+Pclass+Sex,titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Sex, fill=Survived)) + facet_grid(Pclass ~ .)

X11()
xtabs(~Survived+Embarked+Sex,titanic_train)
ggplot(titanic_train) + geom_bar(aes(x=Sex, fill=Survived)) + facet_grid(Embarked ~ .)

titanic_test = read.csv("test.csv")
dim(titanic_test)
nrow(titanic_test)
str(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)

tmp = numeric(nrow(titanic_test))
for(i in 1:nrow(titanic_test)) {
   if(titanic_test[i,"Sex"]== "female") {
     if(titanic_test[i,"Pclass"]== "1" | titanic_test[i,"Pclass"]== "2")
        tmp[i] = 1
     else if(titanic_test[i,"Embarked"]== "S")
       tmp[i] = 0
     else
       tmp[i] = 1
   } else  {
     if(titanic_test[i,"Pclass"]== "2")
       tmp[i] = 0
     else if(titanic_test[i,"Embarked"]== "S")
       tmp[i] = 1
     else
       tmp[i] = 0
   }   
}

titanic_test$Survived = tmp
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names=F)

 