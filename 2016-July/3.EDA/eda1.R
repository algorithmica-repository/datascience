install.packages("ggplot2")
library(ggplot2)
setwd("D:/kaggle/titanic/data/")
titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)
summary(titanic_train$Pclass)
summary(titanic_train$Sex)
summary(titanic_train$Survived)
#create new window 
X11()
ggplot(titanic_train) + geom_bar(aes(x=Survived))

summary(titanic_train$Fare)
X11()
ggplot(titanic_train) + geom_histogram(aes(x=Fare),fill = "white", colour = "black")
ggplot(titanic_train) + geom_boxplot(aes(x=factor(0),y=Fare)) + coord_flip()
ggplot(titanic_train) + geom_density(aes(x=Fare))

xtabs(~Survived+Sex,titanic_train)

xtabs(~Survived+Pclass,titanic_train)

xtabs(~Survived+Pclass+Sex,titanic_train)

xtabs(~Survived+Embarked+Pclass+Sex,titanic_train)

titanic_test = read.csv("test.csv")
dim(titanic_test)
nrow(titanic_test)
str(titanic_test)

# tmp = numeric(nrow(titanic_test))
# for(i in 1:nrow(titanic_test)) {
#     if(titanic_test[i,"Sex"]== "female" && titanic_test[i,"Pclass"]== "1")
#       tmp[i] = 1
#     if(titanic_test[i,"Sex"]== "male" && titanic_test[i,"Pclass"]== "1")
#       tmp[i] = 1
# }

titanic_test$Survived = ifelse(titanic_test$Pclass == "1",1,0)
write.csv(titanic_test[,c("PassengerId","Survived")],"subm5.csv", row.names=F)

 