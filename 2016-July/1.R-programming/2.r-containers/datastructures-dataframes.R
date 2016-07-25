setwd("D:\\kaggle\\titanic\\data") 

#r function that return data frame
titanic = read.csv("train.csv")
class(titanic)
dim(titanic)
str(titanic)

#access the contents of dataframe
titanic[,1]
titanic$PassengerId
titanic[1:10,c(1,3,5)]
titanic[1:10,c("PassengerId","Pclass")]
titanic[titanic$Sex=="Female",c(1,2,3,4,5)]

#creating data frames
age = c(20,50,10,40,80)
names = c("aaa","bbb",NA,"ddd","eee")
survived=c("T","F","T","F","F")
passengers = data.frame(age,names,survived)
names(passengers) = c("c1","c2","c3")
dim(passengers)
str(passengers)
