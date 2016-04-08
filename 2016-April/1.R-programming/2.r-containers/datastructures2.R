getwd()
setwd("D:\\kaggle\\titanic\\data") 
titanic = read.csv("train.csv")
class(titanic)
rm(titanic)
gc()
dim(titanic)
str(titanic)
titanic$Name = as.character(titanic$Name)
is.vector(titanic)

titanic[,4]
titanic[1:10,c(1,3,5)]
titanic[1:10,c("PassengerId","Pclass")]

age = c(20,50,10,40,80)
names = c("aaa","bbb",NA,"ddd","eee")
survived=c("T","F","T","F","F")
passengers = data.frame(age,names,survived)
names(passengers) = c("c1","c2","c3")
dim(passengers)
str(passengers)
write.csv(passengers,"passengers.csv")
