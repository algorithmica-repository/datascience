library(caret)
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(titanic)
str(titanic)
class(titanic)

names(titanic)
head(titanic)
head(titanic)
tail(titanic)

summary(titanic)
summary(titanic$Sex)
summary(titanic$Age)


