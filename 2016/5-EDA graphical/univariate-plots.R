library(ggplot2)
library(Amelia)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic = read.table("train.csv", header = TRUE, sep= ",",na.strings=c("NA",""))
dim(titanic)
str(titanic)

titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
missmap(titanic, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

# exploring survived feature using bar chart
table(titanic$Survived)
ggplot(titanic, aes(x = Survived)) + geom_bar()

# exploring pclass feature using bar chart
table(titanic$Pclass)
ggplot(titanic, aes(x = Pclass)) + geom_bar()

# exploring fare feature using boxplot and historgram
summary(titanic$Fare)
ggplot(titanic, aes(x = factor(0), y = Fare)) + geom_boxplot() + coord_flip()
ggplot(titanic, aes(x = Fare)) + geom_histogram()
ggplot(titanic, aes(x = Fare)) + geom_histogram(fill = "white", colour = "black")
ggplot(titanic, aes(x = Fare, y=..density..)) + geom_histogram(fill = "white", colour = "black")
ggplot(titanic, aes(x = Fare)) + geom_histogram(binwidth = 2, fill = "white", colour = "black")
ggplot(titanic, aes(x = Fare)) + geom_density()

# exploring age feature using boxplot and histogram
summary(titanic$Age)
ggplot(titanic, aes(x = factor(0), y = Age)) + geom_boxplot() + coord_flip()

