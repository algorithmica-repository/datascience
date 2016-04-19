library(ggplot2)
library(Amelia)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic = read.csv("train.csv",na.strings=c("NA",""))
dim(titanic)
str(titanic)

#converting the inferred types to required types
titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
missmap(titanic, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)

# exploring survived feature using frequency table and bar chart
xtabs(~Survived,data = titanic)
ggplot(titanic) + geom_bar(aes(x = Survived))

# exploring pclass feature using frequency table and bar chart
xtabs(~Pclass,data = titanic)
ggplot(titanic) + geom_bar(aes(x = Pclass))

# exploring fare feature using centre+spread and boxplot, historgram and density plots
summary(titanic$Fare)
ggplot(titanic) + geom_boxplot(aes(x = factor(0), y = Fare))
ggplot(titanic) + geom_boxplot(aes(x = factor(0), y = Fare)) + coord_flip()
ggplot(titanic) + geom_histogram(aes(x = Fare))
ggplot(titanic) + geom_histogram(aes(x = Fare), fill = "white", colour = "black")
ggplot(titanic) + geom_histogram(aes(x = Fare), binwidth = 2, fill = "white", colour = "black")
ggplot(titanic) + geom_histogram(aes(x = Fare, y=..density..), fill = "white", colour = "black")
ggplot(titanic) + geom_density(aes(x = Fare))

# exploring age feature using centre+spread and boxplot
summary(titanic$Age)
ggplot(titanic, aes(x = factor(0), y = Age)) + geom_boxplot() + coord_flip()

