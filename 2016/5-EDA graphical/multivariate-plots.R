library(ggplot2)
library(GGally)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic = read.table("train.csv", header = TRUE, sep= ",",na.strings=c("NA",""))
dim(titanic)
str(titanic)

# converting the types of features
titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Family = titanic$SibSp + titanic$Parch + 1
titanic = titanic[c("Pclass","Survived","Sex","Age","Family","Fare","Embarked")]

ggpairs(titanic, columns=1:4,axisLabels="show")

plot  = ggpairs(titanic, axisLabels="show")

pdf("out2.pdf", height=500, width=500)
print(plot)
dev.off()

ggpairs(titanic,columns=1:4, 
             upper = list(continuous = "density", discrete="facetbar", combo="box"),
             lower = list(discrete="facetbar", continous="cor", combo = "box")
             )

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")

winedata = read.csv("wine.txt", header = TRUE)
dim(winedata)
str(winedata)
head(winedata)

cor_matrix = cor(winedata)
plot = ggpairs(winedata,columns=1:4, axisLabels="show")

plot = ggpairs(winedata, axisLabels="show")



