library(ggplot2)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic = read.csv("train.csv",na.strings=c("NA",""))
dim(titanic)
str(titanic)

# converting the types of features
titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
titanic = titanic[c("Pclass","Survived","Sex","Age","Family","Fare","Embarked")]

ggplot(titanic, aes(x=Sex, fill=Pclass)) + geom_bar(position="fill") + facet_grid(Survived ~ .)

#Plotting all pair wise relationships
ggpairs(titanic, columns=1:4,axisLabels="show")

#Saving the plot
plot  = ggpairs(titanic, axisLabels="show")
pdf("out2.pdf", height=500, width=500)
print(plot)
dev.off()

#Plotting all pair wise relationships with custom plots
ggpairs(titanic,columns=1:4, 
        upper = list(continuous = "density", discrete="facetbar", combo="box"),
        lower = list(discrete="facetbar", continous="cor", combo = "box")
)
