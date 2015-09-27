library(caret)
library(e1071)
setwd("E:/data analytics/datasets")

winedata = read.csv("wine.data", header = TRUE)
dim(winedata)
str(winedata)
head(winedata)
names(winedata) = c('Label','Alcohol','MalicAcid')

summary(winedata)

hist(winedata$Alcohol,col="lightblue")
skewness(winedata$Alcohol)
hist(winedata$MalicAcid,col="lightblue")
skewness(winedata$MalicAcid)

preObj = preProcess(winedata[2:3], method=c("BoxCox"))
preObj$bc
winedata1 = predict(preObj,winedata[2:3])
hist(winedata1$Alcohol,col="lightblue")
hist(winedata1$MalicAcid,col="lightblue")

