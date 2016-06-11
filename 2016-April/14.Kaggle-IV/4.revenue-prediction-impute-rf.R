library(ggplot2)
library(rpart)
library(caret)
library(corrplot)
library(reshape2)
library(Amelia)
library(doParallel)

#register cluster for parallel processing
cl = makeCluster(detectCores())
registerDoParallel(cl)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets\\restaurant-revenue")

restaurant_train = read.csv("train.csv", na.strings=c("","NA"))
restaurant_train$City = as.character(restaurant_train$City)
levels(restaurant_train$Type) = c(levels(restaurant_train$Type),"MB")
dim(restaurant_train)
str(restaurant_train)

###Step1: EDA
#Exploring structure of data
summary(restaurant_train)
X11()
ggplot(restaurant_train) + geom_histogram(aes(x = revenue), fill = "white", colour = "black")

#Exploring missing data
X11()
missmap(restaurant_train)
X11()
d = melt(restaurant_train[,-c(1:5)])
str(d)
dim(d)
ggplot(d,aes(x = value)) + facet_wrap(~variable,scales = "free_x") + geom_histogram()

#Exploring relation between city group and revenue features
X11()
ggplot(restaurant_train, aes(x=City.Group, y=revenue)) + geom_point(shape=1) 

#Exploring relation between restaurant type and revenue features
X11()
ggplot(restaurant_train, aes(x=Type, y=revenue)) + geom_point(shape=1) 

#Exploring relation between demographic, real-estate, commercial and revenue features
X11()
featurePlot(restaurant_train[,c('P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12')],
            restaurant_train$revenue,
            plot="scatter",
            type = c("g", "p", "smooth"),
            between = list(x = 1, y = 1),
            labels = rep("", 2))
X11()
featurePlot(restaurant_train[,c('P13','P14','P15','P16','P17','P18','P19', 'P20','P21','P22','P23','P24')],
            restaurant_train$revenue,
            plot="scatter",
            type = c("g", "p", "smooth"),
            between = list(x = 1, y = 1),
            labels = rep("", 2))
X11()
featurePlot(restaurant_train[,c('P25','P26','P27','P28','P29','P30', 'P31','P32','P33','P34','P35','P36', 'P37')],
            restaurant_train$revenue,
            plot="scatter",
            type = c("g", "p", "smooth"),
            between = list(x = 1, y = 1),
            labels = rep("", 2))

#Exploring correlations among features
numeric_attr = sapply(restaurant_train, is.numeric)
correlations = cor(restaurant_train[,numeric_attr])
X11()
corrplot(correlations, order = "hclust", addrect=3)


###Step:2 Data preprocessing
impute = function(data) {
  imputeObj = preProcess(data, method="bagImpute")
  return (imputeObj)
}
restaurant_train1 = restaurant_train[,-c(1:3,ncol(restaurant_train))]
restaurant_train1[restaurant_train1==0] = NA
imputeObj = impute(restaurant_train1)
restaurant_train1 = predict(imputeObj,restaurant_train1)
dim(restaurant_train1)
str(restaurant_train1)

###Step3: Feature engineering
#a.finding and filtering zero/near-zero variance features
remove.nzv.features = function(data) {
  var_obj = nearZeroVar(data,saveMetrics=TRUE)
  return(var_obj)
}
var_obj = remove.nzv.features(restaurant_train1)
restaurant_train2 = restaurant_train1[, var_obj$zeroVar==FALSE]
dim(restaurant_train2)
str(restaurant_train2)

#b.adding new features
add.features = function(data) {
  tmp.df = data.frame(Date = as.Date(data, "%m/%d/%Y"))
  
  year = as.numeric(substr(as.character(tmp.df$Date),1,4))
  tmp.df = cbind(tmp.df, year = year)
  
  month = as.numeric(substr(as.character(tmp.df$Date),6,7))
  tmp.df = cbind(tmp.df, month = month)
  
  day = as.numeric(substr(as.character(tmp.df$Date),9,10))
  tmp.df = cbind(tmp.df, day = day)
  
  days = as.numeric(as.Date("01/01/2015", format="%m/%d/%Y") - tmp.df$Date) 
  tmp.df = cbind(tmp.df, days = days)
  
  return(tmp.df)
}
features_new_df = add.features(restaurant_train$Open.Date)
restaurant_train3 = cbind(restaurant_train2, features_new_df)
dim(restaurant_train3)
str(restaurant_train3)

##Step4: Model building - Random Forest
set.seed(100)
tr_ctrl = trainControl(method="boot")
model_rf = train(x = restaurant_train3, y = restaurant_train[,"revenue"],
               method = "rf", trControl = tr_ctrl, importance=T)
model_rf
plot(model_rf)
model_rf$finalModel
varImp(model_rf)
X11()
varImpPlot(model_rf$finalModel)

stopCluster(cl)

##Step5: Deployment and prediction outcomes for test data
restaurant_test = read.csv("test.csv", na.strings=c("","NA"))
dim(restaurant_test)
str(restaurant_test)

restaurant_test1 = restaurant_test[,-c(1:3)]
restaurant_test1[restaurant_test1==0] = NA
restaurant_test1 = predict(imputeObj,restaurant_test1)
dim(restaurant_test1)
str(restaurant_test1)

restaurant_test2 = restaurant_test1[, var_obj$zeroVar==FALSE]
dim(restaurant_test2)
str(restaurant_test2)

features_new_df = add.features(restaurant_test$Open.Date)
restaurant_test3 = cbind(restaurant_test2, features_new_df)
dim(restaurant_test3)
str(restaurant_test3)

restaurant_test3$revenue = predict(model_rf,restaurant_test3)
result = data.frame(Id=restaurant_test[,"Id"], revenue=restaurant_test3[,"revenue"])
write.csv(result,"submission.csv",row.names = F)

