library(ggplot2)
library(rpart)
library(caret)
library(corrplot)
library(reshape2)
library(Amelia)
library(doParallel)
library(mice)

#register cluster for parallel processing
cl = makeCluster(detectCores())
registerDoParallel(cl)


setwd("C:/Users/Thimma Reddy/Documents/GitHub/datascience/datasets/restaurant-revenue")

restaurant_train = read.csv("train.csv", na.strings=c("","NA"))
restaurant_test = read.csv("test.csv", na.strings=c("","NA"))

#combining train and test datasets for handling factor type differences
restaurant_test$revenue = NA
restaurant = rbind(restaurant_train, restaurant_test)
dim(restaurant)
str(restaurant)

restaurant_train = restaurant[1:137,]
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
#handling missing data
impute = function(data, method='rf') {
  data[data==0] = NA
  tempData = mice(data, m=1, maxit=2, meth=method, 
                   seed=501, printFlag=FALSE)
  return (complete(tempData, 1))
}
restaurant_train1 = impute(restaurant_train)
dim(restaurant_train1)
str(restaurant_train1)
restaurant_train1 = restaurant_train

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

#b.finding and filtering highly correlated features
remove.correlated.features = function(data) {
  nums = sapply(data, is.numeric)
  correlations = abs(cor(data[,nums]))
  highCorr = findCorrelation(correlations, cutoff = .85)
  return(highCorr)
}
features_highCorr = remove.correlated.features(restaurant_train2)
tmp.df = restaurant_train2[,c("Open.Date")]
restaurant_train3 = cbind(restaurant_train2[,-features_highCorr], Open.Date = tmp.df)
dim(restaurant_train3)
str(restaurant_train3)

#c.adding new features
add.features = function(data) {
  tmp.df = data.frame(Date = as.Date(data, "%m/%d/%Y"))
  
  year = as.numeric(substr(as.character(tmp.df$Date),1,4))
  tmp.df = cbind(tmp.df, year = year)
  
  month = as.numeric(substr(as.character(tmp.df$Date),6,7))
  tmp.df = cbind(tmp.df, month = month)
  
  day = as.numeric(substr(as.character(tmp.df$Date),9,10))
  tmp.df = cbind(tmp.df, day = day)
  
  min.year = head(sort(year))[1]
  days = as.numeric(tmp.df$Date - as.Date(paste0(min.year,"-01-01"))) 
  tmp.df = cbind(tmp.df, days = days)
  
  return(tmp.df)
}
features_new_df = add.features(restaurant_train3$Open.Date)
restaurant_train4 = cbind(restaurant_train3, features_new_df)
dim(restaurant_train4)
str(restaurant_train4)

##Step4: Model building
set.seed(100)
tr_ctrl = trainControl(method="cv", number = 10)
features.exclude = c(1,2,25,26)
model = train(x = restaurant_train4[,-features.exclude], y = restaurant_train4[,"revenue"],
              method = "rf", trControl = tr_ctrl)
model
model$finalModel

gbmGrid = expand.grid(interaction.depth = seq(1, 7, by = 2),
                      n.trees = seq(100, 1000, by = 50),
                      shrinkage = c(0.01, 0.1),
                      n.minobsinnode = c(5,10))
model3 = train(revenue ~ ., data = restaurant_train2,method = "gbm", trControl = tr_ctrl, tuneGrid = gbmGrid, importance=T)
model3
model3$finalModel
varImp(model3, scale=F)
X11()
plot(model3)

stopCluster(cl)

##Step5: Deployment and prediction outcomes for test data
restaurant_test = restaurant[138:nrow(restaurant),]
dim(restaurant_test)
str(restaurant_test)

restaurant_test1 = impute(restaurant_test, method="sample")

restaurant_test2 = restaurant_test1[, var_obj$zeroVar==FALSE]

tmp.df = restaurant_test2[,c("Open.Date")]
restaurant_test3 = cbind(restaurant_test2[,-features_highCorr], Open.Date = tmp.df)

features_new_df = add.features(restaurant_test3$Open.Date)
restaurant_test4 = cbind(restaurant_test3, features_new_df)
dim(restaurant_test4)
str(restaurant_test4)

restaurant_test4$revenue = predict(model,restaurant_test4[,-features.exclude])
result = restaurant_test4[,c("Id","revenue")]
names(result) = c("Id","Prediction")
write.csv(result,"submission.csv",row.names = F)

