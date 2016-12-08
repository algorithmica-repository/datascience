library(caret)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""), stringsAsFactors = F)
dim(restaurant_train)
str(restaurant_train)

restaurant_train$City.Group = as.factor(restaurant_train$City.Group)
restaurant_train$Type = as.factor(restaurant_train$Type)
#add the missing level to existing levels
levels(restaurant_train$Type) = c(levels(restaurant_train$Type), "MB")

#explore the relation between revenue vs type of restaurant
X11()
ggplot(restaurant_train) + geom_histogram(aes(x = revenue)) + facet_grid(Type ~ .)

restaurant_train$num_days = as.numeric(as.Date("31-12-2014", format="%d-%m-%Y") - as.Date(restaurant_train$Open.Date, format= "%m/%d/%Y"))
restaurant_train1 = restaurant_train[,-c(1,2,3)]

set.seed(100)
resampling_strategy = trainControl(method="cv", number=50)
knn_model = train(revenue ~ ., restaurant_train1, method="knn", trControl = resampling_strategy)

restaurant_test = read.csv("test.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_test)
restaurant_test1 = restaurant_test[,-c(1,2,3)] 

restaurant_test1$Id = restaurant_test$Id
restaurant_test1$Prediction = predict(knn_model, restaurant_test1)
result = restaurant_test1[,c("Id","Prediction")]
write.csv(result,"submission.csv",row.names = F)

