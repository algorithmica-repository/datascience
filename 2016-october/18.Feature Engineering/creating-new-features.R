library(caret)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""), stringsAsFactors = T)
dim(restaurant_train)
str(restaurant_train)

restaurant_train$Open.Date = as.character(restaurant_train$Open.Date)
#add the missing level to existing levels
levels(restaurant_train$Type) = c(levels(restaurant_train$Type), "MB")

#explore the relation between revenue vs type of restaurant
X11()
ggplot(restaurant_train) + geom_histogram(aes(x = revenue)) + facet_grid(Type ~ .)

#creating a new feature
restaurant_train$num_days = as.numeric(as.Date("31-12-2014", format="%d-%m-%Y") - as.Date(restaurant_train$Open.Date, format= "%m/%d/%Y"))

#filter unwanted features
restaurant_train1 = restaurant_train[,-c(1,2,3,43)]
dim(restaurant_train1)

