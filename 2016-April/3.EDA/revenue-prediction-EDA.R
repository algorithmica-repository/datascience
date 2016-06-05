library(ggplot2)
library(rpart)
library(caret)
library(corrplot)
library(reshape2)
library(Amelia)

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

##EDA
#Exploring numerical summaries
summary(restaurant_train)

#Exploring data relationships
X11()
ggplot(restaurant_train) + geom_histogram(aes(x = revenue), fill = "white", colour = "black")

X11()
ggplot(restaurant_train, aes(x=City.Group, y=revenue)) +
  geom_point(shape=1) 

X11()
ggplot(restaurant_train, aes(x=Type, y=revenue)) +
  geom_point(shape=1) 

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
corrplot(correlations)
corrplot(correlations, order = "hclust")
corrplot(correlations, order = "hclust", addrect=3)
corrplot(correlations, method = "circle", type="upper", order = "hclust")
#Exploring missing data
X11()
missmap(restaurant_train)
X11()
d = melt(restaurant_train[,-c(1:5)])

d = melt(restaurant_train[,-c(1:5)], variable_name = "p", value.name="v")
str(d)
ggplot(d,aes(x = value)) + 
  facet_wrap(~p,scales = "free_x") + 
  geom_histogram()