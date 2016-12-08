library(caret)
library(corrplot)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_train)
str(restaurant_train)
restaurant_train1 = restaurant_train[,-1]

# picking only numerical attributes for correlation matrix
numeric_attr = sapply(restaurant_train1, is.numeric)
correlations = cor(restaurant_train1[,numeric_attr])
#plotting correlation matrix
X11()
corrplot(correlations)
corrplot(correlations, order = "hclust")
corrplot(correlations, order = "hclust", addrect=3)

# finding highly correlated featues using correlation matrix
filtered_features_correlation = findCorrelation(abs(correlations), cutoff = 0.95)
restaurant_train1 = restaurant_train[,-filtered_features_correlation]

