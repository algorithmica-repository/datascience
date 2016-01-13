x1 = c(10,2,8,9,12)
x2 = c(20,5,17,20,22)
x3 = c(10,2,7,10,11)
x4 = c(25,20,18,7,5)
data = data.frame(x1,x2,x3,x4)

dim(data)
str(data)
head(data)

#check the assumption of PCA
cor(data)
cov(data)
sum(diag(cov(data)))

#PCA computed using Covariance/correlation matrix
pca = princomp(data, cor=F)
names(pca)
summary(pca)
plot(pca, type="lines")
pca$loadings
pca$scores
sum(diag(cov(pca$scores)))

# pca using caret package
library(caret)
preObj = preProcess(data, method=c("pca"), thresh = 1.0)
preObj$rotation
newdata = predict(preObj,data)


#PCA computed using SVD
pca = prcomp(data, scale.=TRUE)
names(pca)
pca
summary(pca)
pca$rotation
pca$x
plot(pca, type="lines")

