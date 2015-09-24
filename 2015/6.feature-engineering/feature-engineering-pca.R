setwd("E:/data analytics/datasets")

data = read.table("protein.txt", header=TRUE, sep="\t")

dim(data)
str(data)
head(data)

# For PCA analysis, keep all the variables
# except the first column with country names:
data = data[, -1]

summary(data)
cor(data)
cov(data)

//PCA computed using Covariance/correlation matrix
pca = princomp(data, cor=T)
names(pca)
pca
summary(pca)
plot(pca, type="lines")
pca$loadings
pca$scores

plot(pca$scores[,1]) 
barplot(pca$scores[,1])

//PCA computed using SVD
pca = prcomp(data[,-1], scale.=TRUE)
names(pca)
pca
summary(pca)
pca$rotation
pca$x
plot(pca, type="lines")


library(caret)
pca = preProcess(data[,-1], method=c("pca"))
str(pca)
names(pca)
pca
pca$rotation
predict(pca,data[,-1])
