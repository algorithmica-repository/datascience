library(caret)
m=matrix(1:8,4,2)
svd

pca = princomp(m, cor=T)
summary(pca)
pca$loadings
pca$scores

preProcess(method=c("pca"))