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

pca = princomp(data, cor=T)
names(pca)
pca
summary(pca)
plot(pca, type="lines")

loadings(pca)
pca$scores

plot(pca$scores[,1]) 
barplot(pca$scores[,1])
