setwd("E:/data analytics/datasets")

winedata = read.csv("wine.data", header = TRUE)
dim(winedata)
str(winedata)
head(winedata)

cov_matrix = cov(winedata)
cor_matrix = cor(winedata)