setwd("E:/data analytics/datasets")

winedata = read.csv("wine.data", header = TRUE)
dim(winedata)
str(winedata)
head(winedata)

pairs(~X2.8+X3.06,data=winedata)
cov_matrix = cov(winedata)
cor_matrix = cor(winedata)