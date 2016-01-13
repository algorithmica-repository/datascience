library(caret)
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")

winedata = read.csv("wine.txt", header = TRUE)
dim(winedata)
str(winedata)

#Remove the variables which have 95% NAs
threshold_val = 0.95 * dim(winedata)[1]
include_cols = !apply(winedata, 2, function(y) sum(is.na(y)) > threshold_val)
winedata = winedata[, include_cols]

#Find the variables which have very less variance
nearZvar = nearZeroVar(winedata, saveMetrics = TRUE)
winedata = winedata[nearZvar$nzv==FALSE]

cor(winedata)
#Find the variables which are highly correlated
corr_matrix = abs(cor(winedata))
diag(corr_matrix) = 0
correlated_col = findCorrelation(corr_matrix, verbose = FALSE , cutoff = .60)
winedata = winedata[-correlated_col]
cor(winedata)
dim(winedata)



