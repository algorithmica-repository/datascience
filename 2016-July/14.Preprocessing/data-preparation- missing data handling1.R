library(caret)
library(RANN)
library(mice)

set.seed(100)
df = data.frame(sample(1:200,50), sample(1:200,50), sample(1:200,50), sample(1:200,50) )

for (i in 1:50) {
  if (i %% 3 == 0) df[i,1] = NA; 
  if (i %% 5 == 0) df[i,2] = NA;
  if (i %% 10 ==0) df[i,3] = NA;
}
#median based imputation
names(df) = c('v1', 'v2', 'v3','v4')
preObj1 = preProcess(df, method = c("medianImpute"))
df1=predict(preObj1,df)
df1

#knn based imputation
preObj2 = preProcess(df, method = c("knnImpute") , k = 3)
df2=predict(preObj2,df)
df2

#bagged trees based imputation
preObj3 = preProcess(df, method = c("bagImpute"))
df3=predict(preObj3,df)
df3

#multiple imputation
md.pattern(df)
imputed = mice(df)
complete(imputed)
