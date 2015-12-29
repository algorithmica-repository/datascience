library(caret)
library(RANN)
library(mice)

set.seed(100)
df = data.frame(sample(1:100,20), sample(1:100,20), sample(1:100,20))

for (i in 1:20) {
  if (i %% 3 == 0) df[i,1] = NA; 
  if (i %% 5 == 0) df[i,2] = NA;
  if (i %% 10 ==0) df[i,3] = NA;
}
#median based imputation
names(df) = c('v1', 'v2', 'v3')
preObj1 = preProcess(df, method = c("medianImpute"))
df1=predict(preObj1,df)
df1

#knn based imputation
preObj2 = preProcess(df, method = c("knnImpute") , k = 1)
df2=predict(preObj2,df)
df2

preObj3 = preProcess(df, method = c("center","scale"))
df3=predict(preObj3,df)
df3

#bagged trees based imputation
preObj4 = preProcess(df, method = c("center", "scale","bagImpute"))
df4=predict(preObj4,df)
df4

#multiple imputation
md.pattern(df)
imputed = mice(df)
complete(imputed)