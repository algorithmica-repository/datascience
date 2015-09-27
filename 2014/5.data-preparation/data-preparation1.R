library(caret)

# Age vector
age = c(25, 35, 50, 40, 60, 25, 35, 50, 80, 100)

# Salary vector
salary = c(200000, 120000, 100000, 300000, 250000, 200000, 100000, 500000, 300000, 350000)

# Data frame created using age and salary
df = data.frame( age, salary)
dim(df)
str(df)
summary(df)

min_max_normalize = function(x,new_min=0, new_max=1) {
  new_min + (x - min(x)) * (new_max - new_min) / (max(x) - min(x));
}

z_score_normalize = function(x) {
  (x - mean(x)) / sd(x)
}

hist(df$age, col="lightblue")

//min-max normalization using custom function
df1 = as.data.frame(apply(df, 2, min_max_normalize))
hist(df1$age, col="lightblue")

//zero-one normalization using preProcess
preObj = preProcess(df, method=c("range"))
preObj$ranges
df2 = predict(preObj,df)
hist(df2$age, col="lightblue")

//z-score normalization using custom function
df3 = as.data.frame(apply(df, 2, z_score_normalize))
hist(df3$age, col="lightblue")

//z-score normalization using preProcess
preObj = preProcess(df, method=c("center","scale"))
preObj$mean
preObj$std
df4 = predict(preObj,df)
hist(df4$age, col="lightblue")
