library(caret)
library(car)

v1 = c("male","female","male","female")
v1_f = factor(v1)

v2 = c("1","2","3","1")
v2_f = factor(v2)
v2_f=recode(v2_f,"'1' = 'a'; '2'= 'b'; '3'= 'c' ")

v3 = c("low","medium","high","low")
v3_f = factor(v3,ordered = TRUE)
v3_f = factor(v3,levels = c("low","medium","high"),ordered = TRUE)

df = data.frame(v1_f,v2_f,v3_f)
names(df) = c("gender","category","rating")

dummyObj = dummyVars(~gender + category + rating,df,fullRank = FALSE)
predict(dummyObj,df)

dummyObj = dummyVars(~gender + category + rating,df,fullRank = TRUE)
predict(dummyObj,df)
