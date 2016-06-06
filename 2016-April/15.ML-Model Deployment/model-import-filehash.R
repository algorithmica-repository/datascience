library(caret)
library(filehash)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets\\restaurant-revenue")

restaurant_test = read.csv("test.csv", na.strings=c("","NA"))
dim(restaurant_test)
str(restaurant_test)

db.models = dbInit("models.db")
env.models = db2env(db.models)
print(ls(env.models))

restaurant_test1 = predict(env.models$imputeObj,restaurant_test[,-1])
dim(restaurant_test1)
str(restaurant_test1)

restaurant_test2 = restaurant_test1[, env.models$var_obj$zeroVar==FALSE]
dim(restaurant_test2)
str(restaurant_test2)

features_new_df = env.models$add.features(restaurant_test2$Open.Date)
restaurant_test3 = cbind(restaurant_test2, features_new_df)
dim(restaurant_test3)
str(restaurant_test3)

features.exclude = c(1,2)
revenue = predict(env.models$model_rf,restaurant_test3[,-features.exclude])
result = data.frame(Id = restaurant_test[,"Id"], Prediction = revenue)

str(result)


