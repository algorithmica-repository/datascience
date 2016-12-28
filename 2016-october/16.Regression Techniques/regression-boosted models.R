library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets\\restaurant-revenue")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_train)
str(restaurant_train)
restaurant_train1 = restaurant_train[,-c(1,2,3,5)]

set.seed(100)

boosting_strategy = trainControl(method="cv", number=10)
gbm_grid = expand.grid(interaction.depth = 3:5,
                       n.trees = seq(400, 500, by = 50),
                       shrinkage = c(0.1,0.3,1),
                       n.minobsinnode = 5)
gbm_model = train(revenue ~ ., restaurant_train1, method="gbm", trControl = boosting_strategy, tuneGrid = gbm_grid)
gbm_model$finalModel
plot(gbm_model)
plot(varImp(gbm_model))

xgb_linear_grid = expand.grid(nrounds = seq(400, 500, by = 50),
                       #eta = c(0.001,0.003,0.01,0.03),
                       lambda = c(0.1,0.2),
                       alpha = c(0.1,0.2) )
xgb_linear_model = train(revenue ~ ., restaurant_train1, method="xgbLinear", trControl = boosting_strategy, tuneGrid = xgb_linear_grid)
plot(xgb_linear_model)
plot(varImp(xgb_linear_model))

xgb_tree_grid = expand.grid(nrounds = seq(400, 500, by = 50), 
                            max_depth = 3:7, 
                            eta = c(0.01, 0.001, 0.0001), 
                            gamma = c(1, 2, 3), 
                            colsample_bytree = c(0.4, 0.7, 1.0), 
                            min_child_weight = c(0.5, 1, 1.5) )
xgb_tree_model = train(revenue ~ ., restaurant_train1, method="xgbTree", trControl = boosting_strategy, tuneGrid = xgb_tree_grid)
plot(xgb_tree_model)
plot(varImp(xgb_tree_model))

