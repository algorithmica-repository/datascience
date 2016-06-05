library(caret)
library(missForest)

x = c(169,126,132,160,105,116,125,112,133,94,109,109,106,176,128,131,130,145,136,146,111,97,134,153,118,137,101,103,78,151)
y = c(148,123,149,169,138,102,88,100,150,113,96,78,148,137,155,131,101,155,140,134,129,85,124,112,118,122,119,106,74,113)
true_data = data.frame(x = x, y = y)
ggplot(true_data, aes(x=x,y=y) ) + geom_point() + scale_x_continuous(name="x", limits=c(75, 200) ) + scale_y_continuous(name="y", limits=c(75, 200) )

mar_x = c(169,160,176,145,146,153,151)
mar_y = c(148,169,137,155,134,112,113)
mar_data = data.frame(x = mar_x, y = mar_y)
ggplot(mar_data, aes(x=mar_x,y=mar_y) ) + geom_point() + scale_x_continuous(name="x", limits=c(75, 200) ) + scale_y_continuous(name="y", limits=c(75, 200) )

missing_x = c(169,126,132,160,105,116,125,112,133,94,109,109,106,176,128,131,130,145,136,146,111,97,134,153,118,137,101,103,78,151)
missing_y = c(148,NA,NA,169,NA,NA,NA,NA,NA,NA,NA,NA,NA,137,NA,NA,NA,155,NA,134,NA,NA,NA,112,NA,NA,NA,NA,NA,113)
missing_data = data.frame(x = missing_x, y = missing_y)
preObj = preProcess(missing_data, method = c("bagImpute"))
newdata = predict(preObj, missing_data)


#ggplot(true_data, aes(x=x,y=y) ) + geom_point() + xlim(75, 200) + ylim(75,200)

set.seed(100)
tr_ctrl = trainControl(method="boot")

model1 = train(Survived ~ FamilySize + Title + Sex + Pclass + Embarked + Fare, data = titanic_train, method='rf', trControl = tr_ctrl, ntree = 500)

model2 = train(x = titanic_train[,c("FamilySize", "Title", "Sex", "Pclass", "Embarked", "Fare")], y = titanic_train[,"Survived"], data = titanic_train, method='rf', trControl = tr_ctrl, ntree = 500)
