library(caret)

# Set seed for reproducibility
set.seed(1234)

# Read the data
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")
tennis = read.csv("playtennis3.csv")
# Explore the dataset
tennis
dim(tennis)
str(tennis)
head(tennis)


#Train the model
ctrl = trainControl(method="LOOCV")
tennis_model = train(tennis, tennis$playtennis, method="nb", trControl=ctrl)
tennis_model$finalModel$tables
tennis_model$finalModel
tennis_model


#Test the model
test1 = data.frame("sunny","26","high","strong")
tennis_predict = predict(tennis_model, test1, type="prob")
tennis_predict
