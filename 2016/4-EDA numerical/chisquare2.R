# checking the depedance between type of handed-ness and gender

# left handed and right handed persons among male and female
left_handed1 = c(12,7)
right_handed1 = c(108,133)
df1 = data.frame(left_handed1, right_handed1)
names(df1) = c("male","female")
df1
chisq.test(df1)

left_handed2 = c(15,50)
right_handed2 = c(105,90)
df2 = data.frame(left_handed2, right_handed2)
names(df2) = c("male","female")
df2
chisq.test(df2)

# checking the dependance between survived and pclass/embarked factor variables
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic_train = read.table("train.csv", TRUE, ",")
dim(titanic_train)
str(titanic_train)

titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
chisq.test(titanic_train$Survived, titanic_train$Pclass)
chisq.test(titanic_train$Survived, titanic_train$Embarked)
