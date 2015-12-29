
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(titanic)
str(titanic)
summary(titanic)

for (i in 1:length(output$Name)){
  matches = regexpr(" (\\w+)\\.", output$Name[i], perl=TRUE, ignore.case=TRUE);
  result  = attr(matches, "capture.start")[,1]
  attr(result, "match.length") = attr(matches, "capture.length")[,1]
  output$title[i] = regmatches(output$Name[i], result)
}
# consolidate the titles
# ----------------------
if (sum((output$title=='Dr') & (output$Sex=='female')) > 0) {
  output$title[(output$title=='Dr') & (output$Sex=='female')] = 'Mrs'
}
if (sum((output$title=="Dr") & (output$Sex=="male")) > 0) {
  output$title[(output$title=="Dr") & (output$Sex=="male")] = 'Mr'
}

output$title[output$title %in% c('Capt','Col','Don','Jonkheer','Major','Rev','Sir')] <- 'Mr'
output$title[output$title %in% c('Countess','Dona','Lady','Mme')] <- 'Mrs'
output$title[output$title %in% c('Mlle','Ms')] <- 'Mrs'

# Replace missing ages with the average for the title
# ###################################################

for (i in 1:length(output$Age)){
  if (is.na(output$Age[i])) {
    output$Age[i] = mean(output$Age[output$title==output$title[1]],na.rm=TRUE)
  }
}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(titanic)
str(titanic)
summary(titanic)

preObj = preProcess(titanic, method=c("medianImpute"))
predict(preObj, titanic)

titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Name = as.character(titanic$Name)
dummies = dummyVars(~ Pclass + Embarked, data = titanic)
titanic_new = predict(dummies, newdata = titanic)

preObj = preProcess(titanic, method=c("knnImpute"))
predict(preObj, titanic)

dv=dummyVars(~v3,data = df)
df_dv = as.data.frame(predict(dv,newdata=df))
df_final = data.frame(df[1:2],df_dv)

featr_nzv=nearZeroVar(df_final, saveMetrics = TRUE)
df_final = df_final[featr_nzv$nzv == FALSE]
str(df_final)
