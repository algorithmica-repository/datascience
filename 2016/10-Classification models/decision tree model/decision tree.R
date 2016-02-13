tree.learn = function(train.set, target, weights=rep(1,nrow(train.set)), maxdepth=30) {
  predictors = paste(names(train.set),collapse="+")
  form = as.formula(paste(target, " ~ ", predictors,"-", target))
  tree = rpart(form, train.set,
                weights=weights, 
                control=rpart.control(xval=1,maxdepth=maxdepth))
  return(tree)
}

tree.predict = function(model, test.set) {
  test.pred  = predict(model, test.set,  type="class")
  test.accuracy = sum(test.pred == test.set[,"target"]) / nrow(test.set)
  list(test.pred, test.accuracy)
}

set.seed(1234)

setwd("C:\\boostbag")
letters = read.table("letterCG.data",sep=" ",header = TRUE )
dim(letters)
str(letters)
train.set = letters[1:500,]
test.set  = letters[-(1:500),]

model = tree.learn(train.set,"target")
model
out = tree.predict(model, test.set)
out
