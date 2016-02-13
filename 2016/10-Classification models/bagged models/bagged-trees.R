bootstrap_train = function(train.set) {
  s = sample(1:nrow(train.set),nrow(train.set),replace = T)
  return(train.set[unique(s),])
}

bagged.tree.learn = function(train.set, target, numTrees, weights=rep(1,nrow(train.set)), maxdepth=30) {
  predictors = paste(names(train.set),collapse="+")
  form = as.formula(paste(target, " ~ ", predictors,"-", target))
  
  forest = list()
  for(i in 1:numTrees) {
    train_b = bootstrap_train(train.set)
    tree = rpart(form, train_b,
                weights=weights[1:nrow(train_b)], 
                control=rpart.control(xval=1,maxdepth=maxdepth))
    forest = c(forest, list(tree))
  }
  return(forest)
}

bagged.tree.predict = function(model, test.set) {
  mat = matrix(NA,nrow(test.set),length(model))
  for(i in 1:length(model)) {
    test.pred  = predict(model[[i]], test.set,  type="class")
    mat[,i]=test.pred
  }
  outcome = numeric(nrow(test.set))
  for(i in 1:nrow(mat)) {
    df = as.data.frame(table(mat[i,]))
    outcome[i] = as.numeric(as.character(df[which.max(df$Freq),1]))
  }
  mat = cbind(mat,outcome)
  test.accuracy = sum(mat[,length(model)+1] == as.numeric(test.set[,"target"])) / nrow(test.set)
  list(mat, test.accuracy)
}

set.seed(1234)
setwd("C:\\boostbag")

letters = read.table("letterCG.data",sep=" ",header = TRUE )
dim(letters)
str(letters)
train.set = letters[1:500,]
test.set  = letters[-(1:500),]

head(test.set)

model = bagged.tree.learn(train.set,"target",51)
model
out = bagged.tree.predict(model, test.set)
out