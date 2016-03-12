compute.error = function(target, predictions, weights) {
  total_error = 0
  for(i in 1:length(target)) {
    if(target[i] != predictions[i])
      total_error = total_error + weights[i]
  }
  return(total_error/length(target))
}

compute.weight.updates = function(target, predictions, alpha, weights) {
  new_weights = numeric(length(target))
  for(i in 1:length(target)) {
    if(target[i] != predictions[i])
      new_weights[i] = weights[i] * exp(alpha)
    else
      new_weights[i] = weights[i] * exp(-1 * alpha)
  }
  return(new_weights/sum(new_weights))
}

boosted.tree.learn = function(train.set, target, iterations, weights=rep(1/nrow(train.set),nrow(train.set)), maxdepth=30) {
  predictors = paste(names(train.set),collapse="+")
  form = as.formula(paste(target, " ~ ", predictors,"-", target))
  
  ensemble = list()
  alpha = numeric(iterations)
  for(i in 1:iterations) {
    tree = rpart(form, train.set,
                weights=weights, 
                control=rpart.control(xval=1,maxdepth=maxdepth))
    predictions  = predict(tree, train.set,  type="class")
    error = compute.error(train.set[,1], predictions, weights)
    if(error >= 0.5) break;
    alpha[i] =  1/2 * log((1-error)/error)
    weights = compute.weight.updates(train.set[,1], predictions, alpha[i], weights)
    ensemble = c(ensemble, list(tree))
  }
  return(list(ensemble, alpha))
}

boosted.tree.predict = function(model, test.set) {
  ensemble = model[[1]]
  mat = matrix(NA,nrow(test.set),length(ensemble))
  for(i in 1:length(ensemble)) {
    test.pred  = predict(ensemble[[i]], test.set,  type="class")
    mat[,i]=test.pred
  }
  
  alpha = model[[2]]
  outcome = numeric(nrow(test.set))
  for(i in 1:nrow(mat)) {
    weighted.sum = 0
    for(j in 1:ncol(mat)) {
      weighted.sum = weighted.sum + alpha[j] * ifelse(mat[i,j]==1,-1,1) 
    }
    outcome[i] = ifelse(weighted.sum < 0,1,2)
  }
  mat = cbind(mat,outcome)
  
  test.accuracy = sum(outcome == as.numeric(test.set[,"target"])) / nrow(test.set)
  list(mat, test.accuracy)
}

set.seed(1234)
setwd("C:\\boostbag")

letters = read.table("letterCG.data",sep=" ",header = TRUE )
dim(letters)
str(letters)
letters$target = as.factor(letters$target)
head(letters)

train.set = letters[1:500,]
test.set  = letters[-(1:500),]

model = boosted.tree.learn(train.set,"target",25)
model
out = boosted.tree.predict(model, test.set)
out