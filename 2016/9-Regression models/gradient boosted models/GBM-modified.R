library(rpart)
library(ggplot2)

plotfunc = function(data,out,num){
  pre = out$predict_y[[num]]
  mse = out$mse[[num]]
  plotdf = data.frame(x=data$x, y=data$y, pre = pre)
  
  p = ggplot(data=plotdf,aes(x=x))
  p = p + geom_point(aes(y=y, color='1'), alpha=0.5)
  p = p + geom_point(aes(y=pre, color='2'))
  p = p +  scale_colour_manual(name="",values = c("1"="black", "2"="red"))
  p = p + xlab(paste0('mse=',mse))
  plot(p)
}

gradient.boosted.tree.learn = function(train.set, iterations, rate=1, max_depth=1) {
  y = train.set[,2]
  x = train.set[,-2]
  dy = y
  ensemble = list() 
  for (i in 1:iterations) {
    # use a weak model to improve
    ensemble[[i]] = rpart(dy~x, control=rpart.control(maxdepth=max_depth,cp=0))
    # modify residuals for next iteration
    dy = dy - rate*predict(ensemble[[i]],as.data.frame(x))
  }
  return(ensemble)
}

gradient.boosted.tree.predict = function(ensemble, test.set, rate=1) {
  y = test.set[,2]
  x = test.set[,-2]
  predict_y = list() 
  predict_y[[1]] = rep(0,nrow(test.set))
  for (i in 2:length(ensemble)) {
    predict_y[[i]] = predict_y[[i-1]] + rate * predict(ensemble[[i-1]],as.data.frame(x))
  }
  # mean sqare error
  mse = sapply(predict_y,function(pre_y) round(mean((y-pre_y)^2),3))
  result = list('predict_y'=predict_y, 'mse'= mse)
  return(result)
}

set.seed(1234)
setwd("C:\\boostbag")
train.set = read.csv("sine_train.csv")
test.set = read.csv("sine_test.csv")
plot(train.set[,1],train.set[,2])
plot(test.set[,1],test.set[,2])

ensemble = gradient.boosted.tree.learn(train.set,350)
ensemble
out = gradient.boosted.tree.predict(ensemble, test.set)
plot(out$mse,type='l')
for(i in seq(2,350,100)) {
  plotfunc(test.set,out,i)
}