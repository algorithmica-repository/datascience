library(plot3D)

sigmoid = function(z) {
  g = 1/(1+exp(-z))
  return(g)
}

logError = function(W,X,Y) {
  totalError = 0
  epsilon = 1e-15
  for(i in 1:nrow(X)) {
    sigma = sigmoid(W %*% X[i,])
    sigma_corrected = min(max(epsilon, sigma), 1-epsilon)
    
    err = Y[i]*log(sigma_corrected) + (1-Y[i])*log(1 - sigma_corrected)
    totalError = totalError + err
  }
  return(-1 * totalError/nrow(X))
}

batch_gradient = function(W,X,Y) {
  delta = rep(0,ncol(X))
  for(i in 1:nrow(X)) {
    sigma = sigmoid(W %*% X[i,])
    delta = delta  + (sigma - Y[i])* X[i,]
  }
  return(delta/nrow(X))
}

batch_gradient_descent = function(X,Y, learningRate, iterations) {
  W = rep(0,ncol(X))
  W_trace = W  
  error_trace = logError(W,X,Y)
  for(iter in 1:iterations) {
    grad = batch_gradient(W,X,Y)
    W = W - (learningRate * grad)
    error_current = logError(W, X, Y)
    
    W_trace = rbind(W_trace, W)
    error_trace = c(error_trace,error_current)
  }
  list(W_trace,error_trace)
}

display.trace = function(solution) {
  solution
}

plot.data = function(X,Y) {
  plot(X[,2],X[,3],col=ifelse(Y == 1,"yellow","blue"),xlab="v1",ylab="v2",main="Logistic Regression")
}

plot.lines = function(solution) {
  for (i in 2:iterations) {
    W = solution[[1]][i,]
    abline(-W[1]/W[3], -W[2]/W[3], col=rgb(0.9,0,0,0.4))
  }
  abline(-W[1]/W[3], -W[2]/W[3], col="green", lwd=4)
}

plot.convergence = function(iterations, solution) {
  plot(1:(iterations+1),solution[[2]],col="blue",xlab="Iterations",ylab="error",main="Convergence")
}

plot.trace = function(X, Y, solution, iterations) {
  windows(width=50, height=60)
  plot.data(X,Y)
  plot.lines(solution)
  x11()
  plot.convergence(iterations,solution)
}

library(caret)
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")
data = read.csv("logistic-train2.csv")
dim(data)
str(data)
data$target = ifelse(data$target==-1,0,1)

#names(data) = c("v1","v2","label")
#preObj = preProcess(data[,1:2],method=c("center","scale"))
#data_n = predict(preObj,data[,1:2])

X = data.frame(v0=rep(1,nrow(data)),data$v1,data$v2)
#X = data.frame(data$v1,data$v2)
Y = data$target
X = as.matrix(X)

stepsize = 0.01
iterations = 1000
solution = batch_gradient_descent(X, Y, stepsize, iterations)
display.trace(solution)
plot.trace(X,Y,solution,iterations)

