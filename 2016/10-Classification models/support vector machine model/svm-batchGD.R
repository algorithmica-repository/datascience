computeError = function(W,X,Y,C) {
  term1 = sum(W^2)/(nrow(X) * C)
  term2 = 0
  for(i in 1:nrow(X)) {
    tmp = 1-Y[i]*(W %*% X[i,])
    if(tmp > 0)
      term2 = term2 + tmp
  }
  return(term1 + term2/nrow(X))
}

batch_gradient = function(W,X,Y,C) {
  term2_delta = rep(0,ncol(X))
  term1_delta = (2*sqrt(sum(W^2)))/(nrow(X) * C)
  for(i in 1:nrow(X)) {
     if((1-Y[i]*(W %*% X[i,]))>0)
       term2_delta = term2_delta - Y[i] * X[i,] 
  }
  return(term1_delta + term2_delta / nrow(X))
}

batch_gradient_descent = function(X,Y, learningRate, iterations, C) {
  W = rep(0,ncol(X))
  W_trace = W  
  error_trace = computeError(W,X,Y,C)
  for(iter in 1:iterations) {
    grad = batch_gradient(W,X,Y,C)
    W = W - (learningRate * grad)
    error_current = computeError(W,X,Y,C)
    
    W_trace = rbind(W_trace, W)
    error_trace = c(error_trace,error_current)
  }
  list(w=W_trace,e=error_trace)
}

display.trace = function(solution) {
  solution
}

plot.data = function(X,Y) {
  plot(X[,2],X[,3],col=ifelse(Y == -1,"yellow","blue"),xlab="v1",ylab="v2",main="SVM")
}

plot.lines = function(iterations,solution) {
  for (i in 2:iterations) {
    W = solution[[1]][i,]
    abline(-W[1]/W[3], -W[2]/W[3], col=rgb(0.9,0,0,0.4))
  }
  abline(-W[1]/W[3], -W[2]/W[3], col="green", lwd=4)
}

plot.convergence = function(iterations, solution) {
  plot(1:(iterations+1),solution[[2]],col="blue",xlab="Iterations",ylab="error",main="Convergence")
}

plot.trace = function(X, Y, solution, iterations,C) {
  windows(width=50, height=60)
  plot.data(X,Y)
  plot.lines(iterations,solution)
  x11()
  plot.convergence(iterations,solution)
}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")
data = read.csv("iris-train.csv", header = TRUE)
dim(data)
str(data)

#data = data.frame(v1=c(1,3,2,5),v2=c(2,5,1,3),target=c(1,1,-1,-1))
X = data.frame(v0=rep(1,nrow(data)),data$v1,data$v2)
Y = data$target
X = as.matrix(X)

stepsize = 0.01
iterations = 30
C = 0.001
solution = batch_gradient_descent(X, Y, stepsize, iterations, C)
display.trace(solution)
plot.trace(X,Y,solution,iterations,C)
