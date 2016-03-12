library(plot3D)
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
  plot(X[,1],X[,2],col=ifelse(Y == -1,"yellow","blue"),xlab="v1",ylab="v2",main="SVM")
}

plot.lines = function(solution) {
  for (i in 2:iterations) {
    W = solution[[1]][i,]
    abline(0, -W[1]/W[2], col=rgb(0.9,0,0,0.4))
  }
  abline(0, -W[1]/W[2], col="green", lwd=4)
}

plot.error = function(solution,X,Y,C) {
  w1 = seq(-20, 10, 1)
  w2 = seq(-20, 5, 1) 
  error = matrix(rep(0,length(w1)*length(w2)),length(w1),length(w2))
  for(i in 1:length(w1)) {
    for(j in 1:length(w2)) {
      error[i,j] = computeError(c(w1[i],w2[j]),X,Y,C)
    }
  }
  persp3D(w1, w2, error, xlab="w1", ylab="w2", zlab="error",col = ramp.col(n = 50, col = c("#FF033E", "#FFBF00", "#FF7E00", "#08E8DE", "#00FFFF", "#03C03C"), alpha = .1), border = "#808080", theta = 10, phi = 20, expand = 0.9, colkey = FALSE, ticktype="detailed")
  points3D(solution[[1]][,1],solution[[1]][,2] , solution[[2]], pch = 20, col = 'red', add = TRUE)
}

plot.convergence = function(iterations, solution) {
  plot(1:(iterations+1),solution[[2]],col="blue",xlab="Iterations",ylab="error",main="Convergence")
}

plot.trace = function(X, Y, solution, iterations,C) {
  windows(width=50, height=60)
  plot.data(X,Y)
  plot.lines(solution)
  x11()
  plot.error(solution,X,Y,C)
  x11()
  plot.convergence(iterations,solution)
}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")
data = read.csv("iris-train.csv", header = TRUE)
dim(data)
str(data)

X = data.frame(data$v1,data$v2)
Y = data$target
X = as.matrix(X)

stepsize = 0.01
iterations = 2000
C = 50
solution = batch_gradient_descent(X, Y, stepsize, iterations, C)
display.trace(solution)
plot.trace(X,Y,solution,iterations,C)
