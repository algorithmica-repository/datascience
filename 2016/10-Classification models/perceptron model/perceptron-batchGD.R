hingeError = function(W,X,Y) {
  misclassf = 0
  totalError = 0
  for(i in 1:nrow(X)) {
    tmp = Y[i]*(W %*% X[i,])
    if(tmp <= 0) 
      misclassf = misclassf + 1
      totalError = totalError - tmp
  }
  c(misclassf,totalError/nrow(X))
}

batch_gradient = function(W,X,Y) {
  delta = rep(0,ncol(X))
  for(i in 1:nrow(X)) {
     if(Y[i]*(W %*% X[i,]) <= 0)
       delta = delta - Y[i]* X[i,]
  }
  delta / nrow(X)
}

batch_gradient_descent = function(X,Y, learningRate, iterations) {
  W = rep(0,ncol(X))
  W_trace = W  
  error_trace = hingeError(W,X,Y)
  for(iter in 1:iterations) {
    grad = batch_gradient(W,X,Y)
    W = W - (learningRate * grad)
    error_current = hingeError(W, X, Y)
    
    W_trace = rbind(W_trace, W)
    error_trace = rbind(error_trace,error_current)
  }
  list(W_trace,error_trace)
}

display.trace = function(solution) {
  solution[[1]]
  solution[[2]]
}

plot.trace = function(X, Y, solution, iterations) {
  plot(X[,2],X[,3],col=ifelse(Y == -1,"yellow","blue"),xlab="v1",ylab="v2",main="Perceptron")
  weights = solution[[1]]
  for (i in 2:iterations) {
    W = weights[i,]
    abline(-W[1]/W[3], -W[2]/W[3], col=rgb(0.9,0,0,0.4))
  }
  abline(-W[1]/W[3], -W[2]/W[3], col="green", lwd=4)
  misclass = solution[[2]]
  plot(1:(iterations+1),misclass[,1],col="blue",xlab="Iterations",ylab="error",main="Convergence")
  
#   w1 = seq(-2, 4, 1)
#   w2 = seq(-2, 4, 1) 
#   err = outer(w1,w2,hingeError,weights,X,Y)
#   
#   persp3D(w1, w2, err[,1], xlab="w1", ylab="w2", zlab="error",col = ramp.col(n = 50, col = c("#FF033E", "#FFBF00", "#FF7E00", "#08E8DE", "#00FFFF", "#03C03C"), alpha = .1), border = "#808080", theta = -90, phi = 10, colkey = FALSE)
#   contour3D(b, m, z = 0, colvar = err, col = c("#FF7E00", "#FF033E"), alpha = .3, add = TRUE, colkey = FALSE)
#   points3D(solution$intercept,solution$slope , solution$error, pch = 20, col = 'red', add = TRUE)
#   

}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")
data = read.csv("iris-train.csv", header = TRUE)
dim(data)
str(data)

X = data.frame(v0=rep(1,nrow(data)),data$v1,data$v2)
Y = data$target
X = as.matrix(X)

stepsize = 1
iterations = 1000
solution = batch_gradient_descent(X, Y, stepsize, iterations)
display.trace(solution)
plot.trace(X,Y,solution,iterations)
