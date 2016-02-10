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
  list(w=W_trace,e=error_trace)
}

display.trace = function(solution) {
  solution
}

plot.data = function(X,Y) {
  plot(X[,1],X[,2],col=ifelse(Y == -1,"yellow","blue"),xlab="v1",ylab="v2",main="Perceptron")
}

plot.lines = function(solution) {
  for (i in 2:iterations) {
    W = solution[[1]][i,]
    abline(0, -W[1]/W[2], col=rgb(0.9,0,0,0.4))
  }
  abline(0, -W[1]/W[2], col="green", lwd=4)
}

plot.error = function(solution) {
  w1 = seq(-20, 10, 1)
  w2 = seq(-20, 5, 1) 
  error = matrix(rep(0,length(w1)*length(w2)),length(w1),length(w2))
  for(i in 1:length(w1)) {
    for(j in 1:length(w2)) {
      error[i,j] = hingeError(c(w1[i],w2[j]),X,Y)[2]
    }
  }
  persp3D(w1, w2, error, xlab="w1", ylab="w2", zlab="error",col = ramp.col(n = 50, col = c("#FF033E", "#FFBF00", "#FF7E00", "#08E8DE", "#00FFFF", "#03C03C"), alpha = .1), border = "#808080", theta = 10, phi = 20, expand = 0.9, colkey = FALSE, ticktype="detailed")
  points3D(solution[[1]][,1],solution[[1]][,2] , solution[[2]][,2], pch = 20, col = 'red', add = TRUE)
}

plot.convergence = function(iterations, solution) {
  plot(1:(iterations+1),solution[[2]][,1],col="blue",xlab="Iterations",ylab="error",main="Convergence")
}

plot.trace = function(X, Y, solution, iterations) {
  windows(width=50, height=60)
  plot.data(X,Y)
  plot.lines(solution)
  x11()
  plot.error(solution)
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

stepsize = 1
iterations = 3000
solution = batch_gradient_descent(X, Y, stepsize, iterations)
solution[[1]]
display.trace(solution)
plot.trace(X,Y,solution,iterations)
