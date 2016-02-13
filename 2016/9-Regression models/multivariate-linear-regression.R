error = function(W,X,Y) {
  residual = Y - (t(W) %*% X)
  totalError = t(residual) %*% residual
  totalError / nrow(points)
}

gradient = function(W,X,Y) {
  tmp = Y - (t(W) %*% X)
  -2 * t(X) %*% tmp
}

gradient_descent = function(X,Y, learningRate, iterations) {
  W = rep(0,nrow(X))
  W_trace = W  
  error_trace = error(W,X,Y)
  for(iter in 1:iterations) {
    grad = gradient(W,X,Y)
    W = W - (learningRate * grad)
    error_current = error(W, X, Y)
    
    W_trace = rbind(W_trace, W)
    error_trace = rbind(error_trace,error_current)
  }
  list(W_trace,error_trace)
}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")
data = read.csv("housing.csv")
str(data)
dim(data)
housing = data[,c("SqFt","Bedrooms","Bathrooms","Price")]


solution = gradient_descent(a,housing$Price, 0.0001, 1000)
solution[1:10,]
plot(1:1001,solution$error,col="blue",xlab="Iterations",ylab="error",main="Convergence")


