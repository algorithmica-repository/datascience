# y = mx + b
# m is slope, b is y-intercept
library(plot3D)
error = function(b,m,data) {
  totalError = 0
  for(i in 1:nrow(data)) {
    x = data[i,1]
    y = data[i,2]
    totalError = totalError + (y - (m * x + b)) ^ 2
  }
  totalError / nrow(data)
}

stochastic_batch_gradient = function(data, b_current, m_current) {
  N = nrow(data)
  
  b_gradient = 0
  m_gradient = 0
  #Lets take a random batch of size 10
  random_batch_ind = sample(1:N,10)
  for(i in 1:length(random_batch_ind)) {
    x = data[random_batch_ind[i], 1]
    y = data[random_batch_ind[i], 2]
    b_gradient = b_gradient - 2 * (y - ((m_current * x) + b_current))
    m_gradient = m_gradient - 2 * x * (y - ((m_current * x) + b_current))
  }
  c(b_gradient/N, m_gradient/N)
}

stochastic_batch_gradient_descent = function(data, learningRate, iterations) {
  b_current = 0
  m_current = 0
  b_trace = b_current
  m_trace = m_current
  error_trace = error(b_current, m_current, data)
  for(iter in 1:iterations) {
    grad = stochastic_batch_gradient(data, b_current, m_current)
    b_current = b_current - (learningRate * grad[1])
    m_current = m_current - (learningRate * grad[2])
    error_current = error(b_current, m_current, data)
    
    b_trace = c(b_trace, b_current)
    m_trace = c(m_trace, m_current)
    error_trace = c(error_trace,error_current)
  }
  data.frame(intercept = b_trace, slope = m_trace, error = error_trace)
}

display.trace = function(solution) {
  solution
}

plot.trace = function(data, solution, iterations) {
  plot(data$x,data$y,xlab="x",ylab="y",main="Linear Regression")
  for (i in 1:iterations) {
    abline(solution$intercept[i], solution$slope[i],col=rgb(0.9,0,0,0.4))
  }
  abline(solution$intercept[i], solution$slope[i], col="green", lwd=4)
  
  b = seq(-2, 4, 0.5)
  m = seq(-2, 4, 0.5) 
  err = outer(b,m,error,data)
  
  persp3D(b, m, err, xlab="b", ylab="m", zlab="error",col = ramp.col(n = 50, col = c("#FF033E", "#FFBF00", "#FF7E00", "#08E8DE", "#00FFFF", "#03C03C"), alpha = .1), border = "#808080", theta = -90, phi = 10, colkey = FALSE)
  contour3D(b, m, z = 0, colvar = err, col = c("#FF7E00", "#FF033E"), alpha = .3, add = TRUE, colkey = FALSE)
  points3D(solution$intercept,solution$slope , solution$error, pch = 20, col = 'red', add = TRUE)
  
  plot(1:(iterations+1),solution$error,col="blue",xlab="Iterations",ylab="error",main="Convergence")
}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets")
data = read.table("regression1.tsv",sep="\t")
names(data) = c("x","y")
str(data)
dim(data)

iterations = 1200
stepsize = 0.001
set.seed(100)
solution = stochastic_batch_gradient_descent(data, stepsize, iterations)
display.trace(solution)
plot.trace(data,solution,iterations)
