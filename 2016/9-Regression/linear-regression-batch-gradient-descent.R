# y = mx + b
# m is slope, b is y-intercept
error = function(b,m,points) {
  totalError = 0
  for(i in 1:nrow(points)) {
    x = points[i,1]
    y = points[i,2]
    totalError = totalError + (y - (m * x + b)) ^ 2
  }
  totalError / nrow(points)
}

gradient = function(points, b_current, m_current) {
  b_gradient = 0
  m_gradient = 0
  N = nrow(points)
  for (i in 1:nrow(points)) {
    x = points[i, 1]
    y = points[i, 2]
    b_gradient = b_gradient - (2/N) * (y - ((m_current * x) + b_current))
    m_gradient = m_gradient - (2/N) * x * (y - ((m_current * x) + b_current))
  }
  c(b_gradient, m_gradient)
}

gradient_descent = function(points, learningRate, iterations) {
  b_current = 0
  m_current = 0
  b_trace = 0
  m_trace = 0
  error_trace = error(b_current, m_current, points)
  for(iter in 1:iterations) {
    grad = gradient(points, b_current, m_current)
    b_current = b_current - (learningRate * grad[1])
    m_current = m_current - (learningRate * grad[2])
    
    error_current = error(b_current, m_current, points)
    b_trace = c(b_trace, b_current)
    m_trace = c(m_trace, m_current)
    error_trace = c(error_trace,error_current)
  }
  data.frame(intercept = b_trace, slope = m_trace, error = error_trace)
}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2016\\9-Regression")
data = read.table("regression1.tsv",sep="\t")
names(data) = c("x","y")
str(data)
dim(data)

plot(data$x,data$y,xlab="x",ylab="y",main="Linear Regression")
solution = gradient_descent(data, 0.0001, 1000)
solution[1:10,]
plot(1:1001,solution$error,col="blue",xlab="Iterations",ylab="error",main="Convergence")


