computeError = function(X,y,w1,w2) {
  fout = feed_forward(X,w1,w2)
  output = fout[["z3"]]
  term1 = -y * log(output)
  term2 = (1 - y) * log(1 - output)
  return(sum(term1 - term2))
}

sigmoid = function(z) {
  g = 1/(1+exp(-z))
  return(g)
}

sigmoid_gradient = function(z) {
  sg = sigmoid(z)
  return(sg * (1-sg))
}

feed_forward = function(X,w1,w2) {
  a1 = c(1,as.vector(X))
  z1 = a1
  a2 = z1 %*% t(w1)
  z2 = sigmoid(a2)
  a3 = c(1,z2) %*% t(w2)
  z3 = sigmoid(a3)
  return(list(a1=a1,z1=z1,a2=a2,z2=z2,a3=a3,z3=z3))
}

stochastic_gradient = function(X,y,w1,w2) {
  rind = sample(1:nrow(X),1)
  solution = feed_forward(X[rind,],w1,w2)
  a2 = solution[["a2"]]
  z1 = solution[["z1"]]
  z2 = solution[["z2"]]
  z3 = solution[["z3"]]
  
  delta3 = y[rind] - z3
  tmp = t(w2) %*% as.matrix(delta3)
  delta2 = tmp[-1,] * sigmoid_gradient(a2)
  z2 = c(1,as.vector(z2))
  gradient2 = delta3 * z2
  gradient1 = delta2 * z1 
  return(list(g1=gradient1,g2=gradient2))
}

stochastic_gradient_descent = function(X,y,nfeatures,nhidden,nout,learningRate, iterations) {
  rdata = runif(nhidden*(nfeatures + 1),-1,1)
  w1 = matrix(rdata, nrow=nhidden,ncol=(nfeatures+1),byrow = TRUE)
  
  rdata = runif(nout*(nhidden + 1),-1,1)
  w2 = matrix(rdata, nrow=nout,ncol=(nhidden+1),byrow = TRUE)
  
  w1_trace = list(iterations+1)
  w1_trace[[1]] = w1
  w2_trace = list(iterations+1)
  w2_trace[[1]] = w2
  error_trace = list(iterations+1)
  error_trace[[1]] = computeError(X,y,w1,w2)
  for(iter in 1:iterations) {
    grad = stochastic_gradient(X,y,w1,w2)
    w1 = w1 - (learningRate * grad[[g1]])
    w2 = w2 - (learningRate * grad[[g2]])
    error_current = computeError(X,y,w1,w2)
    
    w1_trace[[iter]] = w1
    w2_trace[[iter]] = w2
    error_trace[[iter]] = error_current
  }
  return(list(w1_trace,w2_trace,error_trace))
}

nfeatures = 2
nobs = 4
nhidden = 2
nout = 1
stepsize = 0.001
iterations = 10

set.seed(100)
X = matrix(NA,nrow=nobs,ncol=nfeatures+1,byrow=TRUE)
X[1,] = c(1,0,0)
X[2,] = c(1,0,1)
X[3,] = c(1,1,0)
X[4,] = c(1,1,1)
y = c(0,1,1,0)
solution = stochastic_gradient_descent(X,y,nfeatures,nhidden,nout,stepsize,iterations)
solution