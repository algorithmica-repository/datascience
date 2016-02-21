error = function(R,P,Q,K) {
  ratingError = 0
  for (i in 1:nrow(R)) {
    for (j in 1:ncol(R)) {
      if (R[i,j] > 0) {
        eij = P[i,] %*% Q[,j]
        ratingError = ratingError + (R[i,j] - eij) ^ 2
      }
    }
  }
  return(ratingError)
}

gradient_descent = function(R,P,Q,K, learningRate, iterations) {
  error_trace = error(R,P,Q,K)
  for (i in 1:iterations) {
    for (i in 1:nrow(R)) {
      for (j in 1:ncol(R)) {
        if (R[i,j] > 0) {
          eij = R[i,j] - P[i,] %*% Q[,j]
          for (k in 1:K) {
            P[i,k] = P[i,k] - learningRate * -2 * eij * Q[k,j]
            Q[k,j] = Q[k,j] - learningRate * -2 * eij * P[i,k]
          }
        }
      }
    }
    error_trace = c(error_trace, error(R,P,Q,K))
  }
  return(list(P,Q,error_trace))
}

R = matrix(NA,nrow=5,ncol=4,byrow = TRUE)
R[1,] = c(5,3,0,1)
R[2,] = c(4,0,0,1)
R[3,] = c(1,1,0,5)
R[4,] = c(1,0,0,4)
R[5,] = c(0,1,5,4)
K = 2

set.seed(100)
random_p = runif(nrow(R) * length(1:K))
P = matrix(random_p,nrow=nrow(R), ncol=length(1:K))
random_q = runif(length(1:K) * ncol(R))
Q = matrix(random_q,nrow=length(1:K), ncol = ncol(R))

learningRate = 0.0001
iterations = 5000
solution = gradient_descent(R,P,Q,K, learningRate, iterations)
solution[[1]] %*% solution[[2]]
R

plot(1:(iterations+1),solution[[3]],xlab="iterations",ylab="error")
