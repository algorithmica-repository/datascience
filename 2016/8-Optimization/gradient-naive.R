f =  function(x) {
  1.2 * (x-2)^2 + 3.2
}

xs = seq(0,4,len=20)
# plot the function 
plot(xs , f (xs), type="l",xlab="x",ylab=expression(1.2(x-2)^2 +3.2))

# calculate the gradeint df/dx

grad = function(x){
  1.2*2*(x-2)
}

iterations = 100 
threshold=1e-5
x = 3 # initialize the first guess for x-value
xtrace = x # store x -values for graphing purposes (initial)
ftrace = f(x) # store y-values (function evaluated at x) for graphing purposes (initial)
for (iter in 1:iterations) {
  x <- x - grad(x) # gradient descent update
  xtrace <- c(xtrace,x) # update for graph
  ftrace <- c(ftrace,f(x)) # update for graph
  if(iter > 1 && (abs(ftrace[iter] - ftrace[iter-1])) < threshold) break
}
lines ( xtrace , ftrace , type="b",col="blue")
