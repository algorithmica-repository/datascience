f =  function(x) {
  1.2 * (x-2)^2 + 3.2
}

xs = seq(0,4,len=20)
windows(width=50, height=60)
plot(xs , f (xs), type="l",xlab="x",ylab=expression(1.2(x-2)^2 +3.2))


grad = function(x){
  1.2*2*(x-2)
}

iterations = 100 
threshold=1e-5
x = 4 
xtrace = x 
ftrace = f(x) 
stepFactor = 0.005 
for (iter in 1:iterations) {
  x = x - stepFactor*grad(x) 
  xtrace = c(xtrace,x) 
  ftrace = c(ftrace,f(x)) 
  if(iter > 1 && (abs(ftrace[iter] - ftrace[iter-1])) < threshold) break
}
df = data.frame(x=xtrace,f=ftrace)
lines(df$x , df$f , type="b",col="blue")
X11()
plot(1:(iterations+1),df$f[1:(iterations+1)],col="blue",xlab="Iterations",ylab="function",main="Convergence")
