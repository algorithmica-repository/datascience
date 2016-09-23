fun =  function(x,y) {
  return(x^2 + y^2)
}

x = seq(-2, 4, 0.25)
y = seq(-2, 4, 0.25) 
f = outer(x,y,fun)
windows(width=50, height=60)
persp3D(x, y, f, xlab="x", ylab="y", zlab="f",col = ramp.col(n = 50,col = c("#FF033E", "#FFBF00", "#FF7E00", "#08E8DE", "#00FFFF", "#03C03C"),alpha = .1), border = "#808080", theta = 30, phi = 10, colkey = FALSE)

grad = function(x,y){
  return(c(2*x,2*y))
}

iterations = 1000 
threshold=1e-5
x = c(4,4) 
xtrace = x 
ftrace = fun(x[1],x[2]) 
stepFactor = 0.05 
for (iter in 1:iterations) {
  x = x - stepFactor*grad(x[1],x[2]) 
  xtrace = rbind(xtrace,x) 
  ftrace = c(ftrace,fun(x[1],x[2])) 
  if(iter > 1 && (abs(ftrace[iter] - ftrace[iter-1])) < threshold) break
}
df = data.frame(x=xtrace[,1], y=xtrace[,2], f=ftrace)
points3D(df$x,df$y,df$f, pch = 20, col = 'red', add = TRUE)
x11()
plot(1:(iterations+1),df$f[1:(iterations+1)],col="blue",xlab="Iterations",ylab="function",main="Convergence")
