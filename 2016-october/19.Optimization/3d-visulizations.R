library(plot3D)
fun = function(x,y) {
  return(x+y)
}
x = seq(-2, 4, 0.5)
y = seq(-2, 4, 0.5) 
f = outer(x,y,fun)

windows(width=50, height=60)
persp3D(x, y, f, xlab="x", ylab="y", zlab="f", theta = 30, phi = 10)
X11()
persp3D(x, y, f, xlab="x", ylab="y", zlab="f",color.palette = heat.colors, theta = 30, phi = 10, colkey = FALSE)
X11()
persp3D(x, y, f, xlab="x", ylab="y", zlab="f",color.palette = heat.colors, border = "#808080", theta = 30, phi = 10, colkey = FALSE, ticktype="detailed")

x = c(-1,1)
y = c(2,0) 
z = c(3,3)
points3D(x,y,z, pch = 20, col = 'red', add = TRUE)
