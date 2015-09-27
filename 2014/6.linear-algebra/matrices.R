m = matrix(c(1,2,2,1),2,2,byrow = TRUE)
e = eigen(m)
str(e)
e$values
e$vectors
det(m)

m = matrix(c(cos(16.26),-sin(16.26),sin(16.26),cos(16.26)),2,2,byrow = TRUE)
m.t = t(m)
m %*% m.t

m1 = matrix(c(1,2,3,1),2,2,byrow = TRUE)
m1.t = t(m1)
m1%*%m1.t

m2 = matrix(c(1,2,3,1,2,2),2,3,byrow = TRUE)
m2.t = t(m2)
m2%*%m2.t

m1 = matrix(c(7,1,1,3),2,2)
m1.inv = solve(m1)
c=c(4,3)
m2=as.matrix(c)
m1.inv%*%m2

a = matrix(c(2,-1,1,1),2,2)
x = matrix(c(1,8),2,1)
a %*% x

e = eigen(a)
p = e$vectors
d=diag(e$values)
p.inv=solve(p)

tmp = p.inv%*%x
tmp = d%*%tmp
p%*%tmp

p = matrix(rep(1,9),3,3)
e = eigen(p)
e$values

a= matrix(c(1,2,3,2,1,4),3,2)
r = svd(a)
str(r)

