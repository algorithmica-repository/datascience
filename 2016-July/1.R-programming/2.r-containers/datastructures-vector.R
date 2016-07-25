#creating vectors
v1 = c(10,20,30)
v2 = 1:10
v3 = seq(1,100,2)
v4 = rep(1,50)
v5 = numeric(10)

#accessing elements of vector
v1[1]
names(v1) = c("e1","e2","e3")
v1["e1"]

#finding the type of v1,v2,v3
class(v1)
is.vector(v1)

#operations on vector
length(v1)
v6 = v3 + v4
v7 = v3 - v4
v8 = v3 * v4

# functions returning vector
set.seed(10)
s1 = sample(1:10, 5)
s2 = sample(1:20, 10)
s3 = sample(1:10, 20, replace=T)