v1 = c(10,20,30)
v1
v1[1]
names(v1) = c("e1","e2","e3")
v1["e1"]

is.vector(v1)
v2 = 1:100
v2
length(v2)
#1,3,5,,...
v3 = seq(1,100,2)
v3
#1,1,1,1,1,
v4 = rep(1,100)
length(v4)
v5 = v2*v4
mean(v4)

cust.names = c("cust1","cust2","cust3")
movie.names = c("movie1","movie2")
m1 = matrix(1:6,3,2,TRUE,list(cust.names,movie.names))
m1[1,]
m1[,2]
m1[3,1]
m1["cust1",]

m2 = matrix(rep(TRUE,6),3,2,TRUE)
dim(m2)
rm(m1)
rm(a)
gc()
rownames(m2) = cust.names
colnames(m2) = movie.names

m3 = matrix(NA,3,2,TRUE)
m3[1,] = c(1,2)
m3[2,] = c(4,5)
m3[3,] = c(5,6)
m3 = rbind(m3,c(10,12))
