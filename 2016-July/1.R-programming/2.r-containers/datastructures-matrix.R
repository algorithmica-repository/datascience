m1 = matrix(1:6,3,2,T)
dim(m1)
m1[1,]
m1[,2]
m1[3,1]

cust.names = c("cust1","cust2","cust3")
movie.names = c("movie1","movie2")
m2 = matrix(1:6,3,2,TRUE,list(cust.names,movie.names))
m2[1,]
m2[,2]
m2[3,1]
m2["cust1",]

m3 = matrix(rep(TRUE,6),3,2,TRUE)
dim(m3)
rownames(m3) = cust.names
colnames(m3) = movie.names

m4 = matrix(NA,3,2,TRUE)
m4[1,] = c(1,2)
m4[2,] = c(4,5)
m4[3,] = c(5,6)
m4 = rbind(m4,c(10,12))
