e1 = 5

e2 = c(1:3)

e3 = matrix(1:6,3,2)

e4 = data.frame(c1=c(1,2),c2=c("a","b"))

list1 = list(e1,e2,e3,e4)
list1[[1]]
names(list1) = c("e1","e2","e3","e4")
list1$e1
length(list1)
