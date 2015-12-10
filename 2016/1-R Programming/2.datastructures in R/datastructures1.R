a = 10
b = 20.3
c = a + b
a=20
ls()
rm("b")

v1 = 1:1000
v1

v = c(10,20,15,22,8,2)
v[1:3]
v[c(3,5)]
v[v>15]
v[-c(1,2)]
which(v>15)

v2=1:6
v3=v+v2

v4 = v * v2

sd(v)
length(v)
v2 = sort(v)

v3=seq(1,100,10)

v4=rep(1,10)

v5 = c("aa","bb","ccc")

students = read.csv("E:/data analytics/datasets/students.csv",TRUE)
students

class(students)
rn = c("user1","user2")
cn = c("movie1","movie2","movie3")

user_movies_ratings=matrix(1:6,2,3,dimnames=list(rn,cn))
user_movies_ratings1=rbind(user_movies_ratings,c(5,2,5))


row.names(user_movies_ratings) =c("user1","user2","user3")
user_movies1=matrix(rep(1,6),2,3)
dimnames(user_movies1)=list(rn,cn)
dim(user_movies_ratings)
nrow(user_movies_ratings)
t(user_movies_ratings)

user_movies_ratings2=matrix(1:6,3,2)s
res =  user_movies_ratings1 %*% user_movies_ratings2
 diag(3)

v1=1:3
dim(v1)
length(v1)
v1=as.matrix(v1)
dim(v1)

m1=matrix(1:6,2,3,TRUE,)
m1
m2=matrix(1:6,2,3)
m2

d = c(1,2,3,4)
e = c("red", "white", "red", NA)
f = c(TRUE,TRUE,TRUE,FALSE)
mydata = data.frame(d,e,f)
names(mydata) = c("ID","Color","Passed") 

f=factor(c("y","n"))
f
