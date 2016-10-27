ids = 1:4
names = c("s1","s2","s3","s4")
students = data.frame(ids, names)
class(students)

#find the structure of frame
str(students)

#find the dimensions of frame
dim(students)

students

#subsetting in a frame
students[1,]
students[1:3,]
students[c(1,4),]

students[,2]
students[,"names"]
students$names
students[ids>2,]
