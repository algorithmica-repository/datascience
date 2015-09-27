library(ggplot2)

students=read.csv("E:/data analytics/datasets/students.csv")
dim(students)
summary(students$Sex)
table(students$Level)
with(students, table(Level))

ggplot(students, aes(x = BloodType)) + geom_bar()
with(students, table(Sex, Level))
ggplot(students, aes(x = Level, fill = BloodType)) + geom_bar(position = "dodge")`-
  
ggplot(students, aes(x = Height)) + geom_dotplot()



