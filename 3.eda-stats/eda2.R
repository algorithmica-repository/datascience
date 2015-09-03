 library(dplyr)

students=read.csv("E:/data analytics/datasets/students.csv")
d=dim(students)
typeof(d)
str(students)
head(students,10)
tail(students)
summary(students)
mean(students$Height)
median(students$Height)
mean(c(46,37,40,33,42,36,40,47,34,45))
sd(students$Height)
mad(students$Height)
IQR(students$Height)
mean(students$MilesHome)
mean( students$MilesHome, na.rm = TRUE)
class(students)
 
students1 = filter(students, Sleep %in% c(6,8) & BloodType=="O")
students2 = select(students1,Height) 
 
 students %>% 
   filter(Sleep %in% c(6,8) & BloodType=="O") %>%
   arrange(Height) %>% mutate(Family=Brothers + Sisters) %>%
   summarise(n())
 
 by.major = group_by(students, Major)
 class(by.major)
 summarise(by.major, count=n()) 
 
 
   
 
 
