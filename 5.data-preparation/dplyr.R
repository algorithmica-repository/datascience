library(dplyr)
library(reshape2)
students = read.csv("E:/data analytics/datasets/students.csv",TRUE)
class(students)
dim(students)
names(students)
str(students)
head(students,10)
tail(students)
summary(students)

students %>% select(Height,BloodType) %>% 
  filter(Height>70) %>%
  arrange(Height) %>%
  mutate(NewHeight = Height + 10)


df1 = data.frame(c("A","B","C"), c("I1","I2","I3"))
names(df1) = c("name1","instrument")
df1   
str(df1)
df2 = data.frame(c("A","B","C"), c(TRUE,FALSE,TRUE))
names(df2) = c("name2","band")
df2  
str(df2)
inner_join(df1,df2,by=c("name1"="name2"))
left_join(df1,df2,by=c("name1"="name2"))
semi_join(df1,df2,by=c("name1"="name2"))
anti_join(df1,df2,by=c("name1"="name2"))

pop = read.csv("E:/data analytics/datasets/pop_density.csv",TRUE)
dim(pop)
head(pop)
str(pop)
names(pop) = c("state", seq(1910,2010,10))

mpop = melt(pop,id.vars = "state", variable.name = "year", value.name = "population")
dim(mpop)

pop = dcast(mpop, state ~ year, value.var = "population" )
dim(pop)
str(pop)
