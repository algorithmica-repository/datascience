age = c(10,20,30,40)
ids1 = 1:100
ids2 = seq(1,100,2)
survived = rep(0,418)
rvector1 = rep(sample(1:10,1),100)
rvector2 = sample(1:100,100)
vector3=100
#vector4=c(10.5,"abc")

class(age)

# subsetting vector
age[1:3]
age > 20
age1 = age + 10

age == 20

age2 = c(age, 60)
rm(age1)
gc()

age3 = c(40,20,10,60)
sort(age3)

#named access for vector data
dummy = c(10,20,30,40)
names(dummy) = c("e1","e2","e3","e4")
dummy[1]
dummy["e1"]

dummy[3]
dummy["e3"]

dummy[c(2,3)]
dummy[c("e2","e3")]



