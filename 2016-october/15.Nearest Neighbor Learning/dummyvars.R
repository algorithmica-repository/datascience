age = c(10,20,30)
gender = factor(c("M","F","M"))
pclass = factor(c("1","2","3"))
passengers = data.frame(age, gender,pclass)
str(passengers)

tmp = dummyVars(~age+gender+pclass, passengers)
predict(tmp, passengers)
