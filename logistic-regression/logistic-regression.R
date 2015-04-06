donner=read.csv("E:/data analytics/datasets/donner.txt", sep = '\t', header = TRUE)
names(donner) = c("age","gender","status")
str(donner)

d2 = function(model) { round(1-(model$deviance/model$null.deviance),4) }


model=glm(status ~ age,data=donner, family=binomial)
summary(model)
d2(model)

model=glm(status ~ age + gender,data=donner, family=binomial)
summary(model)
d2(model)

model=glm(status ~ age + gender + age * gender,data=donner, family=binomial)
summary(model)
d2(model)

model=glm(status ~ age * gender,data=donner, family=binomial)
summary(model)
d2(model)

