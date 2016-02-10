#understanding logistic function behaviour
v1 = seq(-100,100,0.1)
v2 = 1/(1+exp(-v1))
plot(v1,v2)


v1 = seq(0,1,0.1)
cost1 = numeric()
for(i in 1:length(v1)) {
  cost1 = c(cost1, -log(v1[i]))
}
plot(v1,cost1,type="l")

cost2 = numeric()
for(i in 1:length(v1)) {
  cost2 = c(cost2, -log(1-v1[i]))
}
plot(v1,cost2,type="l")

cost = numeric()
for(i in 1:length(v1)) {
  cost1 = c(cost1, -log(v1[i]))
}
plot(v1,cost1,type="l")
