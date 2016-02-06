#understanding logistic function behaviour
v1 = seq(-100,100,0.1)
v2 = 1/(1+exp(-v1))
plot(v1,v2)

