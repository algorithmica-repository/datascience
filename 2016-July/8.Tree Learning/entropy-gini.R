entropy = function(p) {
  ent = 0
  for(i in 1:length(p)) {
    ent = ent + (-p[i] * log2(p[i]))
  }
  return(ent)
}
entropy(c(0.01, 0.99))
entropy(c(0.33, 0.33, 0.33))
entropy(c(0.8, 0.1, 0.1))

prob = seq(0,1,0.1)
ent = numeric()
for(i in 1:length(prob)) {
  ent = c(ent, entropy(c(prob[i], 1-prob[i])))
}
X11()
plot(prob, ent, type="l")


gini_ind = function(p) {
  gini = 0
  for(i in 1:length(p)) {
    gini = gini + (p[i] * p[i])
  }
  return(1 - gini)
}
gini_ind(c(0.5, 0.5))
gini_ind(c(0.33, 0.33, 0.33))
gini_ind(c(0.8, 0.1, 0.1))

gini = numeric()
for(i in 1:length(prob)) {
  gini = c(gini, gini_ind(c(prob[i], 1-prob[i])))
}
X11()
plot(prob, gini, type="l")
