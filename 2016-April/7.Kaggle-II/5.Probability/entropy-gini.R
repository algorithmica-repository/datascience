entropy = function(p) {
  ent = 0
  for(i in 1:length(p)) {
    ent = ent + (-p[i] * log2(p[i]))
  }
  return(ent)
}

p1 = c(0.5, 0.5)
entropy(p1)

p2 = c(0.33, 0.33, 0.33)
entropy(p2)

p3 = c(0.6, 0.3, 0.1)
entropy(p3)

p4 = c(0.8, 0.1, 0.1)
entropy(p4)

prob = seq(0,1,0.1)
ent = numeric()
for(i in 1:length(prob)) {
  ent = c(ent, entropy(c(prob[i], 1-prob[i])))
}
plot(prob, ent, type="l")


gini_ind = function(p) {
  gini = 0
  for(i in 1:length(p)) {
    gini = gini + (p[i] * p[i])
  }
  return(1 - gini)
}

gini = numeric()
for(i in 1:length(prob)) {
  gini = c(gini, gini_ind(c(prob[i], 1-prob[i])))
}
plot(prob, gini, type="l")