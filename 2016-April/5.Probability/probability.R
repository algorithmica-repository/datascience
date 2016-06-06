coin_toss = function(n) {
  sample(0:1,n,rep=T)
}
res1=coin_toss(1000)
table(res1)

dice_throw = function(n) {
  sample(1:6,n,rep=T)
}
res2=dice_throw(2000)
table(res2)

#A = coin is fair A'= coin is double headed
#p(A) = 0.9
#P(A|H) = P(A) * P(H|A) / P(H)
posterior = function(prior, likelihood,evidence) {
  (prior * likelihood) / evidence
}
#P(A) = 0.9, P(H|A) = 0.5
#P(H) = P(H|A) * P(A) + P(H|A') * P(A') = 0.5 * 0.9 + 1 * 0.1
post_prob1 = posterior(0.9,0.5,0.55)

#P(A) = 0.81 P(H|A) = 0.5
#P(H) = 0.5 * 0.81 + 1 * 0.19
post_prob2 = posterior(0.81,0.5,0.595)


bernouli_prob = function(data, theta) {
  total = 1
  for(i in 1:length(data)) {
    total = total * theta ^ data[i] * (1 - theta) ^ (1 - data[i])
  }
  total
}

data1 = c(1,1,0,0)
bernouli_prob(data1,0.5)

data2 = c(1,1,0,0)
bernouli_prob(data1,0.9)

data1 = c(1,1,1,1,0)
theta = seq(0,1,0.1)
likelihood = numeric()
for(i in 1:length(theta)) {
  out = bernouli_prob(data1,theta[i])
  likelihood = c(likelihood, out)
  cat(theta[i],":",out,log(out), "\n")
}
plot(theta,likelihood,type="l")
