#computing probabilities for specific outcome
dbinom(2, 5, 0.8)
10* (0.8 ^ 2) * (0.2^3)

pbinom(2, 5, 0.8)

1* (0.8 ^ 0) * (0.2^5) +
  5* (0.8 ^ 1) * (0.2^4) +
  10* (0.8 ^ 2) * (0.2^3)

#plotting binomial distribution
binomial_distribution = function(n, p) {
  x = 0:n
  b = dbinom(x,n,p)
  barplot(b, names.arg = x)
}
X11()
binomial_distribution(10,0.5)

#plotting cumulative binomial distribution
cumulative_binomial_distribution = function(n, p) {
  x = 0:n
  b = pbinom(x,n,p)
  barplot(b, names.arg = x)
}
X11()
cumulative_binomial_distribution(10,0.3)
  
#generate random data from binomial distribution
set.seed(100)
rbinom(15, 10, 0.5)
sample(1:10, 4)
