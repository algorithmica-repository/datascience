#5 models with 70% accuracy
1 - pbinom(2, 5, 0.7)

# 5 models with 80% accuracy
1 - pbinom(2, 5, 0.8)

#1001 models with 70% accuracy
1 - pbinom(500, 1001, 0.7)

#2001 models with 70% accuracy
1 - pbinom(1000, 2001, 0.7)

#101 models with 70% accuracy
1 - pbinom(50, 101, 0.7)

#bootstrap sample
n = 100000
bsample = sample(1:n, n, replace = T)
#63% unique data will result from bootstrap resampling
length(unique(bsample))/n

#strength of bootstrapping 
population = 1:10000
mean(population)

mean(sample(population, 0.63 * length(population)))
mean(sample(population, length(population), T))


