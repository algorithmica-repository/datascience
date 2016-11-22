dbinom(2, 5, 0.8)
10* (0.8 ^ 2) * (0.2^3)

pbinom(2, 5, 0.8)

1* (0.8 ^ 0) * (0.2^5) +
  5* (0.8 ^ 1) * (0.2^4) +
  10* (0.8 ^ 2) * (0.2^3)

rbinom(4, 10, 0.8)
sample(1:10, 4)

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
+
  
  
  
  
  
  
  
  
  
  
  
































0