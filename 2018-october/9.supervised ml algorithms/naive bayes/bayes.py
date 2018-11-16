def bayes(prior, likelihood, evidence):
    return (prior*likelihood)/(evidence)

##cancer diagnosis
#prior:2% reliability:90%
print(bayes(0.02, 0.9, (0.02*0.9 + 0.98*0.1)))
 
#prior:2% reliability:99%
print(bayes(0.02, 0.99, (0.02*0.99 + 0.98*0.01)))
 
#prior:2% reliability:100%
print(bayes(0.02, 1, (0.02*1 + 0.98*0)))
        
#prior:50% reliability:90%
print(bayes(0.5, 0.9, (0.5*0.9 + 0.5*0.1)))

#prior:50% reliability:99%
print(bayes(0.5, 0.99, (0.5*0.99 + 0.5*0.01)))

##sequence of random experiments
#prior:90%
#1st coin toss = heads 
print(bayes(0.9, 0.5, (0.9*0.5 + 0.1*1)))
#2nd coin toss = heads
print(bayes(0.81, 0.5, (0.81*0.5 + 0.19*1)))
#3rd coin toss = heads
print(bayes(0.68, 0.5, (0.68*0.5 + 0.32*1)))
#4th coin toss = tails
print(bayes(0.51, 0.5, (0.51*0.5 + 0.49*0)))
