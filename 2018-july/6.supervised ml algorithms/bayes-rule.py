def bayes(prior, likelihood, evidence):
    return (prior* likelihood) / evidence

##cancer diagnosis based on test result
#prior belief: 2% test reliability:90%
print(bayes(0.02, 0.9, (0.9*0.02 + 0.1*0.98)))

#prior belief: 50% test reliability:90%
print(bayes(0.5, 0.9, (0.9*0.5 + 0.1*0.5)))

#prior belief: 80% test reliability:90%
print(bayes(0.8, 0.9, (0.9*0.8 + 0.1*0.2)))

#prior belief: 2% test reliability:99%
print(bayes(0.02, 0.99, (0.99*0.02 + 0.01*0.98)))

#prior belief: 2% test reliability:99.99%
print(bayes(0.02, 0.9999, (0.9999*0.02 + 0.0001*0.98)))

#prior belief: 2% test reliability:100%
print(bayes(0.02, 1, (1*0.02 + 0*0.98)))

##reason about coin fairness based on coin toss result
#belief of on coin fairness after first toss gives head
print(bayes(0.9, 0.5, (0.5*0.9 + 1*0.1)))
#belief of on coin fairness after second toss gives head
print(bayes(0.81, 0.5, (0.5*0.81 + 1*0.19)))
#belief of on coin fairness after third toss gives head
print(bayes(0.68, 0.5, (0.5*0.68 + 1*0.32)))
#belief of on coin fairness after toss gives tail
print(bayes(0.68, 0.5, (0.5*0.68 + 1*0)))
