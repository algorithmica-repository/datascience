import  numpy.random as rand
import numpy as np

heads = 0
tails = 0
print(np.arange(10))
for i in np.arange(1000000):
    x = rand.sample()              
    if x > 0.5:
        heads = heads + 1
    else:
        tails = tails + 1
print(heads)
print(tails)


heads = 0
tails = 0
for i in np.arange(1000000):
    x = rand.randint(0,2)
    if x == 0:
        heads = heads + 1
    else:
        tails = tails + 1
print(heads)
print(tails)

def bayes(prior,likelihood,evidence):
    return prior*(likelihood/evidence)
#test is 90% accurate, 2% cancer
print(bayes(0.02, 0.9, ((0.9*0.02) + (0.1*0.98))))
#test is 98% accurate, 2% cancer
print(bayes(0.02, 0.98, ((0.98*0.02) + (0.02*0.98))))
#test is 100% accurate, 2% cancer
print(bayes(0.02, 1, ((1*0.02) + (0*0.98))))
#test is 90% accurate, 50% cancer
print(bayes(0.5, 0.9, ((0.9*0.5) + (0.1*0.5))))
#test is 98% accurate, 50% cancer
print(bayes(0.5, 0.98, ((0.98*0.5) + (0.02*0.5))))

bayes()