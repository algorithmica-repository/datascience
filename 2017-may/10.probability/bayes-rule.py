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

#after 1st evidence of head outcome
print(bayes(0.9, 0.5, (0.5*0.9 + 1*0.1)))
#after 2nd evidence of head outcome
print(bayes(0.81, 0.5, (0.5*0.81 + 1*0.19)))
#after 3rd evidence of head outcome
print(bayes(0.68, 0.5, (0.5*0.68 + 1*0.32)))