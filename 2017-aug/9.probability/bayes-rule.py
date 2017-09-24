def bayes(prior, likelihood, evidence):
    return (prior * likelihood) / evidence

#prior belief:2% and test reliability:90%
print(bayes(0.02, 0.9, (0.02*0.9 + 0.98*0.1)))

#prior belief:2% and test reliability:99%
print(bayes(0.02, 0.99, (0.02*0.99 + 0.98*0.01)))

###applying bayes rule for cancer diagnosis
#prior belief:2% and test reliability:100%
print(bayes(0.02, 1, (0.02*1 + 0.98*0)))

#prior belief:50% and test reliability:90%
print(bayes(0.5, 0.9, (0.5*0.9 + 0.5*0.1)))

#prior belief:50% and test reliability:95%
print(bayes(0.5, 0.95, (0.5*0.95 + 0.5*0.05)))

###applying bayes rule for detecting fair coin or not
#prior belief on being fair:90% and coin toss gives head
print(bayes(0.9, 0.5, (0.9*0.5 + 0.1*1)))

#prior belief on being fair:81% and coin toss gives head
print(bayes(0.81, 0.5, (0.81*0.5 + 0.19*1)))

#prior belief on being fair:68% and coin toss gives head
print(bayes(0.68, 0.5, (0.68*0.5 + 0.32*1)))

#prior belief on being fair:51% and coin toss gives tail
print(bayes(0.51, 0.5, (0.51*0.5 + 0.49*0)))





