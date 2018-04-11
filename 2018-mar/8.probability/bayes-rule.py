def bayes(prior, likelihood, evidence):
    return prior * likelihood / evidence

##cance diagnosis
#prior:2% test:90%
bayes(0.02, 0.9, (0.02*0.9 + 0.98*0.1))
#prior:2%  test:98%
bayes(0.02, 0.98, (0.02*0.98 + 0.98*0.02))
#prior:50% test:98%
bayes(0.5, 0.98, (0.5*0.98 + 0.5*0.02))
#prior:70% test:98%
bayes(0.7, 0.98, (0.7*0.98 + 0.3*0.02))
#prior:10% test:100%
bayes(0.1, 1, (0.1*1 + 0.9*0))
#prior:100% test:90%
bayes(1, 0.9, (1*0.9 + 0*0.1))

##coin is fair or double headed
#prior:90%  outcome=H
bayes(0.9, 0.5, (0.9*0.5+0.1*1))
#prior:81%  outcome=H
bayes(0.81, 0.5, (0.81*0.5+0.19*1))
#prior:68%  outcome=H
bayes(0.68, 0.5, (0.68*0.5+0.32*1))
#prior:51%  outcome=T
bayes(0.51, 0.5, (0.51*0.5+0.49*0))




