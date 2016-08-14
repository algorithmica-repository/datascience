coin_tosses = sample(0:1,10000,T)
table(coin_tosses)

dice_throws = sample(1:6,10000,T)
table(dice_throws)

bayes_prob = function(prior, likelihood, evidence) {
  return(prior * likelihood / evidence)
}
#Prior prob of having cancer = 2%
#evidence from cancer test = 90% reliability
cancer_diagnosis1 = bayes_prob(0.02,0.9,0.02 * 0.9 + 0.98 * 0.1)

#Prior prob of having cancer = 2%
#evidence from cancer test = 99% reliability
cancer_diagnosis2 = bayes_prob(0.02,0.99,0.02 * 0.99 + 0.98 * 0.01)

#Prior prob of having cancer = 2%
#evidence from cancer test = 100% reliability
cancer_diagnosis3 = bayes_prob(0.02,1,0.02 * 1 + 0.98 * 0)

#Prior prob of having cancer = 50%
#evidence from cancer test = 90% reliability
cancer_diagnosis4 = bayes_prob(0.5,0.9,0.5 * 0.9 + 0.5 * 0.1)




