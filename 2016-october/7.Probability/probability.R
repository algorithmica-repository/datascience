coin_tosses = sample(0:1, 1000, replace = T)
table(coin_tosses)

dice_throws= sample(1:6, 1000000, replace = T)
table(dice_throws)

posterior = function(prior, likelihood, normalize_factor) {
   return( prior * (likelihood / normalize_factor) )
}

## applying bayes rule for cancer diagnosis
#varying the reliability of test
posterior(0.02, 0.9, (0.02*0.9 + 0.98*0.1))
posterior(0.02, 0.98, (0.02*0.98 + 0.98*0.02))
posterior(0.02, 1, (0.02*1 + 0.98*0))

#varying the prior probabilities
posterior(0.6, 0.9, (0.6*0.9 + 0.4*0.1))
posterior(0.6, 0.98, (0.6*0.98 + 0.4*0.02))

## applying bayes rule for deciding coin fair or not

posterior(0.9, 0.5, (0.9 * 0.5 + 0.1 * 1))
posterior(0.8, 0.5, (0.8 * 0.5 + 0.2 * 1))
posterior(0.67, 0.5, (0.67 * 0.5 + 0.33 * 1))
posterior(0.5, 0.5, (0.5 * 0.5 + 0.5* 0))
