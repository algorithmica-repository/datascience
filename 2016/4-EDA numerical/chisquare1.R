# coin fairness test
outcomes1=c(60,40)
chisq.test(outcomes1)

outcomes2=c(50,50)
chisq.test(outcomes2)

outcomes3=c(55,45)
chisq.test(outcomes3)

outcomes4=c(80,20)
chisq.test(outcomes4)

# fairness of dice
dice_outcomes1 = c(10,15,20,30,40,5)
chisq.test(dice_outcomes1)

dice_outcomes2 = c(20,20,20,15,30,15)
chisq.test(dice_outcomes2)

dice_outcomes3 = c(20,20,20,20,20,20)
chisq.test(dice_outcomes3)