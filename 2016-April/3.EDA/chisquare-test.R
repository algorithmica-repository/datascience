# 1.checking fairness of coin 

# observed data for 100 coin tosses
outcomes1=c(60,40)
chisq.test(outcomes1)

outcomes2=c(50,50)
chisq.test(outcomes2)

outcomes3=c(55,45)
chisq.test(outcomes3)

outcomes4=c(80,20)
chisq.test(outcomes4)

# 2.checking fairness of dice

# observed data for 120 dice throws
dice_outcomes1 = c(10,15,20,30,40,5)
chisq.test(dice_outcomes1)

dice_outcomes2 = c(20,20,20,15,30,15)
chisq.test(dice_outcomes2)

dice_outcomes3 = c(20,20,20,20,20,20)
chisq.test(dice_outcomes3)

# 3.checking the depedance between type of handed-ness and gender

# observed data for left handed and right handed persons among male and female(260 people)
left_handed1 = c(12,7)
right_handed1 = c(108,133)
df1 = data.frame(left_handed1, right_handed1)
names(df1) = c("male","female")
df1
chisq.test(df1)

left_handed2 = c(15,50)
right_handed2 = c(105,90)
df2 = data.frame(left_handed2, right_handed2)
names(df2) = c("male","female")
df2
chisq.test(df2)