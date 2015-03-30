library('ggplot2')

setwd("E:/data analytics/datasets")
heights.weights = read.csv(file.path('.',
                                      '01_heights_weights_genders.csv'),
                            header = TRUE,
                            sep = ',')
str(heights.weights)
dim(heights.weights)
ggplot(heights.weights, aes(x = Height, y = Weight)) +
  geom_point() +
  geom_smooth(method = 'lm')

fitted.regression <- lm(Weight ~ Height,
                        data = heights.weights)
coef(fitted.regression)

#(Intercept) Height
#-350.737192 7.717288
intercept <- coef(fitted.regression)[1]
slope <- coef(fitted.regression)[2]

# predicted.weight <- intercept + slope * observed.height
# predicted.weight == -350.737192 + 7.717288 * observed.height
predict(fitted.regression)

true.values <- with(heights.weights, Weight)
errors <- true.values - predict(fitted.regression)

residuals(fitted.regression)

ggplot(cbind(fitted.regression$model, predict = predict(fitted.regression)), aes(predict, Weight - predict)) +
  geom_point(alpha = 0.3) + geom_smooth(color = "red", method = "lm")

summary(fitted.regression)

summary(fitted.regression)$r.squared
#Therefore, about 74% of the data is predicted correctly by the linear regression line.
