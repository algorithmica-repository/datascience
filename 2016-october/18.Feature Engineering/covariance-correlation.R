library(ggplot2)
stock_plot = function(s1,s2) {
  df = data.frame(a=s1,b=s2)
  X11()
  print(ggplot(df) + geom_point(aes(x = a, y = b)))
  print(cov(df$a, df$b))
  print(cor(df$a, df$b))
}

s1 = c(100, 200, 300, 400)
s2 = c(10, 20, 30, 50)
stock_plot(s1,s2)


s3 = c(100, 200, 300,  400)
s4 = c(50, 40, 35, 32)
stock_plot(s3, s4)

s5 = c(100, 200, 300, 400)
s6 = c(1, 2, 3, 5)
stock_plot(s5,s6)

s7 = c(100, 200, 300, 400)
s8 = c(500, 600, 700, 800)
stock_plot(s7,s8)

#why mean of z-scores is 0?
x = c(10,20,30,40, 50, 60, 70)
x_z = (x - mean(x) ) / sd(x)
df = data.frame(x, x_z)
mean(x)
mean(x_z)

