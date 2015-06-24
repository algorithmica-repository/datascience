opts_chunk$set(cache=TRUE, message=FALSE)
# smaller font size for chunks
opts_chunk$set(size = 'footnotesize')
options(width = 60)



## install.packages("ggplot2", dependencies = TRUE)



head(iris)



ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point()



## ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width))
##  + geom_point()
## myplot <- ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width))
## myplot + geom_point()



ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point(size = 3)



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(size = 3)



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(aes(shape = Species), size = 3)



## # Make a small sample of the diamonds dataset
## d2 <- diamonds[sample(1:dim(diamonds)[1], 1000), ]

# Efficient and more readable random sample
# d2 <- diamonds[sample(nrow(diamonds), 1000), ]

# Robust and more readable random sample
# d2 <- diamonds[sample(1:nrow(diamonds), 1000, replace = FALSE), ]


d2 <- diamonds[sample(1:dim(diamonds)[1], 1000), ]
ggplot(d2, aes(carat, price, color = color)) + geom_point() + theme_gray()



library(MASS)
ggplot(birthwt, aes(factor(race), bwt)) + geom_boxplot()



h <- ggplot(faithful, aes(x = waiting))
h + geom_histogram(binwidth = 30, colour = "black")



h <- ggplot(faithful, aes(x = waiting))
h + geom_histogram(binwidth = 8, fill = "steelblue",
                   colour = "black")



setwd('~/Github/ggplot2-lecture/')



climate <- read.csv("climate.csv", header = T)
ggplot(climate, aes(Year, Anomaly10y)) +
  geom_line()



ggplot(climate, aes(Year, Anomaly10y)) +
  geom_ribbon(aes(ymin = Anomaly10y - Unc10y,
                  ymax = Anomaly10y + Unc10y),
              fill = "blue", alpha = .1) +
  geom_line(color = "steelblue")



cplot <- ggplot(climate, aes(Year, Anomaly10y))
cplot <- cplot + geom_line(size = 0.7, color = "black")
cplot <- cplot + geom_line(aes(Year, Anomaly10y + Unc10y), linetype = "dashed", size = 0.7, color = "red")
cplot <- cplot + geom_line(aes(Year, Anomaly10y - Unc10y), linetype = "dashed", size = 0.7, color = "red")
cplot + theme_gray()



ggplot(iris, aes(Species, Sepal.Length)) +
  geom_bar(stat = "identity")



df  <- melt(iris, id.vars = "Species")
ggplot(df, aes(Species, value, fill = variable)) +
  geom_bar(stat = "identity")



iris[1:2, ]
df  <- melt(iris, id.vars = "Species")
df[1:2, ]



ggplot(df, aes(Species, value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge")



ggplot(d2, aes(clarity, fill = cut)) +
  geom_bar(position = "dodge",stat = "bin") + theme_gray()



clim <- read.csv('climate.csv', header = TRUE)
clim$sign <- ifelse(clim$Anomaly10y<0, FALSE, TRUE)
# or as simple as
# clim$sign <- clim$Anomaly10y < 0
ggplot(clim, aes(Year, Anomaly10y)) + geom_bar(stat = "identity", aes(fill = sign)) + theme_gray()



ggplot(faithful, aes(waiting)) + geom_density()



ggplot(faithful, aes(waiting)) +
  geom_density(fill = "blue", alpha = 0.1)



ggplot(faithful, aes(waiting)) +
  geom_line(stat = "density")



## aes(color = variable)
## aes(color = "black")
## # Or add it as a scale
## scale_fill_manual(values = c("color1", "color2"))



## library(RColorBrewer)
## display.brewer.all()



df  <- melt(iris, id.vars = "Species")
ggplot(df, aes(Species, value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set1")



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point() +
  facet_grid(Species ~ .) +
  scale_color_manual(values = c("red", "green", "blue"))



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point() +
  facet_grid(Species ~ .)



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point() +
  facet_grid(. ~ Species)



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point() +
  facet_wrap( ~ Species)



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(aes(shape = Species), size = 3) +
  geom_smooth(method = "lm")



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(aes(shape = Species), size = 3) +
  geom_smooth(method = "lm") +
  facet_grid(. ~ Species)



## + theme()
## # see ?theme() for more options



## ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
## geom_point(size = 1.2, shape = 16) +
## facet_wrap( ~ Species) +
## theme(legend.key = element_rect(fill = NA),
## legend.position = "bottom",
## strip.background = element_rect(fill = NA),
## axis.title.y = element_text(angle = 0))



ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(size = 1.2, shape = 16) +
  facet_wrap( ~ Species) +
  theme(legend.key = element_rect(fill = NA),
        legend.position = "bottom",
        strip.background = element_rect(fill = NA),
        axis.title.y = element_text(angle = 0))



## install.packages('ggthemes')
## library(ggthemes)
## # Then add one of these themes to your plot
##  + theme_stata()
##  + theme_excel()
##  + theme_wsj()
##  + theme_solarized()



## my_custom_plot <- function(df, title = "", ...) {
##     ggplot(df, ...) +
##     ggtitle(title) +
##     whatever geoms() +
##     theme(...)
## }



## plot1 <- my_custom_plot(dataset1, title = "Figure 1")



## scale_fill_discrete(), scale_colour_discrete()
## scale_fill_hue(), scale_color_hue()
## scale_fill_manual(),  scale_color_manual()
## scale_fill_brewer(), scale_color_brewer()
## scale_linetype(), scale_shape_manual()



library(MASS)
ggplot(birthwt, aes(factor(race), bwt)) +
  geom_boxplot(width = .2) +
  scale_y_continuous(labels = (paste0(1:4, " Kg")),
                     breaks = seq(1000, 4000, by = 1000))



## # Assign the plot to an object
## dd <- ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
## geom_point(size = 4, shape = 16) +
## facet_grid(. ~Species)
## # Now add a scale
## dd +
## scale_y_continuous(breaks = seq(2, 8, by = 1),
## labels = paste0(2:8, " cm"))



h + geom_histogram( aes(fill = ..count..), color="black") +
  scale_fill_gradient(low="green", high="red")



## ggsave('~/path/to/figure/filename.png')



## ggsave(plot1, file = "~/path/to/figure/filename.png")



## ggsave(file = "/path/to/figure/filename.png", width = 6,
## height =4)



## ggsave(file = "/path/to/figure/filename.eps")
## ggsave(file = "/path/to/figure/filename.jpg")
## ggsave(file = "/path/to/figure/filename.pdf")