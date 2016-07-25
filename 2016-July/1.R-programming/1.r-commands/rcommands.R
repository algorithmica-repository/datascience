###package related commands

#shows all packages installed on machine(may or maynot be loaded into R workspace)
library()
#install the required package
install.packages(nnet)

#show all the packages loaded into R workspace and also shows the search path order
search()
#load the required package
library(nnet)

#explore functions and objects in a package
ls("package:nnet")

###getting help in R
help(name) 
?name
example(name)

###environment control commands
ls()
ls(pattern="")
rm(a)
getwd()
setwd("D:\\")
