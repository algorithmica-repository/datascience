library(foreach)
x = foreach(i=1:3) %do% 
        sqrt(i)

y = foreach(a=1:3, b=rep(10, 3)) %do% {
      a + b
}

z = foreach(a=1:10, b=rep(10, 3)) %do% {
  a + b
}

x = foreach(i=1:3, .combine='c') %do% 
      sqrt(i)

cfun = function(a, b) {
  return(a+b)
}
x = foreach(i=1:3, .combine='cfun') %do% 
      sqrt(i)

cfun = function(...) {
  NULL
}
x = foreach(i=1:3, .combine='cfun', .multicombine = T) %do% 
  sqrt(i)

