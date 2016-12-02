# %d -> Day
# %m -> Numeric Month 
# %b -> Abbreviated Month
# %B -> Full Month
# %y -> 2-digit year
# %Y -> 4-digit year

dates1 = c('1/10/2006','4/25/2006')
class(dates1)
dates1_std = as.Date(dates1,format= "%m/%d/%Y")
class(dates1_std)

dates2 = c('10-1-98','25-4-06')
class(dates2)
dates2_std = as.Date(dates2,format= "%d-%m-%y")
class(dates2_std)

weekdays(dates1_std)
months(dates1_std)

dates1_std[1] - dates1_std[2]

#current data
Sys.Date()
