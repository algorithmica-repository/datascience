setwd("E:\\data analytics\\datasets")

dates = read.table("date-input.txt",header=TRUE, sep="\t", stringsAsFactors=FALSE)
dim(dates)
str(dates)
class(dates$month_day_year)
dates$month_day_year

# %d -> Day
# %m -> Numeric Month 
# %b -> Abbreviated Month
# %B -> Full Month
# %y -> 2-digit year
# %Y -> 4-digit year

col1 = as.Date(dates$month_day_year,format= "%m/%d/%Y")
class(col1)

col4 = as.Date(dates$year_month_day)
class(col4)

#R stores dates internally as the number of days 
#since the first day of 1970, with dates before 1970 
#being stored as negative numbers. 
as.numeric(col4)

col4[2] - col4[1]

weekdays(col4[1])
months(col4[1])

#current data
Sys.Date()

#current time
date()
