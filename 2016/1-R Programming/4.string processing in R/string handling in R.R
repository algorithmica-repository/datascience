library(stringr)

setwd("C:/Users/Thimma Reddy/Documents/GitHub/datascience/datasets")
files = list.files()
nchar(files[1])
str_length(files[1])

#extract substring
substr(files,1,2)
str_sub(files,1,2)

#split a string with pattern
strsplit(files,".")
str_split(files,".")

#locate the pattern in string and return indices
grep("txt", files)
which(str_detect(files,"txt"))

#locate the pattern in string and return the strings
grep("txt", files,value=TRUE)
files[!is.na(str_extract(files,"txt"))]

#locate the pattern in string and return the start positions
regexpr("txt",files)
str_locate(files, "txt")

#locate the names having txt as part of them and replace txt with csv
sub("txt","csv",files)
gsub("txt","csv",files)

pattern = "txt | dat | zip | tsv"
sub(pattern,"csv",files)



