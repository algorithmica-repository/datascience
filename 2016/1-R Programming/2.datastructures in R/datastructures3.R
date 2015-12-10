
## Vector

vec_1 # whole vector
vec_1[1] # 1st value
vec_1[c(1,4)] #1st and 2nd value
vec_1[1:4] # 1st thru 4th value

## Matrix

mat_1
mat_1[1,] # 1st row
mat_1[,1] # 1st column
mat_1[1,1] # "1st" cell

mat_4
mat_4[c("A"),] # row "A"


## Array

array_2
array_2[,,2] #2nd dimension
array_2[,,c("BBB")] # same as above
array_2[2,,]

## Data Frame

df_1
df_1[1,] #1st row - see Matrix
df_1$var_2 # coloumn/variable named "var_2"
df_1$var_3 # coloumn/variable named "var_3"
df_1[,3] # 3rd column

## List

list_3
list_3[[3]] # selects the matrix
list_3[3] # selects the matrix, but stays as list

list_3$factor # selects list entry "factor"
list_3$data_frame # selects list entry "data_frame"

# why you should use double brackets
test_1 <- list_3[[3]]
test_2 <- list_3[3]
test_1
test_2
is.matrix(test_1) # is matrix
is.list(test_2) # is list

list_3[[c("matrix")]] # works also
list_3$matrix # works also


## nesting

list_3$data_frame$var_4 # selects column "var_4" in list entry "data_frame"
list_3[[6]]$var_4 # same as above
list_3[[c("data_frame")]]$var_4 # same as above


## different ways to access data

array_2
array_2[1,2,2]
array_2[,,2][1,2] # same as above
