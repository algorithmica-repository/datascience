

### Let's create some data!


## Vector

vec_1 <- c(1,2,3,4,5,6)
vec_1

vec_2 <- 1:6
vec_2

vec_3 <- c("One", "Two", "Three", "Four", "Five", "Six")
vec_3

vec_4 <- 1:500
vec_4


## Matrix

mat_1 <- matrix(1:16, nrow = 4, ncol = 4)
mat_1

mat_2 <- matrix(vec_1, nrow = 3, ncol = 2, byrow = TRUE)
mat_2

mat_3 <- matrix(vec_1, nrow = 3, ncol = 2, byrow = FALSE)
mat_3

rows <- c("A", "B", "C")
cols <- c("AA", "BB")

mat_4 <- matrix(vec_1, nrow = 3, ncol = 2, dimnames = list(rows, cols))
mat_4

rownames(mat_3) <- rows # rename rows afterwards
mat_3

## Array

array_1 <- array(1:18, c(3,2,3))
array_1

dims <- c("AAA", "BBB", "CCC")
array_2 <- array(1:18, c(3, 2, 3), dimnames = list(rows, cols, dims))
array_2


## Factor

vec_1
vec_4 <- c("odd", "even", "odd", "even", "odd", "even")
vec_4

fac_1 <- factor(vec_4)
fac_1 
as.numeric(fac_1) # Levels are created alphabetiacal

vec_5 <- c("small", "small", "medium", "medium", "high", "high")
vec_5

fac_2 <- factor(vec_5, ordered = TRUE)
fac_2 # look what happened to the Levels output!
as.numeric(fac_2) # still alphabetical

fac_3 <- factor(vec_5, ordered = TRUE, levels = c("small", "medium", "high"))
fac_3 
as.numeric(fac_3) # Yeah!


## Data Frame

df_1 <- data.frame(vec_1, vec_2, vec_3, fac_1, fac_3)
df_1
str(df_1) # shows structure - data.frame() usually converts characters into factors!

names(df_1) <- c("var_1", "var_2", "var_3", "var_4", "var_5")
df_1


## List

list_1 <- list(vec_1, vec_2, vec_3)
list_1

list_2 <- list(vector_1 = vec_1, vector_2 = vec_2, vector_3 = vec_3)
list_2

list_3 <- list(text = "Sample text", vector = vec_1, matrix = mat_2, array = array_2, factor = fac_3, data_frame = df_1)
list_3


### saving all data
save.image(file="data.RData")
