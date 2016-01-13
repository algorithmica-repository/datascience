migration = matrix(c(.9,.05,.1,.95),2,2,byrow = TRUE)
#initial_population = c(300,100)
#initial_population = c(200,100)
initial_population = c(100,100)
initial_population_mat = as.matrix(initial_population)

after_population_frame = data.frame(v=c(),h=c())
for(i in 1:100) {
  after_population = migration %*% initial_population_mat
  after_population_frame[i,1] = round(after_population[1,1])
  after_population_frame[i,2] = round(after_population[2,1])
  initial_population_mat = after_population
}

e = eigen(migration)
