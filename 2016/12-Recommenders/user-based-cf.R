cosine.similarity = function(x,y) {
  cosine = sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))
  return(cosine)
}

user.similarities = function(pref) {
  user_sim = matrix( NA, nrow = nrow(pref),ncol = nrow(pref) )
  for (i in 1:nrow(user_sim)) {
    for (j in i:ncol(user_sim)) {
      if (i == j)
        user_sim[i,i] = 0
      else {
        user_sim[i,j] = cosine.similarity(pref[i,],pref[j,])
        user_sim[j,i] = user_sim[i,j]
     }
    }
  }
  return(user_sim)
}
v=c(25,20,10)
order(v)
topN = function(pref,user_sim,k) {
  all_recom = list()
  for(user in 1:nrow(pref)) {
    
    #find the k nearest neighbours
    nn = order(user_sim[user,],decreasing=TRUE)[1:k]
    
    #generate candidate recommendation items from neighbours
    candidate_items = numeric()
    for(neighbor in 1:length(nn)) {
      candidate_items = union(candidate_items,which(pref[neighbor,]==1))
    }
    #remove the items liked by user from candidate items
    recm_items = setdiff(candidate_items,which(pref[user,]==1))
    
    #reorder the items based on user-user similarity
    weighted_recm_items = list()
    for(recm.ind in 1:length(recm_items)) {
      weight = 0
      for(neighbor in 1:length(nn)) {
        weight = weight + user_sim[user,neighbor] * pref[neighbor,recm_items[recm.ind]]
      }
      item_key = paste("item",recm_items[recm.ind],sep="")
      weighted_recm_items[[item_key]] = weight
    }
    user_key = paste("user",user,sep="")
    all_recom[[user_key]] = weighted_recm_items
  }
  return(all_recom)
}

pref = matrix(NA,nrow=5,ncol=6,byrow = TRUE)
pref[1,] = c(1,1,0,0,0,1)
pref[2,] = c(1,0,1,0,1,0)
pref[3,] = c(0,0,0,0,1,1)
pref[4,] = c(1,0,0,0,0,1)
pref[5,] = c(0,1,1,0,0,1)

user_sim = user.similarities(pref)
user_sim
recm = topN(pref,user_sim,3)
recm