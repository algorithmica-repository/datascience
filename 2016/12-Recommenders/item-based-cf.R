cosine.similarity = function(x,y) {
  cosine = sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))
  return(cosine)
}

item.similarities = function(pref) {
  item_sim = matrix( NA, nrow = ncol(pref),ncol = ncol(pref) )
  for (i in 1:nrow(item_sim)) {
    for (j in i:ncol(item_sim)) {
      if (i == j)
        item_sim[i,i] = 0
      else {
        item_sim[i,j] = cosine.similarity(pref[,i],pref[,j])
        item_sim[j,i] = item_sim[i,j]
     }
    }
  }
  return(item_sim)
}

topN = function(pref,item_sim,k) {
  all_recom = list()
  for(user in 1:nrow(pref)) {
    
    #get all the items liked by the current user
    liked_items = which(pref[user,]==1)
    
    #foreach liked item, find k closest neighbours and generate
    #candidate items by union of all of them
    candidate_items = numeric()
    for(item in 1:length(liked_items)) {
      nn = order(item_sim[item,],decreasing=TRUE)[1:k]
      candidate_items = union(candidate_items,nn)
    }
    #remove the items liked by the user from candidate items
    recm_items = setdiff(candidate_items,liked_items)
    
    #reorder the items based on user-user similarity
    weighted_recm_items = list()
    for(recm.ind in 1:length(recm_items)) {
      weight = 0
      for(liked.ind in 1:length(liked_items)) {
        weight = weight + item_sim[recm_items[recm.ind],liked_items[liked.ind]] * pref[user,liked_items[liked.ind]]
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
pref[3,] = c(0,0,0,1,1,1)
pref[4,] = c(1,0,0,0,0,1)
pref[5,] = c(0,1,1,0,0,1)

item_sim = item.similarities(pref)
recm = topN(pref,item_sim,3)
recm