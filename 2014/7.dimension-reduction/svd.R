library(jpeg)
library(ripa)
res = readJPEG("E:/data analytics/testpic.jpg")
class(res)
dim(res)
grey=rgb2grey(res)
class(grey)
dim(grey)
plot(grey)

img.svd = svd(grey)

d <- img.svd$d
class(d)
length(d)
u <- img.svd$u
class(u)
dim(u)
v <- img.svd$v
class(v)
dim(v)

for (i in c(3, 4, 5, 10, 20, 30, 50,100,200))
{
  img.compressed.mat <- u[,1:i] %*% diag(d[1:i]) %*% t(v[,1:i])  
  img.compressed=imagematrix(img.compressed.mat, type = "grey")
  plot(img.compressed)
}

img.compressed.mat <- u[,100:460] %*% diag(d[100:460]) %*% t(v[,100:460])  
img.compressed=imagematrix(img.compressed.mat, type = "grey")
plot(img.compressed)



#plot(1:1,type='n')
#plot(1,1,xlim=c(1,res[1]),ylim=c(1,res[2]),asp=1,type='n',xaxs='i',yaxs='i',xaxt='n',yaxt='n',xlab='',ylab='',bty='n')
#rasterImage(x,0,0,1,1)