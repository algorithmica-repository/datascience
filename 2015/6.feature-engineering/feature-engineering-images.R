library(pixmap)
library(caret)

setwd("E:\\data analytics\\datasets\\att_faces")

files=list.files(pattern="*.pgm", recursive = TRUE)
faces = matrix(nrow=length(files), ncol=112*92)
for(i in 1:length(files)) {
  gray_file = read.pnm(paste(getwd(),files[i],sep="/"))
  faces[i,] = gray_file@grey
}
dim(faces)

img = pixmapGrey(faces[1,],112,92)
plot(img)

pca = preProcess(faces, method=c("pca"),pcaComp=300)
pca
pca$rotation
faces_reduced = predict(pca,faces)
dim(faces_reduced)
