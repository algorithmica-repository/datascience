f1 = factor(c(rep("M",5),rep("F",5))
f2 = factor(c(rep("M",5),rep("F",5))
              ,levels = c("M","F"))
f1[1] = "F"
as.integer(f1)
as.integer(f2)
f1[1] == "F"
levels(f1)
                        