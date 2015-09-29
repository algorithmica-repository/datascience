#Problem 1
#Q1
rain.df = read.table("E:/data analytics/datasets/rainfall.dat")
#Q2
dim(rain.df)[1] # for number of rows
#Q3
dim(rain.df)[2]
#Q4
colnames(rain.df)
#Q5
rain.df[2,4]
#Q6
rain.df[2, ]
#Q7
names(rain.df) = c("year", "month", "day", seq(0,23))
str(rain.df)
#Q8
rain.df$daily = apply(sum, X=rain.df[,4:27], MARGIN=1)
head(rain.df)
tail(rain.df)

#rain.df = rain.df %>% rowwise() %>% mutate(daily=sum(0,1))
#rain.df = rain.df %>% group_by(1:n()) %>% mutate(daily=sum(0:23))
#rain.df[20,]

#Problem 2

# Q1
DeepSea <- read.table(file ="E:/data analytics/datasets/ISIT.txt", header = TRUE)

# Q2
names(DeepSea)
str(DeepSea)
dim(DeepSea)
head(DeepSea)

# Q3
unique(DeepSea$Station)

# Q4
DeepSea.sta1 = DeepSea[DeepSea$Station==1,]
DeepSea.sta1

DeepSea.sta1 = DeepSea %>% filter(Station==1)
DeepSea.sta1

# Q5
nrow(DeepSea.sta1)
min(DeepSea.sta1$SampleDepth)
mean(DeepSea.sta1$SampleDepth)
max(DeepSea.sta1$SampleDepth)

DeepSea.sta1 %>% summarise(total=n(),min_depth=min(SampleDepth), mean_depth=mean(SampleDepth), max_depth=max(SampleDepth))

# Q6
nrow(DeepSea[DeepSea$Station==1,])
nrow(DeepSea[DeepSea$Station==2,])
nrow(DeepSea[DeepSea$Station==3,])
nrow(DeepSea[DeepSea$Station==4,])
nrow(DeepSea[DeepSea$Station==5,])

DeepSea %>% filter(Station==1 | Station==2 | Station==3 | Station==4 | Station==5) %>% group_by(Station) %>% summarize(count=n())

# Q7
DeepSea %>% group_by(Station) %>% summarize(count=n()) %>% filter(min_rank(count)>2)


# Q8
DeepSea.fall = DeepSea[DeepSea$Month==8 | DeepSea$Month==9 | DeepSea$Month==10, ]
nrow(DeepSea.fall)

DeepSea %>% filter(Month==8 | Month==9 | Month==10) %>% summarise(count=n())

# Q9
DeepSea.dep2000 = DeepSea[DeepSea$SampleDepth>2000, ]
nrow(DeepSea.dep2000)

DeepSea.dep2000 = DeepSea %>% filter(SampleDepth>2000) %>% summarise(count=n())

# Q10
DeepSea.dep2000.fall2001 <- DeepSea[DeepSea$SampleDepth>2000 & (DeepSea$Month==8 | DeepSea$Month==9 | DeepSea$Month==10) & DeepSea$Year==2001, ]
nrow(DeepSea.dep2000.fall2001)

DeepSea %>% filter(SampleDepth>2000 & (Month==8 | Month==9 | Month==10) & Year==2001) %>% summarise(count=n())


#Problem 3

# Step 1
DeepSea1 = read.table(file ="E:/data analytics/datasets/DeepSea1.txt", header = TRUE)

# Step 2
DeepSea2 = read.table(file ="E:/data analytics/datasets/DeepSea2.txt", header = TRUE)

# Step 3
head(DeepSea1)
head(DeepSea2)

# Step 4
DeepSea <- merge(DeepSea1, DeepSea2, by.x = 'ID', by.y = 'SampleID')
nrow(DeepSea)
nrow(DeepSea1)
nrow(DeepSea2)

DeepSea=inner_join(DeepSea1, DeepSea2, by=c('ID'='SampleID'))
nrow()

# Step 5
DeepSea.full = merge(DeepSea1, DeepSea2, by.x = 'ID', by.y = 'SampleID', all = TRUE)
nrow(DeepSea.full)
head(DeepSea.full)

nrow(right_join(DeepSea1, DeepSea2, by=c('ID'='SampleID')))

# Step 6
DeepSea[,c('Year','Month','Station','SampleDepth')]

DeepSea %>% select(Year,Month,Station,SampleDepth)

# Step 7
DeepSea[order(DeepSea$SampleDepth), c('Year','Month','Station','SampleDepth')]

DeepSea %>% select(Year,Month,Station,SampleDepth) %>% arrange(SampleDepth)

# Step 8
DeepSea[order(DeepSea$Station,-DeepSea$SampleDepth), c('Year','Month','Station','SampleDepth')]

DeepSea %>% select(Year,Month,Station,SampleDepth) %>% arrange(Station,desc(SampleDepth))

# Step 9
DeepSea$fYear <- factor(DeepSea$Year)
DeepSea$fMonth <- factor(DeepSea$Month)

DeepSea = DeepSea %>% mutate(fYear=factor(Year), fMonth=factor(Month))
str(DeepSea)

# Step 10
levels(DeepSea$fYear)
levels(DeepSea$fMonth)

# Step 11
DeepSea$fMonthName <- factor(DeepSea$Month, levels=c(3,4,8,10), labels = c('March','April','August','October'))

DeepSea = DeepSea %>% mutate(fMonthName=factor(DeepSea$Month, levels=c(3,4,8,10), labels = c('March','April','August','October')))
str(DeepSea)


# Step 12
write.table(DeepSea[,c('ID','Year','Month','Station','SampleDepth','fYear','fMonth','fMonthName')], file = "DeepSea.txt", sep="\t", quote = TRUE, append = FALSE, na = "NA", row.names = FALSE)

res=DeepSea %>% select(ID, Year, Month, Station, SampleDepth, fYear, fMonth, fMonthName)
write.table(res, file = "DeepSea.txt", sep="\t", quote = TRUE, append = FALSE, na = "NA", row.names = FALSE)


#Problem 3
# Step 1
setwd('E:/data analytics/datasets')
Temp = read.table(file = "Temperature.txt", header = TRUE)

# Step 2
names(Temp)
str(Temp)
dim(Temp)
head(Temp)

# Step 3
tapply(Temp$Temperature, INDEX = Temp$Month, FUN = mean, na.rm = TRUE)
tapply(Temp$Temperature, INDEX = Temp$Month, FUN = sd, na.rm = TRUE)

Temp %>% group_by(Month) %>% summarise(temp_mean=mean(Temperature, na.rm=TRUE), temp_sd=sd(Temperature,na.rm=TRUE))

# Step 4
tapply(Temp$Temperature, INDEX = list(Temp$Month,Temp$Station), FUN = mean, na.rm = TRUE)
tapply(Temp$Temperature, INDEX = list(Temp$Month,Temp$Station), FUN = sd, na.rm = TRUE)

Temp %>% group_by(Month,Station) %>% summarise(temp_mean=mean(Temperature, na.rm=TRUE), temp_sd=sd(Temperature,na.rm=TRUE))


# Step 5
Temp.1990 <- Temp[Temp$Year == 1990 , ]
tapply(Temp.1990$Temperature, INDEX = Temp.1990$Month, FUN=mean, na.rm = TRUE)
tapply(Temp.1990$Temperature, INDEX = Temp.1990$Month, FUN=sd, na.rm = TRUE)

Temp %>% filter(Year==1990) %>% group_by(Month) %>% summarise(temp_mean=mean(Temperature, na.rm=TRUE), temp_sd=sd(Temperature,na.rm=TRUE))


# Step 6
sapply(Temp[, c('Salinity','Temperature','CHLFa')], FUN = mean, na.rm = TRUE)
sapply(Temp[, c('Salinity','Temperature','CHLFa')], FUN = sd, na.rm = TRUE)

Temp %>% summarise(temp_mean=mean(Salinity, na.rm=TRUE), temp_sd=sd(Salinity,na.rm=TRUE))
Temp %>% summarise(temp_mean=mean(Temperature, na.rm=TRUE), temp_sd=sd(Temperature,na.rm=TRUE))
Temp %>% summarise(temp_mean=mean(CHLFa, na.rm=TRUE), temp_sd=sd(CHLFa,na.rm=TRUE))

Temp %>% select(Salinity,Temperature,CHLFa) %>% summarise_each(funs(mean(.,na.rm=TRUE),sd(.,na.rm=TRUE)))

# Step 7
Temp.dant <- Temp[Temp$Station == 'DANT' , ]
sapply(Temp.dant[, c('Salinity','Temperature','CHLFa')], FUN = mean, na.rm = TRUE)
sapply(Temp.dant[, c('Salinity','Temperature','CHLFa')], FUN = sd, na.rm = TRUE)

Temp %>% filter(Station=='DANT') %>% select(Salinity,Temperature,CHLFa) %>% summarise_each(funs(mean(.,na.rm=TRUE),sd(.,na.rm=TRUE)))


# Step 8
summary(Temp[, c('Salinity','Temperature','CHLFa')])

Temp %>% select(Salinity,Temperature,CHLFa) %>% summary()

# Step 9
table(Temp$Station)
table(Temp$Year)
table(Temp$Station, Temp$Year)



