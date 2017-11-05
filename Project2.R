#download.file("https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/gender_submission.csv?sv=2015-12-11&sr=b&sig=CvTQuyAA32lTpPxTZbFhmvrPDgzGoU6owd0d3NJAs%2Bo%3D&se=2017-05-17T10%3A15%3A23Z&sp=r","/Users/hangyu/Documents/RProjects/gender.csv")
#download.file("https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/test.csv?sv=2015-12-11&sr=b&sig=e%2BGdffylTNViFa%2BShj2OWr1FGWL6o9Zeq%2F%2BWjSCTAtc%3D&se=2017-05-17T10%3A21%3A48Z&sp=r","/Users/hangyu/Documents/RProjects/test.csv")
#download.file("https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/train.csv?sv=2015-12-11&sr=b&sig=iS7OjFl4SscI6L1lKJDi%2FSgBOdKBnaKD9CwnITJHFm4%3D&se=2017-05-17T10%3A22%3A53Z&sp=r","/Users/hangyu/Documents/RProjects/train.csv")
library(caret)
library(DMwR)
trainDT <- read.csv("train.csv")
trainDT$Fare <- as.numeric(trainDT$Fare)
testDT <- read.csv("test.csv")
gender <- read.csv("gender.csv")
head(trainDT)
head(testDT)
head(gender)
summary(trainDT)
table(trainDT$Sex,trainDT$Survived)
prop.table(table(trainDT$Sex,trainDT$Survived),1)
summary(trainDT$Age)
plot(trainDT)
########################feature engineering#############################
trainDT$Title <- NA
trainDT$Name <- as.character(trainDT$Name)
trainDT$Title <- sapply(trainDT$Name,function(x){strsplit(x,"[,.]")[[1]][2]})
trainDT$family <- trainDT$SibSp + trainDT$Parch

testDT$Title <- NA
testDT$Name <- as.character(testDT$Name)
testDT$Title <- sapply(testDT$Name,function(x){strsplit(x,"[,.]")[[1]][2]})
testDT$family <- testDT$SibSp + testDT$Parch
################data imputation using knn#############################
ImputeModel1 <- preProcess(trainDT,method = "knnImpute",k=5)
trainDTImpute <- predict(ImputeModel1,trainDT)
ImputeModel2 <- preProcess(testDT,method = "knnImpute",k=5)
testDTImpute <- predict(ImputeModel2,testDT)

trainDTImpute$Name <- as.factor(trainDTImpute$Name)
trainDTImpute$Title <- as.factor(trainDTImpute$Title)
testDTImpute$Name <- as.factor(testDTImpute$Name)
testDTImpute$Title <- as.factor(testDTImpute$Title)
#####################EDA look for features############################
aggregate(Age~Survived,trainDT,mean)
boxplot(Age~Survived,data = trainDT)
aggregate(Survived~Sex,trainDT,mean)#sex is one
aggregate(Survived~Pclass,trainDT,mean)#Pclass is one
aggregate(Fare~Survived,trainDT,mean)#Pclass is one
#explore fare vs survived
#first divide fare into regions
qt1 <- quantile(subset(trainDT,Pclass==1)$Fare)
qt2 <- quantile(subset(trainDT,Pclass==2)$Fare)
qt3 <- quantile(subset(trainDT,Pclass==3)$Fare)
trainDT$FareLab <- "high"
trainDT[which(trainDT$Pclass==1 & trainDT$Fare < qt1[2]),]$FareLab <- "low"
trainDT[which(trainDT$Pclass==1 & trainDT$Fare >= qt1[2] & trainDT$Fare < qt1[3]),]$FareLab <- "medium"
trainDT[which(trainDT$Pclass==1 & trainDT$Fare >= qt1[3]),]$FareLab <- "high"

with(aggregate(Survived~FareLab,subset(trainDT,Pclass==1),mean),plot(as.factor(FareLab),Survived,type="l"))
#higher price higher chance to survive
##################prediction######################################
modelrpart <- train(Survived~Sex+Pclass+Fare+Age+Title+family,data=trainDTImpute,method="rpart")
modelrf <- train(Survived~Sex+Pclass+Fare+Age+Title+family,data=trainDTImpute,method="rf")
#modelrf <- randomForest(Survived~Sex+Pclass+Fare+Age+Title+family,data = trainDTImpute,ntree=400,importance=TRUE)
modelcif <- train(Survived~Sex+Pclass+Fare+Age+Title+family,data=trainDTImpute,method="cforest")
cforest <- ctree(Survived~Sex+Pclass+Fare+Age+Title+family,data=trainDTImpute)
plot(cforest)

varImpPlot(modelrf$finalModel)
varImp(modelcif$finalModel)
png("tree.png")
plot(modelrpart$finalModel)
text(modelrpart$finalModel)
dev.off

levels <- levels(testDTImpute$Title)
levels[length(levels) + 1] <- " Don"
testDTImpute$Title <- factor(testDTImpute$Title, levels = levels)
testDTImpute[415,]$Title <- " Don"

pred <- predict(modelcif,newdata = testDTImpute)
pred <- ifelse(pred>=0,1,0)
testDT$Survived <- pred
submit <- data.frame(PassengerId = testDT$PassengerId, Survived = pred)
write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)
