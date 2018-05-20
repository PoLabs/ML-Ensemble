setwd("~/Crypto/NANO")
# library(xgboost)
library(mxnet)

pcklibs <- c("dplyr", "caTools", "Metrics", "rpart", "RJSONIO", "pROC", "TTR", "caret", "caretEnsemble")#, "caret")#"httr",RCurl, zoo,caretEnsemble, data.table
lapply(pcklibs, require, character.only=TRUE)

set.seed(42)
mx.set.seed(42)
moneyp1 <- as.double(0.010)
moneyp6 <- as.double(0.006)
moneyp8 <- as.double(0.008)
moneyp5 <- as.double(0.005)

BinanceData.f <- fread(file="NANO-BTCdata.csv")
n5minObs <- nrow(BinanceData.f)
BinanceData.f <- BinanceData.f[complete.cases(BinanceData.f),]

#############################
#Prediction vectors
##############################
L5 <- data.frame(matrix(ncol=4, nrow=n5minObs-5))
L5[,1] <- BinanceData.f[6:n5minObs,4]
colnames(L5) <- c("lag5close", "lag5percent", "A1", "A2")
L5$lag5percent <- (L5$lag5close - BinanceData.f[1:(n5minObs-5),4]) / BinanceData.f[1:(n5minObs-5),4]
L5$A1 <- ifelse (L5$lag5percent > moneyp5, 1, 0)
L5$A2 <- ifelse (L5$lag5percent < 0-moneyp5, 1, 0)

L10 <- data.frame(matrix(ncol=4, nrow=n5minObs-10))
L10[,1] <- BinanceData.f[11:n5minObs,4]
colnames(L10) <- c("lag10close", "lag10percent", "A3", "A4")
L10$lag10percent <- (L10$lag10close - BinanceData.f[1:(n5minObs-10),4]) / BinanceData.f[1:(n5minObs-10),4]
L10$A3 <- ifelse (L10$lag10percent > moneyp5, 1, 0)
L10$A4 <- ifelse (L10$lag10percent < 0-moneyp5, 1, 0)

L15 <- data.frame(matrix(ncol=4, nrow=n5minObs-15))
L15[,1] <- BinanceData.f[16:n5minObs,4]
colnames(L15) <- c("lag15close", "lag15percent", "A5", "A6")
L15$lag15percent <- (L15$lag15close - BinanceData.f[1:(n5minObs-15),4]) / BinanceData.f[1:(n5minObs-15),4]
L15$A5 <- ifelse (L15$lag15percent > moneyp5, 1, 0)
L15$A6 <- ifelse (L15$lag15percent < 0-moneyp5, 1, 0)

L20 <- data.frame(matrix(ncol=4, nrow=n5minObs-20))
L20[,1] <- BinanceData.f[21:n5minObs,4]
colnames(L20) <- c("lag20close", "lag20percent", "A7", "A8")
L20$lag20percent <- (L20$lag20close - BinanceData.f[1:(n5minObs-20),4]) / BinanceData.f[1:(n5minObs-20),4]
L20$A7 <- ifelse (L20$lag20percent > moneyp5, 1, 0)
L20$A8 <- ifelse (L20$lag20percent < 0-moneyp5, 1, 0)

L30 <- data.frame(matrix(ncol=4, nrow=n5minObs-30))
L30[,1] <- BinanceData.f[31:n5minObs,4]
colnames(L30) <- c("lag30close", "lag30percent", "A9", "A10")
L30$lag30percent <- (L30$lag30close - BinanceData.f[1:(n5minObs-30),4]) / BinanceData.f[1:(n5minObs-30),4]
L30$A9 <- ifelse (L30$lag30percent > moneyp5, 1, 0)
L30$A10 <- ifelse (L30$lag30percent < 0-moneyp5, 1, 0)


save.image(file='BTCenv.RData')


#Algorithms:      #6869
###############################
#- log regression: A1-6~ . 
###############################
#1. build new df with BinanceData and future vector 
A1.df <- cbind(L5$A1, BinanceData.f[1:(nrow(BinanceData.f)-5),])
names(A1.df)[1] <- "A1"
A1model <- glm(A1 ~ ., data=A1.df, family="binomial")
A1.df <- A1.df[sample(nrow(A1.df)),]
A2.df <- cbind(L5$A2, BinanceData.f[1:(nrow(BinanceData.f)-5),])
names(A2.df)[1] <- "A2"
A2model <- glm(A2 ~ ., data=A2.df, family="binomial")
A2.df <- A2.df[sample(nrow(A2.df)),]
A3.df <- cbind(L10$A3, BinanceData.f[1:(nrow(BinanceData.f)-10),])
names(A3.df)[1] <- "A3"
A3model <- glm(A3 ~ ., data=A3.df, family="binomial")
A3.df <- A3.df[sample(nrow(A3.df)),]
A4.df <- cbind(L10$A4, BinanceData.f[1:(nrow(BinanceData.f)-10),])
names(A4.df)[1] <- "A4"
A4model <- glm(A4 ~ ., data=A4.df, family="binomial")
A4.df <- A4.df[sample(nrow(A4.df)),]
A5.df <- cbind(L15$A5, BinanceData.f[1:(nrow(BinanceData.f)-15),])
names(A5.df)[1] <- "A5"
A5model <- glm(A5 ~ ., data=A5.df, family="binomial")
A5.df <- A5.df[sample(nrow(A5.df)),]
A6.df <- cbind(L15$A6, BinanceData.f[1:(nrow(BinanceData.f)-15),])
names(A6.df)[1] <- "A6"
A6model <- glm(A6 ~ ., data=A6.df, family="binomial")
A6.df <- A6.df[sample(nrow(A6.df)),]
A7.df <- cbind(L20$A7, BinanceData.f[1:(nrow(BinanceData.f)-20),])
names(A7.df)[1] <- "A7"
A7model <- glm(A7 ~ ., data=A7.df, family="binomial")
A7.df <- A7.df[sample(nrow(A7.df)),]
A8.df <- cbind(L20$A8, BinanceData.f[1:(nrow(BinanceData.f)-20),])
names(A8.df)[1] <- "A8"
A8model <- glm(A8 ~ ., data=A8.df, family="binomial")
A8.df <- A8.df[sample(nrow(A8.df)),]
A9.df <- cbind(L30$A9, BinanceData.f[1:(nrow(BinanceData.f)-30),])
names(A9.df)[1] <- "A9"
A9model <- glm(A9 ~ ., data=A9.df, family="binomial")
A9.df <- A9.df[sample(nrow(A9.df)),]
A10.df <- cbind(L30$A10, BinanceData.f[1:(nrow(BinanceData.f)-30),])
names(A10.df)[1] <- "A10"
A10model <- glm(A10 ~ ., data=A10.df, family="binomial")
A10.df <- A10.df[sample(nrow(A10.df)),]

A1.df <- A1.df[complete.cases(A1.df),]
A2.df <- A2.df[complete.cases(A2.df),]
A3.df <- A3.df[complete.cases(A3.df),]
A4.df <- A4.df[complete.cases(A4.df),]
A5.df <- A5.df[complete.cases(A5.df),]
A6.df <- A6.df[complete.cases(A6.df),]
A7.df <- A7.df[complete.cases(A7.df),]
A8.df <- A8.df[complete.cases(A8.df),]
A9.df <- A9.df[complete.cases(A9.df),]
A10.df <- A10.df[complete.cases(A10.df),]

# A1train <- A1.df[2000:nrow(A1.df),]
# A1test <- A1.df[1:2000,]


################################################
#XGB w caret
##################################################### 1,4,9,10,11,12
param <- list("objective"="binary:logistic", "nthread"=16, "verbose"=1)
xgb_grid_A6 = expand.grid(
  nrounds = c(1200),#, lambda=0, alpha=0)   
  eta = c(0.01),
  max_depth = c(10),
  colsample_bytree = c(0.1),# 0.9),
  min_child_weight = c(1), subsample = c(0.9), gamma = c(1))

xgb_trcontrol_A1 = trainControl(method = "cv",  number = 5,  verboseIter = TRUE,  returnData = FALSE, trim=TRUE, returnResamp = "all", allowParallel = FALSE, sampling = "up",                                                        
  classProbs = TRUE, summaryFunction = twoClassSummary)

y = (A6.df[,1])
XGBtrain = data.matrix(A6.df[,-1])
yf <- y
yf <- ifelse(y == 0, "no", "yes") 
yf <- factor(yf)
xgb_train_A6_up = train(x = XGBtrain, y = yf, trControl = xgb_trcontrol_A1, tuneGrid = xgb_grid_A3, method = "xgbTree", preProcess = c("center", "scale"))
#################################

###############################
#- mxnet nn 
###############################
mx.set.seed(42)
train.x = data.matrix(A9.df[,2:(ncol(A9.df))]) 
train.y = A9.df[,1]
mxnet_grid_A1 = expand.grid(layer1 = c(8),   #layer1, layer2, layer3, learning.rate, momentum, dropout, activation
                            layer2 = c(4),
                            layer3 = c(0),
                            learningrate = c(0.0001),
                            dropout = c(0),
                            beta1 = .9, beta2 = 0.999, activation = 'relu')

mxnet_trcontrol_A1 = trainControl(method = "cv",  number = 4,  verboseIter = TRUE, returnData = FALSE, returnResamp = "all", trim=TRUE, classProbs = TRUE,    
  savePredictions = TRUE,  summaryFunction = twoClassSummary,  allowParallel = TRUE,  sampling="smote")

train.yf <- train.y
train.yf <- ifelse(train.y == 0, "no", "yes") 
train.yf <- factor(train.yf)

mxnet_train_A9 = caret::train(
  x = train.x,  y = train.yf,  trControl = mxnet_trcontrol_A1, tuneGrid = mxnet_grid_A9, preProcess=c("center", "scale"),#tuneLength=5,
  method = "mxnetAdam",  metric = "ROC", ctx=mx.gpu(), num.round=1200)

save.image(file='NANOenvMX.RData')

##########################################################

#################
#SVM tune grid
svm_trcontrol_A4 = trainControl(  method = "cv",  number = 2, verboseIter = TRUE,  returnData = TRUE,  returnResamp = "all",  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
   summaryFunction = twoClassSummary,  allowParallel = TRUE,  sampling = "up")
#SVM-RBF tuner
svmrbf_grid_A4 = expand.grid(C=c(2), sigma=c(0.01)) #method = 'svmRadial'
svmrbf_train_A4 = caret::train(x = XGBtrain,  y = yf,  trControl = svm_trcontrol_A4,  tuneGrid = svmrbf_grid_A4, #tuneLength = 10,
  method = "svmRadial",  metric = "ROC")

###################
#random forest tune grid
rf_grid_A4 = expand.grid(mtry = c(2))
rf_trcontrol_A4 = trainControl(  method = "cv",  number = 2,  verboseIter = TRUE,  returnData = TRUE, returnResamp = "all",
  classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = TRUE, sampling = "up")
rf_train_A4 = caret::train(  x = XGBtrain,  y = yf,  trControl = rf_trcontrol_A4,  tuneGrid = rf_grid_A4,  #tuneLength = 10,
  method = "rf",  metric = "ROC")

#A1
############################################################
Ensemble_control_A1 <- trainControl(
  method = "repeatedcv",  number = 4,  repeats = 3,  verboseIter = TRUE,  returnData = FALSE, trim=TRUE, returnResamp = "all", 
  classProbs = TRUE, summaryFunction = twoClassSummary,  savePredictions = TRUE,  allowParallel = TRUE,  sampling = "smote")

yE = A10.df[,1]
xE = data.matrix(A10.df[,-1])
yf <- yE
yEf <- ifelse(yE == 0, "no", "yes") 
yEf <- factor(yEf)

Ensemble_list_A10sm <- caretList(preProcess=c("center", "scale"),  x=xE,  y=yEf,  trControl=Ensemble_control_A1,  metric="ROC",  methodList=c("glm", "rpart", "bayesglm"),
  tuneList=list(mxA2=caretModelSpec(method="mxnetAdam", tuneGrid=mxnet_grid_A1, num.round=1000, ctx=mx.gpu())))

Ensemble_greedy_A10sm <- caretEnsemble(Ensemble_list_A10sm, metric="ROC",
  trControl=trainControl(number=4, summaryFunction=twoClassSummary, trim=TRUE, returnData=FALSE, classProbs=TRUE ))
summary(Ensemble_greedy_A10sm)

save.image(file='NANOenv-Ens.RData')

#Save-predict
###########################
#saveRDS(xgb_train_A5_up, "Action/A5-caretXGB.rds")
#saveRDS(mxnet_train_A2_smote, "Action/A2-caretMX.rds")
saveRDS(Ensemble_greedy_A1sm, "A1sm-NANO.rds")
saveRDS(Ensemble_greedy_A2sm, "A2sm-NANO.rds")
saveRDS(Ensemble_greedy_A3sm, "A3sm-NANO.rds")
saveRDS(Ensemble_greedy_A4sm, "A4sm-NANO.rds")
saveRDS(Ensemble_greedy_A5sm, "A5sm-NANO.rds")
saveRDS(Ensemble_greedy_A6sm, "A6sm-NANO.rds")



saveRDS(Ensemble_greedy_A7sm, "A7sm-NANO.rds")
saveRDS(Ensemble_greedy_A8sm, "A8sm-NANO.rds")
saveRDS(Ensemble_greedy_A9sm, "A9sm-NANO.rds")
saveRDS(Ensemble_greedy_A10sm, "A10sm-NANO.rds")


A5.pred <- predict(Ensemble_greedy_A5sm, as.matrix(A5test[,-1]))
A5.predB<- ifelse(A5.pred=="yes",1,0) 
sum(A5test[,1])
sum(A5.predB)
wrong <- ifelse(A5test[,1] != A5.predB,1,0)
sum(wrong)

save.image(file='NEOenvEns1.RData')



