setwd("~/Dropbox/prudential-life-insurance-assessment/R")

train <- read.csv("../data/train.csv",stringsAsFactors = T) #59,381 observations, 128 variables
test <- read.csv("../data/test.csv",stringsAsFactors = T) #19,765 observations, 127 variables - test does not have a response field

Response<-train$Response
train$Response <- NULL
  
#concatenate train and test together, any features we create will be on both data sets with the same code. This will make scoring easy
All_Data <- rbind(train,test) #79,146 observations, 129 variables 

#Define variables as either numeric or factor, Data_1 - Numeric Variables, Data_2 - factor variables
Data_1 <- All_Data[,names(All_Data) %in% c("Product_Info_4",  "Ins_Age",  "Ht",  "Wt",  "BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep=""))]
Data_2 <- All_Data[,!(names(All_Data) %in% c("Product_Info_4",	"Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep="")))]
Data_2<- data.frame(apply(Data_2, 2, as.factor))
All_Data <- cbind(Data_1,Data_2) #79,146 observations, 129 variables

discrete_missing<-c('Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32');
continuous_missing<-c('Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_History_2','Family_History_3','Family_History_4','Family_History_5')
missing<-c(continuous_missing,discrete_missing)
incomplete_data=All_Data[,names(All_Data) %in% missing]
str(incomplete_data)
non_missing<-c('Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI' )

library(mice)

mice_dat<-All_Data[,names(All_Data) %in% c(non_missing,missing)]
imp<-mice(mice_dat,seed=42)
save(imp,file = 'impute_missing_values.RData')

# first of the (default 5) imputed datasets
data_set_1<-complete(imp,1) 



