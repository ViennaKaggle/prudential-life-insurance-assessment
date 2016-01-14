# Kaspar Sakmann, 2015
#
# Script for imputing missing values in the kaggle prudential data set using the mice package.
# All missing values are imputed using predictive mean matching (the default). They are drawn from a bootstrap 
# sample of the most similar entries. For details see the mice doocumentation.
# Imputation is performed on the training and the test data set. 
# In the form given the script imputes all missing values based on all other variables (except Medical_History_2)
# and writes 5 complete data sets to files. These can then be used like full data sets and the results averaged.
# 
# missing     :  contains the names of the columns that are to be imputed, please adjust.
# non_missing :  contains the names of the variables that are used to compute similar entries. Ideally these would 
#                be strongly correlated with the missing variables. please adjust.  
# nsets       :  number of imputed data sets
# maxit       :  number of iterations in the imputation

train <- read.csv("../data/train.csv",stringsAsFactors = T) #59,381 observations, 128 variables
test <- read.csv("../data/test.csv",stringsAsFactors = T) #19,765 observations, 127 variables, Response is missing here
Response<-train$Response  # save it for later
train$Response <- NULL
All_Data <- rbind(train,test) #79,146 observations, 129 variables 

#Define variables as either numeric or factor, Data_1 - Numeric Variables, Data_2 - factor variables
Data_1 <- All_Data[,names(All_Data) %in% c("Product_Info_4",  "Ins_Age",  "Ht",  "Wt",  "BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_10", "Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep=""))]
Data_2 <- All_Data[,!(names(All_Data) %in% c("Product_Info_4",	"Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_10", "Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep="")))]
Data_2<- data.frame(apply(Data_2, 2, as.factor))
All_Data <- cbind(Data_1,Data_2) #79,146 observations, 129 variables

# leave out Medical_History_2 for now as it has 628(!) levels
categoricals=c('Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 
               'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 
               'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 
               'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 
               'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 
               'Family_Hist_1', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 
               'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 
               'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 
               'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 
               'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 
               'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 
               'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 
               'Medical_History_40', 'Medical_History_41')


discrete_missing<-c('Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32');
continuous_missing<-c('Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5',
                      'Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5')

missing<-c(continuous_missing,discrete_missing)
non_missing<-c('Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', paste("Medical_Keyword_",1:48,sep=""),categoricals)

incomplete_data=All_Data[,names(All_Data) %in% missing]
str(incomplete_data)

library(mice)
nsets=5
mice_dat<-All_Data[,names(All_Data) %in% c(non_missing,missing)]

# impute the missing values using Multiple imputation by chained equations, m is the number of imputed datasets generated
imp<-mice(mice_dat,seed=42,visitSequence = 'monotone',m = nsets, maxit = 5)
save(imp,file = 'imp.RData')

# build the first complete data set, replacing the columns with missing values 
imp_ds <- All_Data
training_names <- paste(paste("imp_train_",1:nsets,sep=""),".csv",sep="")
test_names     <- paste(paste("imp_test_",1:nsets,sep=""),".csv",sep="")
ntrain <- nrow(train)
ntot   <- nrow(imp_ds)

for (i in 1:nsets) {
  # replace missing columns with those of the i-th imputed data set 
  imp_ds[missing] = complete(imp,action=i)[missing]

  # split the completed dataset in train and test and reinsert 'Response' in the training set
  imp_train <- imp_ds[c(1:ntrain),]
  imp_test  <- imp_ds[c((ntrain+1):ntot),]
  imp_train$Response<-Response

  # write imputed training and test set to files
  # note that the order of the columns is not the same as in the original
  write.table(imp_train, file = training_names[i],row.names=FALSE, na="",col.names=TRUE, sep=",")
  write.table(imp_test,  file = test_names[i],row.names=FALSE, na="",col.names=TRUE, sep=",")
}