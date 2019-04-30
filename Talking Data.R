setwd("C:\\Users\\Ravi Singh\\Desktop")
library(xgboost)
library(pryr)
library(caTools)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, tidyverse, highcharter, data.table, lubridate, pROC, tictoc, DescTools, lightgbm)
set.seed(84)               
options(scipen = 9999, warn = -1, digits= 5)
data<-fread("train_sample.csv")
str(data)

tic("Total processing time for feature engineering on training data --->")
train <- fread("train_sample.csv", nrows=90000, 
               col.names =c("ip", "app", "device", "os", "channel", "click_time", 
                            "attributed_time", "is_attributed"), , 
               showProgress = FALSE) %>%
  select(-c(attributed_time)) %>%
  mutate(wday = Weekday(click_time), hour = hour(click_time)) %>% 
  select(-c(click_time)) %>%
  add_count(ip, wday, hour) %>% rename("nip_day_h" = n) %>%
  add_count(ip, hour, channel) %>% rename("nip_h_chan" = n) %>%
  add_count(ip, hour, os) %>% rename("nip_h_osr" = n) %>%
  add_count(ip, hour, app) %>% rename("nip_h_app" = n) %>%
  add_count(ip, hour, device) %>% rename("nip_h_dev" = n) %>%
  select(-c(ip))
toc()
str(train)
tic("Total processing time for feature engineering on testing data --->")
test <- fread("train_sample.csv",skip =90000 , nrows=10000, 
               col.names =c("ip", "app", "device", "os", "channel", "click_time", 
                            "attributed_time", "is_attributed"), , 
               showProgress = FALSE) %>%
  select(-c(attributed_time)) %>%
  mutate(wday = Weekday(click_time), hour = hour(click_time)) %>% 
  select(-c(click_time)) %>%
  add_count(ip, wday, hour) %>% rename("nip_day_h" = n) %>%
  add_count(ip, hour, channel) %>% rename("nip_h_chan" = n) %>%
  add_count(ip, hour, os) %>% rename("nip_h_osr" = n) %>%
  add_count(ip, hour, app) %>% rename("nip_h_app" = n) %>%
  add_count(ip, hour, device) %>% rename("nip_h_dev" = n) %>%
  select(-c(ip))
toc()
str(test)
kable(as.data.frame(lapply(train,function(x)length(unique(x)))))
kable(as.data.frame(lapply(test,function(x)length(unique(x)))))

kable(table(train$is_attributed))
kable(table(test$is_attributed))
h1<-train %>% group_by(app) %>%summarise(count=n())%>% arrange(desc(count))%>% head(15)%>%mutate(app=as.character(app))%>%hchart("bar", hcaes(x = app, y = count, color =-count)) %>%hc_add_theme(hc_theme_ffx()) %>% hc_title(text = "Top Apps")                                                                                                 
h1
h1 <- train %>% filter(is_attributed == 1) %>% group_by(app) %>% summarise(count = n()) %>% 
  arrange(desc(count)) %>% head(20) %>% mutate(app = as.character(app)) %>%
  hchart("bar", hcaes(x = app, y = count, color =-count)) %>%
  hc_add_theme(hc_theme_ffx()) %>% hc_title(text = "Top Apps")
h1

kable(table(train$is_attributed))

#tr_index <- nrow(train)
#train <- train %>% head(0.95 * tr_index) # 95% data for training
#valid <- train %>% tail(0.05 * tr_index) # 5% data for validation
#rm(train)
str(train)

sample = sample.split(train$is_attributed, SplitRatio = .9)
dtrain =subset(train , sample==TRUE)
valid = subset(train , sample==FALSE)
rm(train,sample)
table(dtrain$is_attributed)
table(valid$is_attributed)
dim(dtrain)
dtrain<- xgb.DMatrix(as.matrix(dtrain[,colnames(dtrain)!="is_attributed"]),label=dtrain$is_attributed)
head(dtrain)
dtrain
str(dtrain)
mem_used()
dvalid<-xgb.DMatrix(as.matrix(valid[,colnames(valid)!="is_attributed"]),label=valid$is_attributed)
str(dvalid)
mem_used()

params <- list(objective= "binary:logistic",
               grow_policy = "lossguide",
               tree_method="auto",
               eval_metric = "auc",
               max_leaves = 7,
               max_delta_step=7,
               scale_pos_weight=9.7,
               eta=0.1,
               max_depth=4,
               subsample=0.9,
               min_child_weight=0,
               colsample_bytree=0.7,
               random_state=84
               )
tic("total time for modelling")
xgb.model<-xgb.train(data=dtrain,params=params,maximize=TRUE,silent=1,watchlist = list(valid=dvalid),
                     nthread=4,nrounds = 1000,print_every_n = 50,early_stopping_rounds = 50)
toc()

xgb.model$best_ntreelimit
xgb.model$best_score
xgb.model$params
head(test)
test<-test%>% select(-c(is_attributed))
str(test)
dtest <-xgb.DMatrix(as.matrix(test[,colnames(test)]))
pred <-predict(xgb.model,newdata = dtest,ntreelimit = xgb.model$best_ntreelimit)

preds<-as.data.frame(pred)
preds
sub$is_attributed= round(preds,5)
head(sub,10)
kable(xgb.importance(model=xgb.model))

