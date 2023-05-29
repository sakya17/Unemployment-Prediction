##################### Unemployment Analysis Project #########################
####Team 2
####Sakyarshi Kurati - 33.3%
####Marco Portillo - 33.35
####Mounir Cheraki - 33.3%


## imported the preprocessed data
method1 <- read.csv("Unemp_Combined_method_1.csv")

#### STR
str(method1)

###### Removing row names
method1 <- method1[-c(1,22)]

########Changing the date in the format used for regression ###########

method1$DATE <- as.Date(method1$DATE,format = "%m/%d/%Y")


###Splitting the data into training and testing sets.
set.seed(800)
c_rs <- sample(nrow(method1), nrow(method1)*0.25)
complete_data_train <- method1[-c_rs,]
complete_data_test <- method1[c_rs,]
complete_data_test2 <- complete_data_test[-6]

###############Regression Model ############################

############With all the variables##########

c_reg_model <- lm(Unemp_rate ~., data = complete_data_train)
summary(c_reg_model)

#### Prediction #####

c_predictions <- predict(c_reg_model, newdata = complete_data_test2)
c_predictions2 <- predict(regression_model10, newdata = complete_data_test2)

check <- data.frame(c_predictions, complete_data_test$Unemp_rate)

#############backward elimination##############
##############Removing insignificant varibales ####################

regression_model1 <- lm(Unemp_rate ~. -CLI, data = complete_data_train)
summary(regression_model1)

regression_model2 <- lm(Unemp_rate ~.-stock_prices -CLI, data = complete_data_train)
summary(regression_model2)

regression_model3 <- lm(Unemp_rate ~.-stock_prices -CLI -CCI , data = complete_data_train)
summary(regression_model3)

regression_model4 <- lm(Unemp_rate ~. -CCI -CLI -GDP -stock_prices, data = complete_data_train)
summary(regression_model4)


regression_model5 <- lm(Unemp_rate ~. -CCI -CLI -GDP -stock_prices -PPI, 
                        data = complete_data_train)
summary(regression_model5)

regression_model6 <- lm(Unemp_rate ~. -CCI -CLI -GDP -stock_prices  -Businees_CI, 
                        data = complete_data_train)
summary(regression_model6)

regression_model7 <- lm(Unemp_rate ~. -CCI -CLI -stock_prices -Import_goods -Businees_CI -GDP, 
                        data = complete_data_train)
summary(regression_model7)

regression_model8 <- lm(Unemp_rate ~. -CCI -CLI -stock_prices -DATE  -Businees_CI -GDP -Import_goods,
                        data = complete_data_train)
summary(regression_model8)

regression_model9 <- lm(Unemp_rate ~. -CCI -CLI -stock_prices -DATE -Businees_CI -GDP - Import_goods -Export_goods,
                        data = complete_data_train)
summary(regression_model9)

########## the f statistic is reducing so we will stick to regression 8

regression_model11 <- lm(Unemp_rate ~. -CCI -CLI -stock_prices -DATE -Businees_CI -Unemp_avg_weeks -GDP - Import_goods -Export_goods,
                        data = complete_data_train)
summary(regression_model11)

regression_model11 <- lm(Unemp_rate ~. -CCI -CLI -stock_prices -DATE -Businees_CI -Unemp_avg_weeks 
                         -GDP - Import_goods -Net_Trade_goods -Export_goods,
                         data = complete_data_train)
summary(regression_model11)

#################### Evaluation ###################

#install.packages("Metrics")
library(Metrics)
c_mae <- mae(c_predictions2, complete_data_test$Unemp_rate)
c_mse <- mse(c_predictions2, complete_data_test$Unemp_rate)
c_rmse <- rmse(c_predictions2, complete_data_test$Unemp_rate)
c_rsq <- cor(c_predictions2, complete_data_test$Unemp_rate)^2

print(paste("MAE-", c_mae, "/// MSE-", c_mse, "/// RMSE-", c_rmse, "/// RSQ-", c_rsq))

pairs(Unemp_rate ~., data = complete_data_train)

######## Decision tree regression#################
library(caret)
library(rpart)
library(rpart.plot)
rpart.plot(reg_tree,box.palette = "RdBu", shadow.col = "grey", type = 4, extra = 1)
varImp(reg_tree)

reg_tree <- rpart(Unemp_rate ~., data = complete_data_train, method = "anova", cp = 0.0001)

reg_tree$variable.importance
reg_tree_pred <- predict(reg_tree, complete_data_test2)

check2 <- data.frame(reg_tree_pred, complete_data_test$Unemp_rate)

######### Evaluation #############
mae3 <- mae(reg_tree_pred, complete_data_test$Unemp_rate)
mse3 <- mse(reg_tree_pred, complete_data_test$Unemp_rate)
rmse3 <- rmse(reg_tree_pred, complete_data_test$Unemp_rate)
rsq3 <- cor(reg_tree_pred, complete_data_test$Unemp_rate)^2
print(paste("MAE-", mae3, "///MSE-", mse3, "///RMSE-", rmse3, "///RSQ-", rsq3))

printcp(reg_tree)

############Random Forest################
#install.packages("randomForest")
library(randomForest)

set.seed(1200)
rf_model <- randomForest(Unemp_rate~., data = complete_data_train, ntree = 100, mtry = 4, max.depth = 6)

rf_pred <- predict(rf_model, complete_data_test2)

rf_model

mean((rf_pred - complete_data_test$Unemp_rate)^2)
library(Metrics)
mae4 <- mae(rf_pred, complete_data_test$Unemp_rate)
mse4 <- mse(rf_pred, complete_data_test$Unemp_rate)
rmse4 <- rmse(rf_pred, complete_data_test$Unemp_rate)
rsq4 <- cor(rf_pred, complete_data_test$Unemp_rate)^2
print(paste("MAE-", mae4, "///MSE-", mse4, "///RMSE-", rmse4, "///RSQ-", rsq4))

check3 <- data.frame(rf_pred, complete_data_test$Unemp_rate)

########### 10 fold CV ###########
library(caret)

rf_ctrl  <- trainControl(method = "cv", number = 10)

rf_grid <- expand.grid(.mtry = c(1,2,3,4,5))
set.seed(1100)
rf_cust <- train(Unemp_rate~., data = complete_data_train, method = "rf", metric= "mse" , trControl = rf_ctrl,
                 tuneGrid = rf_grid )

rf_cust

rf_cust_pred <- predict(rf_cust, complete_data_test2)




mae5 <- mae(rf_cust_pred, complete_data_test$Unemp_rate)
mse5 <- mse(rf_cust_pred, complete_data_test$Unemp_rate)
rmse5 <- rmse(rf_cust_pred, complete_data_test$Unemp_rate)
rsq5 <- cor(rf_cust_pred, complete_data_test$Unemp_rate)^2

print(paste("MAE-", mae5, "///MSE-", mse5, "///RMSE-", rmse5, "///RSQ-", rsq5))

check4 <- data.frame(rf_cust_pred, complete_data_test$Unemp_rate)

################### model tree ########
#install.packages("party")
library("party")

complete_data_train1 <- complete_data_train
complete_data_train1$DATE <- as.numeric(complete_data_train1$DATE - as.Date("1954-01-01"))
complete_data_test1 <- complete_data_test2
complete_data_test1$DATE <- as.numeric(complete_data_test2$DATE - as.Date("1954-01-01"))

model_tree_reg <- ctree(Unemp_rate ~., data = complete_data_train1)

model_tree_pred <- predict(model_tree_reg, complete_data_test1)

c_mae_mt <- mae(model_tree_pred, complete_data_test$Unemp_rate)
c_mse_mt <- mse(model_tree_pred, complete_data_test$Unemp_rate)
c_rmse_mt <- rmse(model_tree_pred, complete_data_test$Unemp_rate)
c_rsq_mt <- cor(model_tree_pred, complete_data_test$Unemp_rate)^2

print(paste("MAE-", c_mae_mt, "/// MSE-", c_mse_mt, "/// RMSE-", c_rmse_mt, "/// RSQ-", c_rsq_mt))



###################### Ridge regression ##############
library(glmnet)
#install.packages("glmnet")
library(car)
x <- as.matrix(complete_data_train1[-6])
y <- complete_data_train$Unemp_rate
cv_model <- cv.glmnet(x,y, alpha=0, lambda = seq(0.001,0.1,0.001), nfolds = 10)
min_l <- cv_model$lambda.min
ridge_model <- glmnet(x,y, alpha = 0, lambda = min_l)

ridge_pred <- predict(ridge_model, as.matrix(complete_data_test1))

mae_r <- mae(ridge_pred, complete_data_test$Unemp_rate)
mse5_r <- mse(ridge_pred, complete_data_test$Unemp_rate)
rmse5_r <- rmse(ridge_pred, complete_data_test$Unemp_rate)
rsq5_r <- cor(ridge_pred, complete_data_test$Unemp_rate)^2

print(paste("MAE-", mae_r, "///MSE-", mse5_r, "///RMSE-", rmse5_r, "///RSQ-", rsq5_r))

#######################XG Boost #####################
#install.packages("xgboost")
library(xgboost)
params <- list(
  booster = "gbtree",
  eta = 0.1,max_depth = 5, subsample = 0.8, colsample_bytree = 0.8, eval_metric = "rmse")
# Train the xgboost model
boost_model <- xgboost(data = as.matrix(complete_data_train1[-6]), 
                 label = complete_data_train1$Unemp_rate, 
                 params = params, 
                 nrounds = 82)

boost_pred <- predict(boost_model, as.matrix(complete_data_test1))

mae_b <- mae(boost_pred, complete_data_test$Unemp_rate)
mse5_b <- mse(boost_pred, complete_data_test$Unemp_rate)
rmse5_b <- rmse(boost_pred, complete_data_test$Unemp_rate)
rsq5_b <- cor(boost_pred, complete_data_test$Unemp_rate)^2

print(paste("MAE-", mae_r, "///MSE-", mse5_r, "///RMSE-", rmse5_r, "///RSQ-", rsq5_r))

########Cluster Analysis ##################
library(xlsx)
read.xlsx("emp-unemployment.xls", 2, startRow = 6)


states <- read.xlsx("emp-unemployment.xls", 2, startRow = 6)

states <- states[-1]
states <- states[-1,]
states <- states[-c(52:59),]


colnames(states) <- gsub("X", "", colnames(states))



################ Clustering #########

states_n1 <- as.data.frame(lapply(states[2:length(states)], scale))
states_n <- as.data.frame(scale(states[2:length(states)]))


library(factoextra)
fviz_nbclust(states_n, kmeans, method = "silhouette", k.max = 8)

set.seed(700)
states_clusters <- kmeans(states_n,3)

states_clusters$size

states_clusters$centers

table(states_clusters$cluster)

library(cluster)

clusplot(states_n, states_clusters$cluster, color = T, shade = T, labels = 2, lines = 0)


fviz_cluster(states_clusters, data = states_n, geom = "point", stand = F, ellipse.type = "norm")



cluster_assign <- data.frame(state = states$Area, cluster = states_clusters$cluster)
cluster_assign[order(cluster_assign$cluster),]
cluster_assign<- cluster_assign[order(cluster_assign$cluster),]
library(dplyr)
cluster_assign%>%
  group_by(cluster)

cluster_means <- aggregate(states[2:length(states)], by = list(cluster= states_clusters$cluster), mean)

library(tidyr)
library(ggplot2)

cluster_trends <- gather(cluster_means, key = "year", value = "unemployment_rate", -cluster)

ggplot(cluster_trends, aes(x= year, y = unemployment_rate, colour = factor(cluster), group = cluster)) + geom_line(size =2) +
  labs(x="year", y ="unemployment rate", color = "Cluster") + scale_color_discrete(name = "Cluster") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5))

fviz_cluster(states_clusters, data = states[2:length(states)], axes = c(1,2))



######################################## PREDICTION Final


test2022_1 <- method1[820:828,]
test2022 <- test2022_1[-6]
test2022_date <-test2022
test2022_date$DATE <- as.numeric(test2022$DATE - as.Date("1954-01-01"))


og_pred <- predict(rf_model, test2022)

check_og <- data.frame(og_pred, test2022_1$Unemp_rate)

pred_2023 <- data.frame(Date = test2022_1$DATE,Unemp_rate = og_pred)


test2022_1$DATE <- seq(as.Date("2023-04-01"), by = "month", length.out = 9)


#######################plot RMSE ###############

metrics_df <- data.frame(Model = c("Ridge Reg", "Regression Trees", "Model Tree", "Random Forest CV", "Random forest","XG Boost"), 
                         RMSE = c(0.0390, 0.1920, 0.20531, 0.109, 0.08, 0.0390),
                         MSE = c(0.0015, 0.03, 0.04, 0.01, 0.007, 0.0015),
                         MAE = c(0.032, 0.036, 0.11, 0.053, 0.050, 0.0328))

df_long <- pivot_longer(metrics_df, cols = c("RMSE", "MSE", "MAE"), names_to = "metric", values_to = "value")
df_long <- df_long%>%
  arrange(value)

# Plot a grouped bar chart
ggplot(df_long, aes(x = Model, y = value, fill = metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Model", y = "Metric Value", fill = "Metric") +
  scale_fill_manual(values = c("red", "blue", "green")) +
  theme_bw()
write.csv(df_long, "metrics.csv", row.names = F)
