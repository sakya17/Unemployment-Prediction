# Unemployment-Prediction using Machine Learning algorithms
This project is done as a part of the Data Mining course in the Master of Data Analytics program at University of Houston - Downtown

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/9be19d09-3ce0-43ba-b5f0-e8d297bd8771)
Unemployment rate in the US from 1954 - 2023

### Origin
Semester Date: Spring 2023
Class: Data Mining (CS 5310)
Program: Master of Science in Data Analytics
School: University of Houston Downtown

### Introduction

This is a Data Mining project during the spring 2023 semester as a part of Masters in Data Analytics(MSDA) program at the University of Houston Downtown (UHD).

This Project is about predicting the unemployment rate for the coming months in 2023 using the historic economic indicators data and demographic indicators data.

I spent a lot of time in collecting the data as the data is needed in monthly fashion to predict the unemployment in monthly fashion for 2023.

I chose this project because it provides an opportunity to explore all the data mining techniques and the main machine learning algorithms to produce the best output.

Through this project i have explored the all the stages of the data analytics workflow which includes :

Data pre-processing

Modeling

Evaluation

Prediction

The data pre-processing tasks employed in this project were : Data merging - combining different data sets, Data cleaning - removing redundant values, addressing missing values and noisy data, changing data types of the attributes to comply with the ML algorithms functions in R, feature engineering using principal component analysis etc.

The next step is modeling , in this step i created different models, which are evaluated , improved and finally the best performing model is selected. The primary evaluation metric was the Root-Mean-Square_Error.

Final steps include prediction. In this step i used the best performing model using the evaluation metric to predict the test data(May to December 2023).
### Data Collection

I have collected the historic data from different data sources which included monthly data of the economic indicators like inflation rates, interest rates, producer price index, consumer confident index, GDP etc. and also demographic data like unemployment of the white race, average hours per week etc. within the time frmae of 1954 -2022. This is the scenario where each attribute is a single data set.

Unemployment rate data for each state is also collected to explore the performance of states throughout the time.

### Data Pre-processing

As every data set is a single attribute and different from other datasets it took a lot of time to preprocess the data. Only required columns are selected from each data set and finally all these columns are merged together to a single data set with all the economic indicators, demographic indicators along with target variable unemployment rate form1954 -2022.

The final data set has the data of economic indicators and demographic data of unemployment rate of 831 months since 1954.

Missing values - there are a lot of missing values. I have imputed the missing values using the KNN algorithm.

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/358175a2-bbb6-4eff-9515-ef790ea0b2b3)
![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/0e2560e6-8622-4af4-be7a-ceb41842ac35)


Here imputing the missing values with mean or median is not relevant so, the step_impute_knn function from recipes package is thought be better here when compared to the other methods as i don't want to delete the rows with NA values.

Imputing the missing values using KNN algorithm, step_impute_knn function is used from recipes package.

### Multiple Regression

Multiple regression analysis is done to check the variation explained by the independent variables, multicollinearity and also the most important variables.
We can see from the summary that these variables are well explaining the unemployment rate.
![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/3572a645-3788-40e0-9ef4-aa8c237a1d13)


The model is good as the f statistic is significant with p value less than 0.01.

Correlation matrix says that the variables are highly correlated.

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/764f00ee-3a52-4a3e-87b9-98c60f7e33bc)

We can see that vif is exceeding 5 for some of the variables. So, these variables are highly correlated.

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/53978fc9-4920-4461-8c9c-35046b6b0c7a)

From the correlation map it is evident that the variables are highly correlated. So, i thought of doing the Principal component analysis. From the PCA, 4 principal components were extracted as these four components are explaining 89% of the variance. This was expected to reduce the multicollinearity. But the results are not so convincing with the principal components. 

The evaluation metrics are better with original variables than with principal components for all the models. So, i do not want to include the principal component analysis here as it gets too much complicated. 

### Model Creation

As the target variable is numerical variable. i wanted to explore ML algorithms like Decision Tree Regression, Model trees, Random forest, Gradient boosting algorithm.

Also, random forest algorithm and gradient boosting algorithms are very good when you have correlated variables. So, i have high expectations on random forest and gradient boosting algorithms.
Splitting the dataset is done with 25% as the test data. random sampling is done.

### Decision Tree Regression Algorithm

``` reg_tree <- rpart(Unemp_rate ~., data = complete_data_train, method = "anova", cp = 0.0001) ```

Method "anova" is used because our problem is regression type.

The complexity parameter of 0.0001 gave the best results after trying the tuning with other values.

Seems like complex tree works better here.

Model Evaluation

``` reg_tree_pred <- predict(reg_tree, complete_data_test2) ```

```r
mae3 <- mae(reg_tree_pred, complete_data_test$Unemp_rate)
mse3 <- mse(reg_tree_pred, complete_data_test$Unemp_rate)
rmse3 <- rmse(reg_tree_pred, complete_data_test$Unemp_rate)
rsq3 <- cor(reg_tree_pred, complete_data_test$Unemp_rate)^2
print(paste("MAE-", mae3, "///MSE-", mse3, "///RMSE-", rmse3, "///RSQ-", rsq3)) 
```

**"MAE- 0.110114471181898 ///MSE- 0.036890008735503 ///RMSE- 0.192067719139638 ///RSQ- 0.985370155158067"**

Most important variable for the unemployment in US.
```r
imp_var <- sort(reg_tree$variable.importance, decreasing = T)
most_imp <- names(imp_var)[1:10]
rpart.plot(reg_tree, type = 2, extra = 101, fallen.leaves = F, under = TRUE, cex = 0.6, branch.lty = 1, branch.lwd = 1,
            box.col = "lightblue") 
```
            
![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/0bc8213e-ecbf-4169-85f8-c6c0ac7c0957)

### Random Forest algorithm

```r 
set.seed(1200)
rf_model <- randomForest(Unemp_rate~., data = complete_data_train, ntree = 100, mtry = 4, max.depth = 6)

rf_pred <- predict(rf_model, complete_data_test2) 
```

After trying several tuning combinations, no.of trees - 100, no of variables at each node -4, depth of tree -6 gave the best results for this model.

model evaluation
```r mae4 <- mae(rf_pred, complete_data_test$Unemp_rate)
mse4 <- mse(rf_pred, complete_data_test$Unemp_rate)
rmse4 <- rmse(rf_pred, complete_data_test$Unemp_rate)
rsq4 <- cor(rf_pred, complete_data_test$Unemp_rate)^2
print(paste("MAE-", mae4, "///MSE-", mse4, "///RMSE-", rmse4, "///RSQ-", rsq4)) 
```

**"MAE- 0.050504106280193 ///MSE- 0.00761183238996231 ///RMSE- 0.0872458158879973 ///RSQ- 0.997163011492964"**
 
### 10 fold cross validation method

```r
rf_ctrl  <- trainControl(method = "cv", number = 10)
rf_grid <- expand.grid(.mtry = c(1,2,3,4,5))
set.seed(1100)
rf_cust <- train(Unemp_rate~., data = complete_data_train, method = "rf", metric= "RMSE" , trControl = rf_ctrl,
                 tuneGrid = rf_grid)
rf_cust_pred <- predict(rf_cust, complete_data_test2)
```
#Evaluation
```r
mae5 <- mae(rf_cust_pred, complete_data_test$Unemp_rate)
mse5 <- mse(rf_cust_pred, complete_data_test$Unemp_rate)
rmse5 <- rmse(rf_cust_pred, complete_data_test$Unemp_rate)
rsq5 <- cor(rf_cust_pred, complete_data_test$Unemp_rate)^2

print(paste("MAE-", mae5, "///MSE-", mse5, "///RMSE-", rmse5, "///RSQ-", rsq5))
```
**"MAE- 0.0533278421900161 ///MSE- 0.0120321158212028 ///RMSE- 0.109691001550732 ///RSQ- 0.995468328871509"**

### Model Tree Regression
Model tree is a variation of decision tree regression where it uses regression model instead of mean at each split.
```r
complete_data_train1 <- complete_data_train
complete_data_train1$DATE <- as.numeric(complete_data_train1$DATE - as.Date("1954-01-01"))
complete_data_test1 <- complete_data_test2
complete_data_test1$DATE <- as.numeric(complete_data_test2$DATE - as.Date("1954-01-01"))
model_tree_reg <- ctree(Unemp_rate ~., data = complete_data_train1)
model_tree_pred <- predict(model_tree_reg, complete_data_test1)
```
**Evaluation**
```r
c_mae_mt <- mae(model_tree_pred, complete_data_test$Unemp_rate)
c_mse_mt <- mse(model_tree_pred, complete_data_test$Unemp_rate)
c_rmse_mt <- rmse(model_tree_pred, complete_data_test$Unemp_rate)
c_rsq_mt <- cor(model_tree_pred, complete_data_test$Unemp_rate)^2
print(paste("MAE-", c_mae_mt, "/// MSE-", c_mse_mt, "/// RMSE-", c_rmse_mt, "/// RSQ-", c_rsq_mt))
```
**"MAE- 0.111207463990694 /// MSE- 0.0421555392641434 /// RMSE- 0.205318141585549 /// RSQ- 0.983852208154005"**


### Ridge Regression

I tried ridge regression methods manages the correlated variables.
```r
x <- as.matrix(complete_data_train1[-6])
y <- complete_data_train$Unemp_rate
cv_model <- cv.glmnet(x,y, alpha=0, lambda = seq(0.001,0.1,0.001), nfolds = 10)
min_l <- cv_model$lambda.min
ridge_model <- glmnet(x,y, alpha = 0, lambda = min_l)

ridge_pred <- predict(ridge_model, as.matrix(complete_data_test1))
```
**Evaluation**
```r
mae_r <- mae(ridge_pred, complete_data_test$Unemp_rate)
mse5_r <- mse(ridge_pred, complete_data_test$Unemp_rate)
rmse5_r <- rmse(ridge_pred, complete_data_test$Unemp_rate)
rsq5_r <- cor(ridge_pred, complete_data_test$Unemp_rate)^2

print(paste("MAE-", mae_r, "///MSE-", mse5_r, "///RMSE-", rmse5_r, "///RSQ-", rsq5_r))
```
**"MAE- 0.0328041786911582 ///MSE- 0.00152426395686021 ///RMSE- 0.039041823175413 ///RSQ- 0.999389660178682"**

### XG Boost algorithm

```r
params <- list(
  booster = "gbtree",
  eta = 0.1,max_depth = 5, subsample = 0.8, colsample_bytree = 0.8, eval_metric = "rmse")
# Train the xgboost model
boost_model <- xgboost(data = as.matrix(complete_data_train1[-6]), 
                 label = complete_data_train1$Unemp_rate, 
                 params = params, 
                 nrounds = 82)

boost_pred <- predict(boost_model, as.matrix(complete_data_test1))
```
**Evaluation**
```r
mae_b <- mae(boost_pred, complete_data_test$Unemp_rate)
mse5_b <- mse(boost_pred, complete_data_test$Unemp_rate)
rmse5_b <- rmse(boost_pred, complete_data_test$Unemp_rate)
rsq5_b <- cor(boost_pred, complete_data_test$Unemp_rate)^2

print(paste("MAE-", mae_r, "///MSE-", mse5_r, "///RMSE-", rmse5_r, "///RSQ-", rsq5_r))
```
**"MAE- 0.0328041786911582 ///MSE- 0.00152426395686021 ///RMSE- 0.039041823175413 ///RSQ- 0.999389660178682"**

### Artificial Neural Network algorithm
After trying a number of combinations of hyper parametric tuning for the learning rate and hidden layers, the learning rate of 0.006 and hidden layers of 4 gave the best results. 

The activation factor is not used at the output node because the target variable here is of continuous type and there should not be any range for the value, so, the activation factor is not used at the output node.
```r
scaled_data <- as.data.frame(sapply(complete_data_train1, 
                      scale))
set.seed(99)
ann_model <- neuralnet(Unemp_rate ~ ., data = scaled_data, hidden = 4, 
                        learningrate = 0.006, threshold = 0.01, linear.output = TRUE)
```
The training data and testing testing data is scaled for standardization. After that the predictions are re scaled to the original scale to evaluate using metrics.

**evaluation**
```r
scaledtestdata <- data.frame(sapply(complete_data_test1,scale))
ann_pred <- compute(ann_model, scaledtestdata)$net.result

rescaled <- ann_pred *sd(complete_data_train1$Unemp_rate) + mean(complete_data_train1$Unemp_rate)
```
```r
c_mae_an <- mae(rescaled, complete_data_test$Unemp_rate)
c_mse_an <- mse(rescaled, complete_data_test$Unemp_rate)
c_rmse_an <- rmse(rescaled, complete_data_test$Unemp_rate)
print(paste("MAE-", c_mae_an, "/// MSE-", c_mse_an, "/// RMSE-", c_rmse_an))
```

**"MAE- 0.0813837249670667 /// MSE- 0.009626285350451 /// RMSE- 0.0981136348855296"**

plotting the ann model

``` plot(ann_model, rep= "best") ```

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/35140add-f8c3-49ed-9559-989291a03c2e)

I choose RMSE as the main evaluation metric for this project as we prefer accuracy here in this scenario of predicting unemployment. 

The best RMSE scores were with XG boost model and with ridge regression model.

I finally chose XG boost model as this model perfectly manages the correlated variables.

#### Plotting the metrics
```r
library(tidyr)
metrics_df <- data.frame(Model = c("Ridge Reg", "Decision Tree", "Model Tree", "RF- CV", "RF","XG Boost", "ANN"), 
                         RMSE = c(0.0390, 0.1920, 0.20531, 0.109, 0.08, 0.0390,
                                  0.0981),
                         MSE = c(0.0015, 0.03, 0.04, 0.01, 0.007, 0.0015,0.0096),
                         MAE = c(0.032, 0.036, 0.11, 0.053, 0.050, 0.0328,0.081))

df_long <- pivot_longer(metrics_df, cols = c("RMSE", "MSE", "MAE"), names_to = "metric", values_to = "value")
df_long <- df_long%>%
  arrange(value)

# Plot a grouped bar chart
ggplot(df_long, aes(x = Model, y = value, fill = metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Model", y = "Metric Value", fill = "Metric") +
  scale_fill_manual(values = c("red", "blue", "green")) +
  theme_bw()
```

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/26a7c570-e502-483b-add7-95a99f23269f)

I have also created ML models with the principal components from PCA, i found these metrics are better with original variables when i compared the models with principal components. So, i have decided to exclude the PCA and stick to the original variables. 

### Final Predictions.
RMSE is used to deploy the final which will be our final model to predict the unemployment for the remaining months of 2023.
The predictors of 2022 are used to predict the final predictions.
```r
#Creating the 2023 test dataset with 2022 predictors as we dont have predictors for 2023 yet! However when the data for those particular months are available we can always use htat recent data predict the unemployment.

#Extracting the 2022 rows
test2022_1 <- method1[820:828,]
#creating the date for next 2023 months
test2022_1$DATE <- seq(as.Date("2023-04-01"), by = "month", length.out = 9)
#removing the target variable
test2022 <- test2022_1[-6]
test2022_date <-test2022
#changing the date format to numeric format.
test2022_date$DATE <- as.numeric(test2022$DATE - as.Date("1954-01-01"))

#PRedictions unsing the XG boost model
og_pred1 <- predict(boost_model, as.matrix(test2022_date))
pred_2023 <- data.frame(Date = test2022_1$DATE, Unemp_rate = og_pred1)
```
![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/8a7b386e-803e-410a-a708-0f58a8cd3bd3)

These are the predictions.

When the real data of economic indicators and demographic indicators of unemployment is available for the coming months of 2023 we can use this XG boost model to predict the unemployment.

### Cluster Analysis using K-means algorithm

#### I have also done the cluster analysis with the unemployment in the states data to see the performance of the states over the years.

Clustering
```r
library(factoextra)
library(cluster)
#Data standardization
states_n1 <- as.data.frame(lapply(states[2:length(states)], scale))
#Finding optimal clusters.
fviz_nbclust(states_n1, kmeans, method = "silhouette", k.max = 8)
```

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/248ed92f-5455-41c8-be15-6a7f492f765c)

This method says 2 clusters but i want to explore 1 more clusters.
#### K-means Clustering
```r
set.seed(700)
states_clusters <- kmeans(states_n1,3)
states_clusters$size
```

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/018c96bc-efa6-43b1-82f6-706c46149b18)

These are the cluster centroids which shows that the Cluster 2 has all the negative values and clusters 3 has all the positive values, cluster 1 has both.

Interpretation: The states in Cluster 2 are the states with unemployment rate less than the mean unemployment rate throughout the years.

The states are cluster 3 are the states with their unemployment greater than the mean unemployment rate throughout the years. 

Cluster has states with unemployment rate fluctuating above and below the mean unemployment rate.

**Visualization:**

``` fviz_cluster(states_clusters, data = states[2:length(states)], axes = c(1,2)) ```

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/9f6fb0f7-9cb7-4cbc-a24e-c39631a74544)


```r
#assigning the clusters to the states
cluster_assign <- data.frame(state = states$Area, cluster = states_clusters$cluster)
#putting in the order
cluster_assign<- cluster_assign[order(cluster_assign$cluster),]
#assigning the clusters with the mean unemployment for states for each year
cluster_means <- aggregate(states[2:length(states)], by = list(cluster= states_clusters$cluster), mean)

#creating a dataset with year, unemployment rate and the cluster.
cluster_trends <- gather(cluster_means, key = "year", value = "unemployment_rate", -cluster)

#Visualizing the performance of the states throughout the time
ggplot(cluster_trends, aes(x= year, y = unemployment_rate, colour = factor(cluster), group = cluster)) + geom_line(linewidth =2) +
  labs(x="year", y ="unemployment rate", color = "Cluster") + scale_color_discrete(name = "Cluster") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = .5))
```

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/369a0d68-ebd1-4593-b848-d8e32f372610)


![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/af96585f-ee1b-41cd-9780-3aaba932c1e0)

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/5d1a47d8-f491-423f-ba1a-34c081319948)

![image](https://github.com/sakya17/Unemployment-Prediction/assets/125946256/05a171d5-47a5-4b8f-9dc5-7a01cf761633)

When you take the consistency into the account, it can be concluded that states like South and North Dacotas, Minnesota, Maryland, Nebraska are performing better.

States like DC, California, Illinois, Louisiana, Oregon are not performing better!

### Conclusion

Economic indicators like Producer price index , interest rates, export goods and inflation rates are highly impacting unemployment. So, these economic factors should be  carefully monitored by the federal government.

Unemployment of white people is the mainly effecting the unemployment we can see this from the decision tree plot. Unemployment of the white people is the most important variable  for this analysis.

Other demographic indicators like unemployment labor force are mainly impacting unemployment in the US.

States like CA, DC, IL, New Mexico are consistently having higher unemployment rate than the mean unemployment rate. These states should be monitored to increase the unemployment in these states.

16 states are performing better and 13 states under-performing according the cluster analysis.

### Future Work

When we get the real data of the economic indicators and other demographic data is available we can check the real accuracy of this model.  Other area that can be improved is the ANN model .
