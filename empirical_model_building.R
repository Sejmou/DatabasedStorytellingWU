#-------------------------------------------------------------------#
#-----------------------Empirical Model Building--------------------#
#-------------------------------------------------------------------# 

# Load packages & adjust settings
## ------------------------------------------------------------------------
library(caret)
library(ggplot2)
library(psych)
library(dplyr)
library(readr)
library(fixest)
library(pROC)
library(psych)
library(boot)
library(xgboost)
library(data.table)
library(tidyr)
library(CausalImpact)
library(tidyverse)
library(MatchIt)
library(lmtest)
library(sandwich)
library(stargazer)
library(MASS)
library(reghelper)
library(DescTools)
library(reshape2)
library(e1071)

# this essentially disables scientific notation
options(scipen = 999)

#-------------------------------------------------------------------#
#------------------------Prediction models--------------------------#
#-------------------------------------------------------------------#

# Load necessary libraries
library(caret)
library(pROC)
library(dplyr)
library(ggplot2)
library(corrplot)
library(reshape2)

# Load the dataset from a URL and view the first few rows to understand the data structure
pred_data <- read.csv("https://raw.githubusercontent.com/WU-RDS/RMA2022/main/data/Targeting.csv", header=TRUE, sep=",")
#head(pred_data)

# Step 1: Data Preparation
# Convert the 'Choice' variable to a categorical factor with levels 'No' and 'Yes'
pred_data$ChoiceCat <- factor(pred_data$Choice, levels = c("0", "1"), labels = c("No", "Yes"))

# Step 2: Visual Data Inspection
# e.g., Association between 'Choice' and 'Amt_purchased'
ggplot(pred_data,aes(Amt_purchased,Choice)) +  
  geom_point(shape=1) +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial"), 
              se = FALSE) +
  theme_bw()

#Step 3: Train the Model with Cross-Validation
#Set a seed for reproducibility of results
set.seed(333)
#Define the training control parameters for cross-validation
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
#Train the logistic regression model using the cross-validation plan defined above
model_cv <- train(ChoiceCat ~ ., data = pred_data %>% dplyr::select(-Customer.ID, -Choice),
                  method = "glm", family = "binomial",
                  trControl = control, metric = "ROC")

#Step 4: Model Evaluation
# Ensure "Yes" is the first level of the 'ChoiceCat' factor
pred_data$ChoiceCat <- relevel(pred_data$ChoiceCat, ref = "Yes")
#Predict probabilities for the 'Yes' class using the cross-validated model
test_prob <- predict(model_cv, newdata=pred_data %>% dplyr::select(-Customer.ID, -Choice, -ChoiceCat), type="prob")[, "Yes"]
#Convert probabilities to binary predictions based on a 0.5 threshold
test_pred <- ifelse(test_prob > 0.5, "Yes", "No")
#Make sure that the predicted values are factored and have the same level order as 'ChoiceCat'
test_pred_factor <- factor(test_pred, levels = levels(pred_data$ChoiceCat))
#Generate a confusion matrix to evaluate model predictions against actual labels
cm <- confusionMatrix(test_pred_factor, pred_data$ChoiceCat)
print(cm)

#Calculate precision, recall, F1 score, and accuracy from the confusion matrix
precision <- cm$byClass['Pos Pred Value']
recall <- cm$byClass['Sensitivity']
F1 <- 2 * (precision * recall) / (precision + recall)
accuracy <- cm$overall['Accuracy']
#Print the metrics with explanations
cat("Precision:", precision, "- Proportion of true positives over total predicted positives.\n")
cat("Recall:", recall, "- Proportion of true positives over all actual positives.\n")
cat("F1 Score:", F1, "- Harmonic mean of precision and recall, usefulwhen seeking a balance between these metrics.\n")
cat("Accuracy:", accuracy, "- Overall proportion of correct predictions.\n")

#Generate and plot the ROC curve to visualize model performance
roc_obj <- roc(response = pred_data$ChoiceCat, predictor = as.numeric(test_prob))
plot(roc_obj, main="ROC Curve")

#Calculate the AUC to quantify the ROC graphically
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "- Measure of model's ability to distinguish between classes; higher is better.\n")

# Step 5: Model Interpretation
# Extract the final logistic regression model after cross-validation
final_model <- model_cv$finalModel

# Display the summary of the final model to interpret coefficients
# The coefficients represent the log-odds impact of each predictor variable
# Positive coefficients increase the log-odds of the positive class (implying higher probability)
# Negative coefficients decrease the log-odds of the positive class (implying lower probability)
summary(final_model)

# Interpret the summary output
# For each one-unit increase in a predictor variable, the log-odds of 'ChoiceCat' being 'Yes' changes by its corresponding coefficient value,
# holding all other predictors constant. A more significant positive coefficient means a stronger positive effect on the probability of 'Yes'.
# Conversely, a more significant negative coefficient means a stronger negative effect on the probability of 'Yes'.
# The intercept is the log-odds of 'ChoiceCat' being 'Yes' when all predictor variables are zero.

# Calculate and print the odds ratios for each predictor
odds_ratios <- exp(coef(final_model))
print(odds_ratios)
# Interpret the odds ratios
# Odds ratios represent how the odds multiply for a one-unit increase in the predictor variable.
# An odds ratio greater than 1 indicates an increase in odds for 'Yes', and less than 1 indicates a decrease.

#xgboost
#-------------------------------------------------------------------
# Load the necessary libraries
library(xgboost)
library(caret)
library(pROC)
library(dplyr)
library(ggplot2)

# Read the dataset
pred_data <- read.csv("https://raw.githubusercontent.com/WU-RDS/RMA2022/main/data/Targeting.csv", header=TRUE, sep=",")

# Convert 'Choice' to a factor
pred_data$Choice <- factor(pred_data$Choice, levels = c("0", "1"), labels = c("No", "Yes"))

# Split the data into training and test sets
set.seed(333) # for reproducibility
index <- createDataPartition(pred_data$Choice, p = 0.8, list = FALSE)
train_set <- pred_data[index,]
test_set <- pred_data[-index,]

#Prepare data for xgboost (xgboost requires numeric matrices)
train_matrix <- xgb.DMatrix(data.matrix(train_set[, -which(names(train_set) %in% c("Customer.ID", "Choice"))]), label = as.numeric(train_set$Choice) - 1)
test_matrix <- xgb.DMatrix(data.matrix(test_set[, -which(names(test_set) %in% c("Customer.ID", "Choice"))]), label = as.numeric(test_set$Choice) - 1)

#Define the control parameters for hyperparameter tuning using caret
xgbGrid <- expand.grid(
  nrounds = c(150),  # Number of boosting rounds
  eta = c(0.05, 0.1),     # Learning rate
  max_depth = c(4), # Maximum depth of a tree
  gamma = c(0),      # Minimum loss reduction required for a split
  colsample_bytree = c(1), # Subsample ratio of columns for each tree
  min_child_weight = c(1),   # Minimum sum of instance weight needed in a child
  subsample = c(0.7)         # Subsample ratio of the training instances
)

#List of hyperparameters to tune
#eta: lower eta = less likely over-fitting and more rounds needed (i.e. more trees); [typical values: 0.01-0.2]
#max.depth: maximum number of nodes of a tree -> Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample (i.e., higher = more likely overfit) [typical values: 3-10]
#nround: max number of boosting iterations
#colsample_bytree: denotes the fraction of columns to be randomly samples for each tree. [typical values: 0.5-1]
#min_child_weight: Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values can lead to under-fitting hence, it should be tuned using CV. [default 1, smaller = more likely overfit]
#n_estimators: Early stoping with validation set
#reg_alpha: l1 regularization (lasso, absolute value) -> regularization on leaf weights -> high value = leafs go to 0; Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
#reg_lambda: l2 regularization (ridge, squared value) -> smooth regularization on leaf weights -> high value = leafs go to 0; This used to handle the regularization part of XGBoost. Though many data scientists don't use it often, it should be explored to reduce overfitting.
#subsample: fraction of observations to subsample at each step -> very high = more likely overfit; very low = more likely underfit; Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting. [typical values: 0.5-1]
#gamma = 0.7, #minimum loss reduction allowed for split to occur -> high value = fewer splits [default = 0]

#Set the training control
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "none",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

#Train the xgboost model using caret to optimize hyperparameters

set.seed(333) # Ensure reproducibility
xgb_model <- train(
  Choice ~ ., data = train_set %>% dplyr::select(-Customer.ID),
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgbGrid,
  metric = "ROC"
)

#Print the best-tuned model parameters
print(xgb_model$bestTune)

#Predict on the test set
test_pred_prob <- predict(xgb_model, newdata = test_set %>% dplyr::select(-Customer.ID, -Choice), type = "prob")
test_pred <- ifelse(test_pred_prob[, "Yes"] > 0.5, "Yes", "No")
#Confusion matrix and model evaluation
conf_matrix <- confusionMatrix(as.factor(test_pred), test_set$Choice)
print(conf_matrix)

# Extract performance metrics from the confusion matrix
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']
F1 <- 2 * (precision * recall) / (precision + recall)

# Print the metrics with explanations
cat("\nModel Evaluation Metrics:\n")
cat("Accuracy:", accuracy, "- Proportion of all predictions that are correct.\n")
cat("Precision:", precision, "- Proportion of positive identifications that were actually correct.\n")
cat("Recall:", recall, "- Proportion of actual positives that were identified correctly.\n")
cat("F1 Score:", F1, "- Weighted average of Precision and Recall. Useful when the class distribution is uneven.\n")

#ROC curve and AUC for the test set
roc_curve <- roc(response = as.numeric(test_set$Choice) - 1, predictor = test_pred_prob[, "Yes"])
plot(roc_curve, main = "ROC Curve for XGBoost Model")
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "- AUC is the area under the ROC curve; higher AUC indicates better model performance.\n")

#Plot variable importance
importance_matrix <- xgb.importance(feature_names = colnames(train_set)[-which(names(train_set) %in% c("Customer.ID", "Choice"))], model = xgb_model$finalModel)
xgb.plot.importance(importance_matrix = importance_matrix, main = "Variable Importance for XGBoost Model")

#-------------------------------------------------------------------#
#------------------Fixed-effects panel data model-------------------#
#-------------------------------------------------------------------#

#panel data models
#-------------------------------------------------------------------
#load data
music_data <- fread("https://raw.githubusercontent.com/WU-RDS/RMA2022/main/data/music_data.csv")
head(music_data)
#convert to factor
music_data$song_id <- as.factor(music_data$song_id)
music_data$genre <- as.factor(music_data$genre)
#number of unique songs in data set
length(unique(music_data$song_id))

#example plot to visualize the data structure
ggplot(music_data, aes(x = week, y = streams/1000000,group = song_id, fill = song_id, color = song_id)) +
  geom_area(position = "stack", alpha = 0.65) +
  labs(x = "Week",y = "Total streams (in million)", title = "Weekly number of streams by song") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5,color = "#666666"),legend.position = "none")

#another example plot for 9 random songs from the sample
sample_songs <- sample(music_data$song_id,9,replace = F)
ggplot(music_data %>% dplyr::filter(song_id %in% sample_songs), aes(x = week, y = streams/1000000)) +
  geom_area(fill = "steelblue", color = "steelblue",alpha = 0.5) + facet_wrap(~song_id, scales = "free_y") +
  labs(x = "Week", y = "Total streams (in million)", title = "Weekly number of streams by country") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5,color = "#666666"))

#another example
sample_song <- sample(music_data$song_id,1,replace = F)
plot_data <- music_data %>% dplyr::filter(song_id %in% sample_song) %>% as.data.frame()
plot_data_long <- gather(plot_data %>% dplyr::select(-release_date,-weeks_since_release), variable, value, streams:adspend, factor_key=TRUE)
plot_data_long
ggplot(plot_data_long, aes(x = week, y = value)) +
  geom_area(fill = "steelblue", color = "steelblue",alpha = 0.5) + facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(x = "Week", y = "Value", title = "Development of key variables over time") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5,color = "#666666"))

hist(music_data$streams)
hist(log(music_data$streams))

#estimate the baseline model
fe_m0 <- lm(log(streams) ~ log(radio+1) + log(adspend+1) ,
            data = music_data)
summary(fe_m0)

#... + control for song age
fe_m1 <- lm(log(streams) ~ log(radio+1) + log(adspend+1) + log(weeks_since_release+1),
            data = music_data)
summary(fe_m1)

#... + playlist follower variable
fe_m2 <- lm(log(streams) ~ log(radio+1) + log(adspend+1) + log(weeks_since_release+1) + log(playlist_follower),
            data = music_data)
summary(fe_m2)

#... + song fixed effects
fe_m3 <- lm(log(streams) ~ log(radio+1) + log(adspend+1) + log(playlist_follower) + log(weeks_since_release+1) +
              as.factor(song_id),
            data = music_data)
summary(fe_m3)
library(stargazer)
stargazer(fe_m0,fe_m1,fe_m2,fe_m3,type="text")

#... same as m3 using the fixest package
library(fixest) #https://lrberge.github.io/fixest/
fe_m4 <- feols(log(streams) ~ log(radio+1) + log(adspend+1) + log(playlist_follower) + log(weeks_since_release+1)
               | song_id,
               data = music_data)
etable(fe_m4,se = "cluster")

#... + week fixed effects
fe_m5 <- feols(log(streams) ~ log(radio+1) + log(adspend+1) + log(playlist_follower) + log(weeks_since_release+1)
               | song_id + week,
               data = music_data)
etable(fe_m4,fe_m5,se = "cluster")
#extract fixed effects coefficients
fixed_effects <- fixef(fe_m5)
summary(fixed_effects)

#mixed effects model
library(lme4) #https://github.com/lme4/lme4
music_data$log_playlist_follower <- log(music_data$playlist_follower)
music_data$log_streams <- log(music_data$streams)
re_m1 <- lmer(log_streams ~ log(radio+1) + log(adspend+1) + log_playlist_follower + log(weeks_since_release+1) + (1 + log_playlist_follower | song_id), data=music_data)
summary(re_m1)

library(sjPlot)
plot_model(re_m1, show.values = TRUE, value.offset = .3)
# plot random-slope-intercept
plot_model(re_m1, type="pred", terms=c("log_playlist_follower","song_id"),
           pred.type="re", ci.lvl=NA) +
  scale_colour_manual(values=hcl(0,100,seq(40,100,length=97))) +
  theme(legend.position = "bottom", legend.key.size=unit(0.3,'cm')) + guides(colour = guide_legend(nrow = 5))

#-------------------------------------------------------------------#
#---------------------Propensity Score Matching---------------------#
#-------------------------------------------------------------------#

#load data
smoking_data <- read_csv("https://raw.githubusercontent.com/gckc123/ExampleData/main/smoking_psyc_distress.csv")

#Since remoteness is a categorical variable with more than two categories. It is necessary to convert it into a factor variable.
#For other categorical variable with only 2 levels, this is optional if the variable is coded as 0 and 1.
smoking_data$remoteness <- factor(smoking_data$remoteness, exclude = c("", NA))

#Use the matchit()-function from the MatchIt-package to match each smoker with a non-smoker (1 to 1 matching)
match_obj <- matchit(smoker ~ sex + indigeneity + high_school + partnered + remoteness + language + risky_alcohol + age,
                     data = smoking_data, method = "nearest", distance ="glm",
                     ratio = 1,
                     replace = FALSE)
summary(match_obj)

#plotting the balance between smokers and non-smokers
plot(match_obj, type = "jitter", interactive = FALSE)
plot(summary(match_obj), abs = FALSE)

#Extract the matched data and save the data into the variable matched_data
matched_data <- match.data(match_obj)

#Run regression model with psychological distress as the outcome, and smoker as the only predictor
#We need to specify the weights - Matched participants have a weight of 1, unmatched participants 
res_matched <- lm(psyc_distress ~ smoker, data = matched_data, weights = weights)
summary(res_matched)
#Test the coefficient using cluster robust standard error
coeftest(res_matched, vcov. = vcovCL, cluster = ~subclass)
#Calculate the confidence intervals based on cluster robust standard error
coefci(res_matched, vcov. = vcovCL, cluster = ~subclass, level = 0.95)

res_nonmatched <- lm(psyc_distress ~ smoker, data = smoking_data)
summary(res_nonmatched)

stargazer(res_matched,res_nonmatched, type="text")

#-------------------------------------------------------------------#
#----------------Difference-in-Differences estimator----------------#
#-------------------------------------------------------------------#

#load data
did_data <- fread("https://raw.github.com/WU-RDS/RMA2022/main/data/did_data_exp.csv")
#pre-processing
did_data$song_id <- as.character(did_data$song_id)
did_data$treated_fct <- factor(did_data$treated,levels = c(0,1),labels = c("non-treated","treated"))
did_data$post_fct <- factor(did_data$post,levels = c(0,1),labels = c("pre","post"))
did_data$week <- as.Date(did_data$week)
did_data <- did_data %>% dplyr::filter(!song_id %in% c("101","143","154","63","161","274")) %>% as.data.frame()

#inspect data
head(did_data)
did_data %>% dplyr::group_by(treated) %>%
  dplyr::summarise(unique_songs = n_distinct(song_id))
library(panelView) #https://yiqingxu.org/packages/panelview/
panelview(streams ~ treated_post, data = did_data,  index = c("song_id","week"), pre.post = TRUE, by.timing = TRUE)
panelview(streams ~ treated_post, data = did_data ,  index = c("song_id","week"), type = "outcome")
did_data <- did_data %>% group_by(song_id) %>% dplyr::mutate(mean_streams = mean(streams)) %>% as.data.frame()
panelview(streams ~ treated_post, data = did_data %>% dplyr::filter(mean_streams<70000),  index = c("song_id","week"), type = "outcome")

#alternatively, split plot by group
#compute the mean streams per group and week
did_data <- did_data %>%
  dplyr::group_by(treated_fct,week) %>%
  dplyr::mutate(mean_streams_grp=mean(log(streams))) %>% as.data.frame()
#set color scheme for songs
cols <- c(rep("gray",length(unique(did_data$song_id))))
#set labels for axis
abbrev_x <- c("-10", "", "-8", "",
              "-6", "", "-4", "",
              "-2", "", "0",
              "", "+2", "", "+4",
              "", "+6", "", "+8",
              "", "+10")
#axis titles and names
title_size = 26
font_size = 24
line_size =1/2
#create plot
ggplot(did_data) +
  geom_step(aes(x=week,y=log(streams), color = song_id), alpha = 0.75) +
  geom_step(aes(x =week, y = mean_streams_grp),color = "black", size = 2, alpha=0.5) +
  #geom_vline(xintercept = as.Date("2018-03-19"),color="black",linetype="dashed") +
  labs(x="week before/after playlist listing",y="ln(streams)",
       title="Number of weekly streams per song") +
  scale_color_manual(values = cols) + theme_bw() +
  scale_x_continuous(breaks = unique(did_data$week), labels = abbrev_x) +
  theme(legend.position = "none",
        panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank(),
        strip.text.x = element_text(size = font_size),
        panel.grid.major.y = element_line(color = "gray75",
                                          size = 0.25,
                                          linetype = 1),
        panel.grid.minor.y =  element_line(color = "gray75",
                                           size = 0.25,
                                           linetype = 1),
        plot.title = element_text(color = "#666666",size=title_size),
        axis.title = element_text(size = font_size),
        axis.text  = element_text(size=font_size),
        plot.subtitle = element_text(color = "#666666",size=font_size),
        axis.text.x=element_text(size=font_size)
  ) + facet_wrap(~treated_fct)

#run baseline did model
did_m1 <- lm(log(streams+1) ~ treated * post,
             data = did_data %>% dplyr::filter(week!=as.Date("2018-03-19")))
summary(did_m1)

#same model using fixed effects specification
did_m2 <- feols(log(streams+1) ~ treated * post |
                  song_id + week,
                cluster = "song_id",
                data = did_data  %>% dplyr::filter(week!=as.Date("2018-03-19")))
etable(did_m2, se = "cluster")

#parallel pre-treatment trend assessment
#1. inspect period-specific effects
did_data <- did_data %>% dplyr::group_by(song_id) %>%
  dplyr::mutate(period = seq(n())) %>%
  as.data.frame()
did_m3 <- fixest::feols(log(streams) ~ i(period, treated, ref = 10) | song_id + period,
                        cluster = "song_id",
                        data = did_data)
etable(did_m3, se="cluster")
fixest::iplot(did_m3,
              xlab = 'Time to treatment (treatment = week 11)',
              main = 'TWFE DiD')

#2. placebo-test for pre-treatment parallel trend test
did_data_placebo <- did_data %>% dplyr::filter(period<11) %>% as.data.frame()
did_data_placebo$post <- ifelse(did_data_placebo$period>=5,1,0)
placebo_model <- feols(log(streams+1) ~ treated * post |
                         song_id + week,
                       cluster = "song_id",
                       data = did_data_placebo)
etable(placebo_model, se = "cluster")

#-------------------------------------------------------------------#
#------------------Heterogeneous treatment effects------------------#
#-------------------------------------------------------------------#

#heterogeneity across treated units
#example genre
did_m4 <- feols(log(streams+1) ~ treated_post * as.factor(genre) |
                  song_id + week,
                cluster = "song_id",
                data = did_data  %>% dplyr::filter(week!=as.Date("2018-03-19")))
etable(did_m4, se = "cluster")

#heterogeneity across time
did_data$treated_post_1 <- ifelse(did_data$week > as.Date("2018-03-19") & did_data$week <= (as.Date("2018-03-19")+21) & did_data$treated==1,1,0)
did_data$treated_post_2 <- ifelse(did_data$week > as.Date("2018-04-09") & did_data$week <= (as.Date("2018-04-09")+21) & did_data$treated==1,1,0)
did_data$treated_post_3 <- ifelse(did_data$week > as.Date("2018-04-30") & did_data$treated==1,1,0)
did_m5 <- feols(log(streams+1) ~ treated_post_1 + treated_post_2 + treated_post_3 |
                  song_id + week,
                cluster = "song_id",
                data = did_data  %>% dplyr::filter(week!=as.Date("2018-03-19")))
etable(did_m5, se = "cluster")

#-------------------------------------------------------------------#
#----------------------Instrumental variables-----------------------#
#-------------------------------------------------------------------#

#this example uses simulated data

#we are really generating x* and c and using a common variance
x_star_c <- mvrnorm(1000, c(20, 15), matrix(c(1, 0.5, 0.5, 1), 2, 2))
x_star <- x_star_c[, 1]
c <- x_star_c[, 2]

#simulate instrument z that is correlated with the endogenous regressor
z <- rnorm(1000)
x <- x_star + z

#generate the outcome as a function of x and the unobserved confounder, and some noise
y <- 1 + x + c + rnorm(1000, 0, 0.5)

#run model with unobserved confounder
miv_1 <- lm(y ~ x + c)
summary(miv_1)
miv_2 <- lm(y ~ x)
summary(miv_2)
stargazer(miv_1,miv_2, type="text")

#2SLS approach
#stage 1: predict x from the instrument z
x_hat <- lm(x ~ z)$fitted.values
#stage 2: replace x with the predicted value of x from stage 1 
miv_3 <- lm(y ~ x_hat)
summary(miv_3)
stargazer(miv_1,miv_2,miv_3, type="text")