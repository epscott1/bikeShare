library(vroom)
library(tidyverse)
library(ggplot2)
library(GGally)
library(skimr)
library(patchwork)
library(tidymodels)
library(dplyr)
library(glm2)
library(glmnet)
library(poissonreg)
library(keras)
library(yardstick)
library(ranger)
library(rpart)

#Read in the data set and fix the format
dat <- vroom("train.csv") %>%
  mutate(datetime = as.POSIXct(datetime, format="%Y-%m-%dT%H:%M:%SZ", tz="UTC")) %>%
  mutate(season = as.factor(season),
         weather = as.factor(weather)) 

dat2 <- vroom("test.csv") %>%
  mutate(datetime = as.POSIXct(datetime, format="%Y-%m-%dT%H:%M:%SZ", tz="UTC")) %>%
  mutate(season = factor(season, levels = levels(dat$season)),
         weather = factor(weather, levels = levels(dat$weather)))

#Random Forest Setup

my_mod <- rand_forest(mtry = 17,
                      min_n = 10,
                      trees = 2500) %>%
  set_engine("ranger") %>% 
  set_mode("regression")

#Recipe Setup
my_recipe <- recipe(count ~ temp + windspeed + holiday + workingday + season + weather + atemp +
                      humidity + datetime, data = dat) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(rushhour = case_when(
    datetime_hour %in% c(6, 7, 8, 9, 16, 17, 18) & workingday == 1 ~ "yes",
    TRUE ~ "no")) %>%
  step_mutate(rushhour = as.factor(rushhour)) %>%
  step_interact(terms = ~ windspeed:weather) %>%
  step_interact(terms = ~ humidity:weather) %>%
  step_interact(terms = ~ windspeed:temp) %>%
  step_interact(terms = ~ windspeed:atemp) %>%
  step_interact(terms = ~ temp:atemp) %>%
  step_rm(datetime, skip = TRUE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

#Workflow Setup with the Decision Tree Model
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = dat)

# Making Predictions
predictions <- predict(preg_wf, new_data = dat2)

# Baking the recipe for the test set
prepped_recipe <- prep(my_recipe)
dat2_baked <- bake(prepped_recipe, dat2)
ncol(dat2_baked)



#The rest of the code is just formatting to make it useable for Kaggle

kaggle_submission <- dat2_baked %>%
  bind_cols(predictions) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = kaggle_submission, file="./finalbike1.csv", delim=",")
