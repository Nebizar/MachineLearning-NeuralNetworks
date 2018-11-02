# Multiple Linear Regression

# Get data
dataset = read.csv('50_Startups.csv')

# Encoding the Categorical Variable
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Split into Test Set and Trainig Set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
#training_set = scale(training_set)
#test_set = scale(test_set)

#Fitting Multiple Linear Regrssion Model
regressor = lm(formula = Profit ~ .,#=Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               training_set)

#Predicting the Test Set results
y_pred = predict(regressor, newdata = test_set)

#Building Optimal Model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               dataset)
summary(regressor)

#regressor = lm(formula = Profit ~ R.D.Spend,
#               dataset)
#summary(regressor)
#FINISH OF BACKWARD ELIMINATION

#Automatic Backward Elimination
backwardElimination <- function(x,sl){
  numVars = length(x)
  for( i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars),"Pr(>|t|)"])
    if(maxVar >sl){
      j = which(coef(summary(regressor))[c(2:numVars),"Pr(>|t|)"] == maxVar)
      x = x[,-j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
backwardElimination(training_set, SL)

