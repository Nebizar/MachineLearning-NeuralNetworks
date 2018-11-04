# Apriori

# Data Preprocessing
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
# Sparse Matrix
# install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori Algorithm on the Dataset
rules = apriori(data = dataset,
                parameter = list(support = round(4*7/7500,3), confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])