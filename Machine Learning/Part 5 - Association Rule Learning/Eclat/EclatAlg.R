# Eclat Algorithm

# Data Preprocessing dataset as sparse matrix
# install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat Algorithm on the Dataset
rules = eclat(data = dataset,
                parameter = list(support = round(4*7/7500,3), minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])