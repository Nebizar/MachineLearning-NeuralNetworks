# Upper Confidence Bound

# Importimg the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# UCB
N = NROW(dataset)
d = NCOL(dataset)
ads_selected = integer(0)
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0
for (n in 1:N){
  max_ub = 0
  ad = 0
  for (i in 1:d){
    if(numbers_of_selections[i]>0){
      avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = avg_reward + delta
    } else {
      upper_bound = 1e400
    }
    if(upper_bound > max_ub){
      max_ub = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n,ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selected',
     xlab = 'Advertisments',
     ylab = 'Number of selections')

