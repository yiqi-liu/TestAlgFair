###############################
# TEST 2 - H0: THERE IS NO LDA
###############################
source("../../../all-func.R")
set.seed(0)

## first train the given logit algorithm being evaluated
train_data1 <- testDGP(5000)
train_data2 <- testDGP(5000)

train_X <- rbind(train_data1$X, train_data2$X)
train_Y <- append(train_data1$Y_balance, train_data2$Y_rskew)

train_data <- data.frame(cbind(train_X, train_Y))
colnames(train_data) <- append(sapply(1:20, function(num){paste0("X",num)}), "Y")

alg_logit <- glm(Y ~ ., data = train_data, family=binomial())

## set up -------------------------
# sample size
n <- 1000
optimizer <- "grid"
kink <- TRUE # whether there is kink (if FALSE, a simplified limit distribution is used for bootstrap)
num_bs <- 1000 # number of bootstrap replications
num_MC <- 1000 # number of MC iterations

# initialize vectors to store results
rej_at_R_balance <- c()
rej_at_B_balance <- c()
rej_at_FAdominated_balance <- c()
rej_at_logit_balance <- c()

rej_at_R_rskew <- c()
rej_at_B_rskew <- c()
rej_at_FAdominated_rskew <- c()
rej_at_logit_rskew <- c()

# loading true values simulated by script `simulation-DGPplot.Rmd`
truth_balance <- read.csv("../../truth_balance.csv")
true_RB_balance <- data.matrix(truth_balance[1:2, 1:2])
FAdominated_e_balance <- matrix(colMeans(true_RB_balance), ncol=2)

truth_rskew <- read.csv("../../truth_rskew.csv")
true_RB_rskew <- data.matrix(truth_rskew[1:2, 1:2])
FAdominated_e_rskew <- matrix(colMeans(true_RB_rskew), ncol=2)

### ------- START OF SIMULATION ------- ###
for (t in 1:num_MC){
  print(paste0("Iter (n=", n, ", B=", num_bs, ", opt=", optimizer, "): ", t))
  print(paste0("kink=", kink))

  start <- Sys.time()
  # each MC iter draws a new copy of data
  test_data <- testDGP(n)

  # get predictions from the logit algorithm
  eval_data_X <- data.frame(test_data$X)
  colnames(eval_data_X) <- sapply(1:20, function(num){paste0("X",num)})
  alg_pred <- matrix(predict.glm(alg_logit, eval_data_X, type="response"), nrow=1)

  ### BALANCED DGP -------
  # estimate nuisance parameters
  est_nuisance_balance <- nuisance(X=test_data$X, G=test_data$G, Y=test_data$Y_balance)

  # results for the non-estimated e^*
  result <- test_LDA(target_e=rbind(true_RB_balance, FAdominated_e_balance),
                     X=test_data$X,
                     G=test_data$G,
                     Y=test_data$Y_balance,
                     est_nuisance=est_nuisance_balance,
                     num_bstp_rep=num_bs,
                     optimizer=optimizer,
                     kink=kink)
  rej_at_R_balance <- append(rej_at_R_balance, result$rej[1])
  rej_at_B_balance <- append(rej_at_B_balance, result$rej[2])
  rej_at_FAdominated_balance <- append(rej_at_FAdominated_balance, result$rej[3])

  # result for the estimated logit e^*
  result <- test_LDA(target_alg_preds=alg_pred,
                     X=test_data$X,
                     G=test_data$G,
                     Y=test_data$Y_balance,
                     est_nuisance=est_nuisance_balance,
                     num_bstp_rep=num_bs,
                     optimizer=optimizer,
                     kink=kink)
  rej_at_logit_balance <- append(rej_at_logit_balance, result$rej[1])

  ### SKEWED DGP -------
  # estimate nuisance parameters
  est_nuisance_rskew <- nuisance(X=test_data$X, G=test_data$G, Y=test_data$Y_rskew)

  # results for the non-estimated e^*
  result <- test_LDA(target_e=rbind(true_RB_rskew, FAdominated_e_rskew),
                     X=test_data$X,
                     G=test_data$G,
                     Y=test_data$Y_rskew,
                     est_nuisance=est_nuisance_rskew,
                     num_bstp_rep=num_bs,
                     optimizer=optimizer,
                     kink=kink)
  rej_at_R_rskew <- append(rej_at_R_rskew, result$rej[1])
  rej_at_B_rskew <- append(rej_at_B_rskew, result$rej[2])
  rej_at_FAdominated_rskew <- append(rej_at_FAdominated_rskew, result$rej[3])

  # result for the estimated logit e^*
  result <- test_LDA(target_alg_preds=alg_pred,
                     X=test_data$X,
                     G=test_data$G,
                     Y=test_data$Y_rskew,
                     est_nuisance=est_nuisance_rskew,
                     num_bstp_rep=num_bs,
                     optimizer=optimizer,
                     kink=kink)
  rej_at_logit_rskew <- append(rej_at_logit_rskew, result$rej[1])
  end <- Sys.time()

  print(paste0("Group-balanced config: "))
  print(paste0("Frequency that `H0: no LDA to true R` is rejected: ", mean(rej_at_R_balance)))
  print(paste0("Frequency that `H0: no LDA to true B` is rejected: ", mean(rej_at_B_balance)))
  print(paste0("Frequency that `H0: no LDA to an FAdominated point` is rejected: ", mean(rej_at_FAdominated_balance)))
  print(paste0("Frequency that `H0: no LDA to alg_logit` is rejected: ", mean(rej_at_logit_balance)))

  print(paste0("r-skewed config: "))
  print(paste0("Frequency that `H0: no LDA to true R` is rejected: ", mean(rej_at_R_rskew)))
  print(paste0("Frequency that `H0: no LDA to true B` is rejected: ", mean(rej_at_B_rskew)))
  print(paste0("Frequency that `H0: no LDA to an FAdominated point` is rejected: ", mean(rej_at_FAdominated_rskew)))
  print(paste0("Frequency that `H0: no LDA to alg_logit` is rejected: ", mean(rej_at_logit_rskew)))
  print(end-start)
}
