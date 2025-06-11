###############################
# TEST 2 - H0: THERE IS NO LDA
# sample size = 5000
###############################

### to replicate the .csv files, run each .R files in /test2n5k-split
# read .csv files and append
test2n5k <- read.csv("test2n5k-split/test2n5k-0.csv")
for (seed in 1:3){
  assign(
    paste0("test2n5k_", seed),
    read.csv(paste0("test2n5k-split/test2n5k-", seed, ".csv"))
    )
  test2n5k <- rbind(test2n5k, get(paste0("test2n5k_", seed)))
}

# summarize results
cat("Total number of MC replications: ", nrow(test2n5k), "\n",
    "--------- Group-balanced DGP --------- \n",
    "Frequency that `H0: no LDA to true R` is rejected: ", mean(test2n5k$rej_at_R_balance), "\n",
    "Frequency that `H0: no LDA to true B` is rejected: ", mean(test2n5k$rej_at_B_balance), "\n",
    "Frequency that `H0: no LDA to FA-dominated point` is rejected: ", mean(test2n5k$rej_at_FAdominated_balance), "\n",
    "Frequency that `H0: no LDA to alg_logit` is rejected: ", mean(test2n5k$rej_at_logit_balance), "\n",
    "--------- r-skewed DGP --------- \n",
    "Frequency that `H0: no LDA to true R` is rejected: ", mean(test2n5k$rej_at_R_rskew), "\n",
    "Frequency that `H0: no LDA to true B` is rejected: ", mean(test2n5k$rej_at_B_rskew), "\n",
    "Frequency that `H0: no LDA to FA-dominated point` is rejected: ", mean(test2n5k$rej_at_FAdominated_rskew), "\n",
    "Frequency that `H0: no LDA to alg_logit` is rejected: ", mean(test2n5k$rej_at_logit_rskew), "\n"
)
