###############################
# TEST 1 - H0: WEAK GROUP SKEW
###############################
source("../../../all-func.R")
set.seed(0)

# sample size
n <- 1000
# number of bootstrap replication
num_bs <- 1000
# buffer used to construct candidate values of (R, B)
buffer <- sqrt(log(n))/sqrt(n)
# size for a preliminary grid of candidate values
num_grid <- 100
# final number of candidate values to test
cand_size <- 2000
# optimizer
optimizer <- "grid"
# whether there is kink (if FALSE, a simplified limit distribution is used for bootstrap)
kink <- TRUE
# number of MC iter
num_MC <- 1000

# initialize vectors to store results
rej_at_truth_balance <- c()
rej_at_truth_rskew <- c()
rej_at_est_balance <- c()
rej_at_est_rskew <- c()
rej_skew_balance <- c()
rej_skew_rskew <- c()

# loading true values simulated by script `simulation-DGPplot.Rmd`
truth_balance <- read.csv("../../truth_balance.csv")
true_sf_balance <- as.numeric(c(truth_balance[1, 1:2], truth_balance[2, 1:2]))

truth_rskew <- read.csv("../../truth_rskew.csv")
true_sf_rskew <- as.numeric(c(truth_rskew[1, 1:2], truth_rskew[2, 1:2]))

### ------- START OF SIMULATION ------- ###
for (t in 1:num_MC){
  print(paste0("Iter (n=", n, ", B=", num_bs, ", kink=", kink, "):  ", t))
  print(paste0("Gaussian grid, opt=", optimizer))
  start <- Sys.time()

  # each MC iter draws a new copy of data
  test_data <- testDGP(n)

  ### BALANCED DGP -------
  # estimate nuisance parameters
  est_nuisance_balance <- nuisance(X=test_data$X, G=test_data$G, Y=test_data$Y_balance)

  ## estimating R and B
  est_sf_balance <- support_function(X=test_data$X,
                                     G=test_data$G,
                                     Y=test_data$Y_balance,
                                     est_nuisance=est_nuisance_balance,
                                     q=matrix(c(-1, 0,
                                                -1, 0,
                                                0, -1,
                                                0, -1), byrow=TRUE, ncol=2),
                                     v=matrix(c(1, 0,
                                                0, 1,
                                                1, 0,
                                                0, 1), byrow=TRUE, ncol=2))

  est_sf_balance <- est_sf_balance$est_sf
  est_R1_balance <- est_sf_balance[1]
  est_R2_balance <- est_sf_balance[2]
  est_B1_balance <- est_sf_balance[3]
  est_B2_balance <- est_sf_balance[4]

  ## constructing confidence sets for R and B
  # first define parameter space of R and B by using the estimated R and B
  # for the group-balanced config
  grid_R_balance <- data.matrix(
    expand.grid(rnorm(num_grid,
                      mean=est_R1_balance,
                      sd=buffer),
                rnorm(num_grid,
                      mean=est_R2_balance,
                      sd=buffer))
  )


  grid_R_balance <- rbind(c(true_sf_balance[1], true_sf_balance[2]),
                          c(est_R1_balance, est_R2_balance),
                          grid_R_balance[sample(1:nrow(grid_R_balance),
                                                cand_size, replace = FALSE),])

  grid_B_balance <- data.matrix(
    expand.grid(rnorm(num_grid,
                      mean=est_B1_balance,
                      sd=buffer),
                rnorm(num_grid,
                      mean=est_B2_balance,
                      sd=buffer))
  )

  grid_B_balance <- rbind(c(true_sf_balance[3], true_sf_balance[4]),
                          c(est_B1_balance, est_B2_balance),
                          grid_B_balance[sample(1:nrow(grid_B_balance),
                                                cand_size, replace = FALSE),])

  test1_balance <- CS_RB(R1=grid_R_balance[,1],
                         R2=grid_R_balance[,2],
                         B1=grid_B_balance[,1],
                         B2=grid_B_balance[,2],
                         est_nuisance=est_nuisance_balance,
                         num_bstp_rep=num_bs,
                         X=test_data$X,
                         G=test_data$G,
                         Y=test_data$Y_balance,
                         optimizer=optimizer,
                         kink=kink)

  # set of indices of candidate (R,B) not rejected
  CS_index_balance <- which(test1_balance$rej==0)

  # result of the test, as per Step 2, Procedure 1 of Liu & Molinari (2024)
  test1_rej_balance <- as.numeric(max((grid_R_balance[CS_index_balance,1]-grid_R_balance[CS_index_balance,2])*(grid_B_balance[CS_index_balance,1]-grid_B_balance[CS_index_balance,2])) < 0)

  ### SKEWED DGP -------
  # estimate nuisance parameters
  est_nuisance_rskew <- nuisance(X=test_data$X, G=test_data$G, Y=test_data$Y_rskew)

  ## estimating R and B
  est_sf_rskew <- support_function(X=test_data$X,
                                   G=test_data$G,
                                   Y=test_data$Y_rskew,
                                   est_nuisance=est_nuisance_rskew,
                                   q=matrix(c(-1, 0,
                                              -1, 0,
                                              0, -1,
                                              0, -1), byrow=TRUE, ncol=2),
                                   v=matrix(c(1, 0,
                                              0, 1,
                                              1, 0,
                                              0, 1), byrow=TRUE, ncol=2))

  est_sf_rskew <- est_sf_rskew$est_sf
  est_R1_rskew <- est_sf_rskew[1]
  est_R2_rskew <- est_sf_rskew[2]
  est_B1_rskew <- est_sf_rskew[3]
  est_B2_rskew <- est_sf_rskew[4]

  ## constructing confidence sets for R and B
  # first define parameter space of R and B by using the estimated R and B
  # for the group-rskewed config
  grid_R_rskew <- data.matrix(
    expand.grid(rnorm(num_grid,
                      mean=est_R1_rskew,
                      sd=buffer),
                rnorm(num_grid,
                      mean=est_R2_rskew,
                      sd=buffer))
  )


  grid_R_rskew <- rbind(c(true_sf_rskew[1], true_sf_rskew[2]),
                        c(est_R1_rskew, est_R2_rskew),
                        grid_R_rskew[sample(1:nrow(grid_R_rskew),
                                            cand_size, replace = FALSE),])

  grid_B_rskew <- data.matrix(
    expand.grid(rnorm(num_grid,
                      mean=est_B1_rskew,
                      sd=buffer),
                rnorm(num_grid,
                      mean=est_B2_rskew,
                      sd=buffer))
  )

  grid_B_rskew <- rbind(c(true_sf_rskew[3], true_sf_rskew[4]),
                        c(est_B1_rskew, est_B2_rskew),
                        grid_B_rskew[sample(1:nrow(grid_B_rskew),
                                            cand_size, replace = FALSE),])

  test1_rskew <- CS_RB(R1=grid_R_rskew[,1],
                       R2=grid_R_rskew[,2],
                       B1=grid_B_rskew[,1],
                       B2=grid_B_rskew[,2],
                       est_nuisance=est_nuisance_rskew,
                       num_bstp_rep=num_bs,
                       X=test_data$X,
                       G=test_data$G,
                       Y=test_data$Y_rskew,
                       optimizer=optimizer,
                       kink=kink)

  # set of indices of candidate (R,B) not rejected
  CS_index_rskew <- which(test1_rskew$rej==0)

  # result of the test, as per Step 2, Procedure 1 of Liu & Molinari (2024)
  test1_rej_rskew <- as.numeric(max((grid_R_rskew[CS_index_rskew,1]-grid_R_rskew[CS_index_rskew,2])*(grid_B_rskew[CS_index_rskew,1]-grid_B_rskew[CS_index_rskew,2])) < 0)

  rej_at_truth_balance <- append(rej_at_truth_balance, test1_balance$rej[1])
  rej_at_est_balance <- append(rej_at_est_balance, test1_balance$rej[2])
  rej_at_truth_rskew <- append(rej_at_truth_rskew, test1_rskew$rej[1])
  rej_at_est_rskew <- append(rej_at_est_rskew, test1_rskew$rej[2])
  rej_skew_balance <- append(rej_skew_balance, test1_rej_balance)
  rej_skew_rskew <- append(rej_skew_rskew, test1_rej_rskew)

  end <- Sys.time()

  print(paste0("Group-balanced config: "))
  print(paste0("Frequency that true (R,B) lie outside of CS: ", mean(rej_at_truth_balance)))
  print(paste0("Frequency that estimated (R,B) lie outside of CS: ", mean(rej_at_est_balance)))
  print(paste0("Frequency that H_0: WEAK GROUP-SKEW is rejected: ", mean(rej_skew_balance)))
  print(paste0("r-skewed config: "))
  print(paste0("Frequency that true (R,B) lie outside of CS: ", mean(rej_at_truth_rskew)))
  print(paste0("Frequency that estimated (R,B) lie outside of CS: ", mean(rej_at_est_rskew)))
  print(paste0("Frequency that H_0: WEAK GROUP-SKEW is rejected: ", mean(rej_skew_rskew)))

  print(end-start)
}


