#########################
# TEST 3 - DISTANCE TO F
#########################
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
optimizer <- "BFGS"
kink <- TRUE # whether there is kink (if FALSE, a simplified limit distribution is used for bootstrap)
num_bs <- 1000 # number of bootstrap replications
num_MC <- 1000 # number of MC iterations
num_grid <- 2000 # number of candidate values of (e, F) tested
# distance function
rho <- function(a1, a2, b1, b2) { # squared Euclidean distance
  return((a1-b1)^2+(a2-b2)^2)
}

# initialize vectors to store results
rej_at_R_balance <- c()
rej_at_B_balance <- c()
rej_at_D_balance <- c()
rej_at_logit_balance <- c()

rej_at_R_rskew <- c()
rej_at_B_rskew <- c()
rej_at_D_rskew <- c()
rej_at_logit_rskew <- c()

noAbove_balance <- c()
noBelow_balance <- c()
noAbove_rskew <- c()
noBelow_rskew <- c()

## loading true values simulated by script `simulation-DGPplot.Rmd`
# BALANCED DGP ------------------
truth_balance <- read.csv("../../truth_balance.csv")
# true F
true_F_balance <- as.numeric(truth_balance[3, 1:2])
# true R, B, and D=(R+B)/2
true_RBD_balance <-  data.matrix(rbind(truth_balance[1, 1:2],
                                       truth_balance[2, 1:2],
                                       colMeans(truth_balance[1:2, 1:2])))
# true e^*
true_logit_e_balance <- as.numeric(truth_balance[4, 1:2])
# true rho(e^*, F)
true_logit_dist_balance <- rho(true_F_balance[1],
                               true_F_balance[2],
                               true_logit_e_balance[1],
                               true_logit_e_balance[2])
# true rho(R, F)
true_R_dist_balance <- rho(true_F_balance[1],
                           true_F_balance[2],
                           true_RBD_balance[1,1],
                           true_RBD_balance[1,2])
# true rho(B, F)
true_B_dist_balance <- rho(true_F_balance[1],
                           true_F_balance[2],
                           true_RBD_balance[2,1],
                           true_RBD_balance[2,2])
# true rho(D, F)
true_D_dist_balance <- rho(true_F_balance[1],
                           true_F_balance[2],
                           true_RBD_balance[3,1],
                           true_RBD_balance[3,2])

# SKEWED DGP ------------------
truth_rskew <- read.csv("../../truth_rskew.csv")
# true F
true_F_rskew <- as.numeric(truth_rskew[3, 1:2])
# true R, B, and D=(R+B)/2
true_RBD_rskew <-  data.matrix(rbind(truth_rskew[1, 1:2],
                                     truth_rskew[2, 1:2],
                                     colMeans(truth_rskew[1:2, 1:2])))
# true e^*
true_logit_e_rskew <- as.numeric(truth_rskew[4, 1:2])
# true rho(e^*, F)
true_logit_dist_rskew <- rho(true_F_rskew[1],
                             true_F_rskew[2],
                             true_logit_e_rskew[1],
                             true_logit_e_rskew[2])
# true rho(R, F)
true_R_dist_rskew <- rho(true_F_rskew[1],
                         true_F_rskew[2],
                         true_RBD_rskew[1,1],
                         true_RBD_rskew[1,2])
# true rho(B, F)
true_B_dist_rskew <- rho(true_F_rskew[1],
                         true_F_rskew[2],
                         true_RBD_rskew[2,1],
                         true_RBD_rskew[2,2])
# true rho(D, F)
true_D_dist_rskew <- rho(true_F_rskew[1],
                         true_F_rskew[2],
                         true_RBD_rskew[3,1],
                         true_RBD_rskew[3,2])

### ------- START OF SIMULATION ------- ###
for (t in 1:num_MC){
  print(paste0("Iter (n=", n, ", B=", num_bs, ", opt=", optimizer, ", kink=", kink, "): ", t))
  start <- Sys.time()

  # each MC iter draws a new copy of data
  test_data <- testDGP(n)

  # get predictions from the logit algorithm
  eval_data_X <- data.frame(test_data$X)
  colnames(eval_data_X) <- sapply(1:20, function(num){paste0("X",num)})
  alg_preds <- matrix(predict.glm(alg_logit, eval_data_X, type="response"), nrow=1)

  ### BALANCED DGP -------
  # estimate nuisance parameters
  est_nuisance_balance <- nuisance(X=test_data$X, G=test_data$G, Y=test_data$Y_balance)

  # estimate F45
  estimate_F45 <- est_F45(X=test_data$X, G=test_data$G, Y=test_data$Y_balance,
                          est_nuisance=est_nuisance_balance, optimizer="SGD")

  # create parameter space of candidate (e, F)
  cand_eF <- create_candidate_eF(X=test_data$X,
                                 G=test_data$G,
                                 Y=test_data$Y_balance,
                                 est_nuisance=est_nuisance_balance,
                                 est_F45_coord=estimate_F45$F45,
                                 num_grid=5*num_grid,
                                 alg_preds=alg_preds)

  F1_above_balance <- cand_eF$cand_F_above[,1]
  F2_above_balance <- cand_eF$cand_F_above[,2]
  e1_above_balance <- cand_eF$cand_e_above[,1]
  e2_above_balance <- cand_eF$cand_e_above[,2]

  if (is.null(cand_eF$cand_e_above)){
    noAbove_balance <- append(noAbove_balance, 1)
  } else{
    # only take non-negative candidate (e, F)
    nonneg_ind <- sample(which(F1_above_balance >= 0 &
                                 F2_above_balance >= 0 &
                                 e1_above_balance >= 0 &
                                 e2_above_balance >= 0), size=num_grid)
    F1_above_balance <- cand_F_above[nonneg_ind]
    F2_above_balance <- cand_F_above[nonneg_ind]
    e1_above_balance <- matrix(e1_above_balance[nonneg_ind], nrow=1)
    e2_above_balance <- matrix(e2_above_balance[nonneg_ind], nrow=1)
  }

  F1_below_balance <- cand_eF$cand_F_below[,1]
  F2_below_balance <- cand_eF$cand_F_below[,2]
  e1_below_balance <- cand_eF$cand_e_below[,1]
  e2_below_balance <- cand_eF$cand_e_below[,2]

  if (is.null(cand_eF$cand_e_below)){
    noBelow_balance <- append(noBelow_balance, 1)
  } else{
    # only take non-negative candidate (e, F)
    nonneg_ind <- sample(which(F1_below_balance >= 0 &
                                 F2_below_balance >= 0 &
                                 e1_below_balance >= 0 &
                                 e2_below_balance >= 0), size=num_grid)
    F1_below_balance <- cand_F_below[nonneg_ind]
    F2_below_balance <- cand_F_below[nonneg_ind]
    e1_below_balance <- matrix(e1_below_balance[nonneg_ind], nrow=1)
    e2_below_balance <- matrix(e2_below_balance[nonneg_ind], nrow=1)
  }

  nonneg_ind <- sample(which(cand_eF$cand_F_cross[,1] > 0 &
                               cand_eF$cand_e_cross[,1] > 0 &
                               cand_eF$cand_e_cross[,2] > 0), size=num_grid)

  cand_F_cross_balance <- rbind(true_F_balance, cand_eF$cand_F_cross[nonneg_ind, ])
  cand_e_cross_balance <- rbind(true_logit_e_balance, cand_eF$cand_e_cross[nonneg_ind, ])

  ## test at alg_logit -----
  # when the feasible set and the 45-degree line are disjoint
  test_disjoint_balance <- CS_eF_disjoint(target_alg_preds=alg_preds,
                                          X=test_data$X,
                                          G=test_data$G,
                                          Y=test_data$Y_balance,
                                          est_nuisance=est_nuisance_balance,
                                          F1_above=F1_above_balance,
                                          F2_above=F2_above_balance,
                                          F1_below=F1_below_balance,
                                          F2_below=F2_below_balance,
                                          e1_above=e1_above_balance,
                                          e2_above=e2_above_balance,
                                          e1_below=e1_below_balance,
                                          e2_below=e2_below_balance,
                                          num_bstp_rep=num_bs,
                                          optimizer=optimizer,
                                          kink=kink)

  # CS_n^+
  CS_above_balance <- cbind(F1_above_balance,
                            F2_above_balance,
                            e1_above_balance,
                            e2_above_balance)[which(test_disjoint_balance$rej_above==0), ,  drop=FALSE]

  # CS_n^-
  CS_below_balance <- cbind(F1_below_balance,
                            F2_below_balance,
                            e1_below_balance,
                            e2_below_balance)[which(test_disjoint_balance$rej_below==0), ,  drop=FALSE]

  # when the feasible set and the 45-degree line have intersection
  test_cross_balance <- CS_eF_cross(target_alg_preds=alg_preds,
                                    X=test_data$X,
                                    G=test_data$G,
                                    Y=test_data$Y_balance,
                                    est_nuisance=est_nuisance_balance,
                                    F1=cand_F_cross_balance[,1],
                                    F2=cand_F_cross_balance[,2],
                                    e1=matrix(cand_e_cross_balance[,1], nrow=1),
                                    e2=matrix(cand_e_cross_balance[,2], nrow=1),
                                    num_bstp_rep=num_bs,
                                    dist_func="euclidean-sq",
                                    optimizer=optimizer,
                                    init_par=estimate_F45$q_F45[2],
                                    kink=kink)

  # CS_n^{45}
  CS_cross_balance <- cbind(cand_F_cross_balance[which(test_cross_balance$rej==0), , drop=FALSE],
                            cand_e_cross_balance[which(test_cross_balance$rej==0), , drop=FALSE])


  # the final confidence set, CS_n^{\rho(\tilde{e}, \tilde{F})}, is the union of CS_n^+, CS_n^-, and CS_n^{45}
  CS_balance <- rbind(CS_cross_balance, CS_above_balance, CS_below_balance)
  CS_dist_balance <- c(min(rho(CS_balance[,1], CS_balance[,2], CS_balance[,3], CS_balance[,4])),
                       max(rho(CS_balance[,1], CS_balance[,2], CS_balance[,3], CS_balance[,4])))

  rej_at_logit_balance <- append(rej_at_logit_balance,
                                 as.numeric(true_logit_dist_balance < CS_dist_balance[1] |  true_logit_dist_balance > CS_dist_balance[2]))

  ## test at true R, B, and D -----
  # when the feasible set and the 45-degree line are disjoint
  test_disjoint_balance <- CS_eF_disjoint(target_e=true_RBD_balance,
                                          X=test_data$X,
                                          G=test_data$G,
                                          Y=test_data$Y_balance,
                                          est_nuisance=est_nuisance_balance,
                                          F1_above=F1_above_balance,
                                          F2_above=F2_above_balance,
                                          F1_below=F1_below_balance,
                                          F2_below=F2_below_balance,
                                          num_bstp_rep=num_bs,
                                          optimizer=optimizer,
                                          kink=kink)

  # CS_n^+
  CS_above_balance <- cbind(F1_above_balance, F2_above_balance)[which(test_disjoint_balance$rej_above==0), , drop=FALSE]

  # CS_n^-
  CS_below_balance <- cbind(F1_below_balance, F2_below_balance)[which(test_disjoint_balance$rej_below==0), , drop=FALSE]

  # when the feasible set and the 45-degree line have intersection
  test_cross_balance <- CS_eF_cross(target_e=true_RBD_balance,
                                    X=test_data$X,
                                    G=test_data$G,
                                    Y=test_data$Y_balance,
                                    est_nuisance=est_nuisance_balance,
                                    F1=cand_F_cross_balance[,1],
                                    F2=cand_F_cross_balance[,2],
                                    num_bstp_rep=num_bs,
                                    dist_func="euclidean-sq",
                                    optimizer=optimizer,
                                    init_par=estimate_F45$q_F45[2],
                                    kink=kink)

  # CS_n^{45}
  CS_cross_R_balance <- cand_F_cross_balance[which(test_cross_balance$rej[1,]==0), ,drop=FALSE]
  CS_cross_B_balance <- cand_F_cross_balance[which(test_cross_balance$rej[2,]==0), ,drop=FALSE]
  CS_cross_D_balance <- cand_F_cross_balance[which(test_cross_balance$rej[3,]==0), ,drop=FALSE]

  # the final confidence set, CS_n^{\rho(\tilde{e}, \tilde{F})}, is the union of CS_n^+, CS_n^-, and CS_n^{45}
  CS_R_balance <- rbind(CS_cross_R_balance, CS_above_balance, CS_below_balance)
  CS_B_balance <- rbind(CS_cross_B_balance, CS_above_balance, CS_below_balance)
  CS_D_balance <- rbind(CS_cross_D_balance, CS_above_balance, CS_below_balance)

  # CS for rho(e, F)
  CS_dist_R_balance <- c(min(rho(CS_R_balance[,1], CS_R_balance[,2], true_RBD_balance[1,1], true_RBD_balance[1,2])), max(rho(CS_R_balance[,1], CS_R_balance[,2], true_RBD_balance[1,1], true_RBD_balance[1,2])))

  CS_dist_B_balance <- c(min(rho(CS_B_balance[,1], CS_B_balance[,2], true_RBD_balance[2,1], true_RBD_balance[2,2])), max(rho(CS_B_balance[,1], CS_B_balance[,2], true_RBD_balance[2,1], true_RBD_balance[2,2])))

  CS_dist_D_balance <- c(min(rho(CS_D_balance[,1], CS_D_balance[,2], true_RBD_balance[3,1], true_RBD_balance[3,2])), max(rho(CS_D_balance[,1], CS_D_balance[,2], true_RBD_balance[3,1], true_RBD_balance[3,2])))

  # collect results
  rej_at_R_balance <- append(rej_at_R_balance,
                             as.numeric(true_R_dist_balance < CS_dist_R_balance[1] | true_R_dist_balance > CS_dist_R_balance[2]))
  rej_at_B_balance <- append(rej_at_B_balance,
                             as.numeric(true_B_dist_balance < CS_dist_B_balance[1] | true_B_dist_balance > CS_dist_B_balance[2]))
  rej_at_D_balance <- append(rej_at_D_balance,
                             as.numeric(true_D_dist_balance < CS_dist_D_balance[1] | true_D_dist_balance > CS_dist_D_balance[2]))


  ### SKEWED DGP -------
  # estimate nuisance parameters
  est_nuisance_rskew <- nuisance(X=test_data$X, G=test_data$G, Y=test_data$Y_rskew)

  # estimate F45
  estimate_F45 <- est_F45(X=test_data$X, G=test_data$G, Y=test_data$Y_rskew,
                          est_nuisance=est_nuisance_rskew, optimizer="SGD")

  # create parameter space of candidate (e, F)
  cand_eF <- create_candidate_eF(X=test_data$X,
                                 G=test_data$G,
                                 Y=test_data$Y_rskew,
                                 est_nuisance=est_nuisance_rskew,
                                 est_F45_coord=estimate_F45$F45,
                                 num_grid=5*num_grid,
                                 alg_preds=alg_preds)

  F1_above_rskew <- cand_eF$cand_F_above[,1]
  F2_above_rskew <- cand_eF$cand_F_above[,2]
  e1_above_rskew <- cand_eF$cand_e_above[,1]
  e2_above_rskew <- cand_eF$cand_e_above[,2]

  if (is.null(cand_eF$cand_e_above)){
    noAbove_rskew <- append(noAbove_rskew, 1)
  } else{
    # only take non-negative candidate (e, F)
    nonneg_ind <- sample(which(F1_above_rskew >= 0 &
                                 F2_above_rskew >= 0 &
                                 e1_above_rskew >= 0 &
                                 e2_above_rskew >= 0), size=num_grid)
    F1_above_rskew <- cand_F_above[nonneg_ind]
    F2_above_rskew <- cand_F_above[nonneg_ind]
    e1_above_rskew <- matrix(e1_above_rskew[nonneg_ind], nrow=1)
    e2_above_rskew <- matrix(e2_above_rskew[nonneg_ind], nrow=1)
  }

  F1_below_rskew <- cand_eF$cand_F_below[,1]
  F2_below_rskew <- cand_eF$cand_F_below[,2]
  e1_below_rskew <- cand_eF$cand_e_below[,1]
  e2_below_rskew <- cand_eF$cand_e_below[,2]

  if (is.null(cand_eF$cand_e_below)){
    noBelow_rskew <- append(noBelow_rskew, 1)
  } else{
    # only take non-negative candidate (e, F)
    nonneg_ind <- sample(which(F1_below_rskew >= 0 &
                                 F2_below_rskew >= 0 &
                                 e1_below_rskew >= 0 &
                                 e2_below_rskew >= 0), size=num_grid)
    F1_below_rskew <- cand_F_below[nonneg_ind]
    F2_below_rskew <- cand_F_below[nonneg_ind]
    e1_below_rskew <- matrix(e1_below_rskew[nonneg_ind], nrow=1)
    e2_below_rskew <- matrix(e2_below_rskew[nonneg_ind], nrow=1)
  }

  nonneg_ind <- sample(which(cand_eF$cand_F_cross[,1] > 0 &
                               cand_eF$cand_e_cross[,1] > 0 &
                               cand_eF$cand_e_cross[,2] > 0), size=num_grid)

  cand_F_cross_rskew <- rbind(true_F_rskew, cand_eF$cand_F_cross[nonneg_ind, ])
  cand_e_cross_rskew <- rbind(true_logit_e_rskew, cand_eF$cand_e_cross[nonneg_ind, ])

  ## test at alg_logit -----
  # when the feasible set and the 45-degree line are disjoint
  test_disjoint_rskew <- CS_eF_disjoint(target_alg_preds=alg_preds,
                                        X=test_data$X,
                                        G=test_data$G,
                                        Y=test_data$Y_rskew,
                                        est_nuisance=est_nuisance_rskew,
                                        F1_above=F1_above_rskew,
                                        F2_above=F2_above_rskew,
                                        F1_below=F1_below_rskew,
                                        F2_below=F2_below_rskew,
                                        e1_above=e1_above_rskew,
                                        e2_above=e2_above_rskew,
                                        e1_below=e1_below_rskew,
                                        e2_below=e2_below_rskew,
                                        num_bstp_rep=num_bs,
                                        optimizer=optimizer,
                                        kink=kink)

  # CS_n^+
  CS_above_rskew <- cbind(F1_above_rskew,
                          F2_above_rskew,
                          e1_above_rskew,
                          e2_above_rskew)[which(test_disjoint_rskew$rej_above==0), ,drop=FALSE]

  # CS_n^-
  CS_below_rskew <- cbind(F1_below_rskew,
                          F2_below_rskew,
                          e1_below_rskew,
                          e2_below_rskew)[which(test_disjoint_rskew$rej_below==0), ,drop=FALSE]

  # when the feasible set and the 45-degree line have intersection
  test_cross_rskew <- CS_eF_cross(target_alg_preds=alg_preds,
                                  X=test_data$X,
                                  G=test_data$G,
                                  Y=test_data$Y_rskew,
                                  est_nuisance=est_nuisance_rskew,
                                  F1=cand_F_cross_rskew[,1],
                                  F2=cand_F_cross_rskew[,2],
                                  e1=matrix(cand_e_cross_rskew[,1], nrow=1),
                                  e2=matrix(cand_e_cross_rskew[,2], nrow=1),
                                  num_bstp_rep=num_bs,
                                  dist_func="euclidean-sq",
                                  optimizer=optimizer,
                                  init_par=estimate_F45$q_F45[2],
                                  kink=kink)

  # CS_n^{45}
  CS_cross_rskew <- cbind(cand_F_cross_rskew[which(test_cross_rskew$rej==0), ,drop=FALSE], cand_e_cross_rskew[which(test_cross_rskew$rej==0), , drop=FALSE])

  # the final confidence set, CS_n^{\rho(\tilde{e}, \tilde{F})}, is the union of CS_n^+, CS_n^-, and CS_n^{45}
  CS_rskew <- rbind(CS_cross_rskew, CS_above_rskew, CS_below_rskew)
  # CS for \rho(e, F)
  CS_dist_rskew <- c(min(rho(CS_rskew[,1], CS_rskew[,2], CS_rskew[,3], CS_rskew[,4])), max(rho(CS_rskew[,1], CS_rskew[,2], CS_rskew[,3], CS_rskew[,4])))

  # collect result
  rej_at_logit_rskew <- append(rej_at_logit_rskew,
                               as.numeric(true_logit_dist_rskew < CS_dist_rskew[1] |  true_logit_dist_rskew > CS_dist_rskew[2]))

  ## test at true R, B, and D -----
  # when the feasible set and the 45-degree line are disjoint
  test_disjoint_rskew <- CS_eF_disjoint(target_e=true_RBD_rskew,
                                        X=test_data$X,
                                        G=test_data$G,
                                        Y=test_data$Y_rskew,
                                        est_nuisance=est_nuisance_rskew,
                                        F1_above=F1_above_rskew,
                                        F2_above=F2_above_rskew,
                                        F1_below=F1_below_rskew,
                                        F2_below=F2_below_rskew,
                                        num_bstp_rep=num_bs,
                                        optimizer=optimizer,
                                        kink=kink)

  # CS_n^+
  CS_above_rskew <- cbind(F1_above_rskew, F2_above_rskew)[which(test_disjoint_rskew$rej_above==0), ,drop=FALSE]

  # CS_n^-
  CS_below_rskew <- cbind(F1_below_rskew, F2_below_rskew)[which(test_disjoint_rskew$rej_below==0), ,drop=FALSE]

  # when the feasible set and the 45-degree line have intersection
  test_cross_rskew <- CS_eF_cross(target_e=true_RBD_rskew,
                                  X=test_data$X,
                                  G=test_data$G,
                                  Y=test_data$Y_rskew,
                                  est_nuisance=est_nuisance_rskew,
                                  F1=cand_F_cross_rskew[,1],
                                  F2=cand_F_cross_rskew[,2],
                                  num_bstp_rep=num_bs,
                                  dist_func="euclidean-sq",
                                  optimizer=optimizer,
                                  init_par=estimate_F45$q_F45[2],
                                  kink=kink)

  # CS_n^{45}
  CS_cross_R_rskew <- cand_F_cross_rskew[which(test_cross_rskew$rej[1,]==0), , drop=FALSE]
  CS_cross_B_rskew <- cand_F_cross_rskew[which(test_cross_rskew$rej[2,]==0), , drop=FALSE]
  CS_cross_D_rskew <- cand_F_cross_rskew[which(test_cross_rskew$rej[3,]==0), , drop=FALSE]


  # the final confidence set, CS_n^{\rho(\tilde{e}, \tilde{F})}, is the union of CS_n^+, CS_n^-, and CS_n^{45}
  CS_R_rskew <- rbind(CS_cross_R_rskew, CS_above_rskew, CS_below_rskew)
  CS_B_rskew <- rbind(CS_cross_B_rskew, CS_above_rskew, CS_below_rskew)
  CS_D_rskew <- rbind(CS_cross_D_rskew, CS_above_rskew, CS_below_rskew)

  # CS for rho(e, F)
  CS_dist_R_rskew <- c(min(rho(CS_R_rskew[,1], CS_R_rskew[,2], true_RBD_rskew[1,1], true_RBD_rskew[1,2])), max(rho(CS_R_rskew[,1], CS_R_rskew[,2], true_RBD_rskew[1,1], true_RBD_rskew[1,2])))

  CS_dist_B_rskew <- c(min(rho(CS_B_rskew[,1], CS_B_rskew[,2], true_RBD_rskew[2,1], true_RBD_rskew[2,2])), max(rho(CS_B_rskew[,1], CS_B_rskew[,2], true_RBD_rskew[2,1], true_RBD_rskew[2,2])))

  CS_dist_D_rskew <- c(min(rho(CS_D_rskew[,1], CS_D_rskew[,2], true_RBD_rskew[3,1], true_RBD_rskew[3,2])), max(rho(CS_D_rskew[,1], CS_D_rskew[,2], true_RBD_rskew[3,1], true_RBD_rskew[3,2])))

  # collect results
  rej_at_R_rskew <- append(rej_at_R_rskew,
                           as.numeric(true_R_dist_rskew < CS_dist_R_rskew[1] | true_R_dist_rskew > CS_dist_R_rskew[2]))
  rej_at_B_rskew <- append(rej_at_B_rskew,
                           as.numeric(true_B_dist_rskew < CS_dist_B_rskew[1] | true_B_dist_rskew > CS_dist_B_rskew[2]))
  rej_at_D_rskew <- append(rej_at_D_rskew,
                           as.numeric(true_D_dist_rskew < CS_dist_D_rskew[1] | true_D_dist_rskew > CS_dist_D_rskew[2]))

  end <- Sys.time()

  print(paste0("Group-balanced config: "))
  print(paste0("Frequency that dist(R,F) is not covered: ", mean(rej_at_R_balance)))
  print(paste0("Frequency that dist(B,F) is not covered: ", mean(rej_at_B_balance)))
  print(paste0("Frequency that dist(FAdom,F) is not covered: ", mean(rej_at_D_balance)))
  print(paste0("Frequency that dist(e*,F) is not covered: ", mean(rej_at_logit_balance)))
  print(paste0("Frequency that CS_n^+ has no candidates: ", mean(noAbove_balance)))
  print(paste0("Frequency that CS_n^- has no candidates: ", mean(noBelow_balance)))

  print(paste0("r-skewed config: "))
  print(paste0("Frequency that dist(R,F) is not covered: ", mean(rej_at_R_rskew)))
  print(paste0("Frequency that dist(B,F) is not covered: ", mean(rej_at_B_rskew)))
  print(paste0("Frequency that dist(FAdom,F) is not covered: ", mean(rej_at_D_rskew)))
  print(paste0("Frequency that dist(e*,F) is not covered: ", mean(rej_at_logit_rskew)))
  print(paste0("Frequency that CS_n^+ has no candidates: ", mean(noAbove_rskew)))
  print(paste0("Frequency that CS_n^- has no candidates: ", mean(noBelow_rskew)))
  print(end-start)
}
