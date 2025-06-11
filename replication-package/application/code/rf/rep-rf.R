source("../../../all-func.R")
library(ggplot2)
library(gridExtra)
library(grid)
library(concaveman)
library(rcdd)
set.seed(0)

## read data
# X covariates (#): demographic (8), comorbidity (34), cost (13), lab (90), med (4); TOTAL: 149.
df <- read.csv("../../data/all_Y_x_df.csv")
X <- df[, 2:150] # 149 covariates
# Y is defined to be 1{greater than 97-percentile active chronic conditions (6)}, as in LLMO
Y <- as.numeric(df$gagne_sum_t > quantile(df$gagne_sum_t, 0.97))
# group indicator; black=1 (or the r group in our notation)
G <- df$dem_race_black
# number of observations = 48784
n <- length(Y)

## generate directions for estimating the feasible set
num_q <- 1000 # number of directions

# grid of (q, v) used to estimate the support sets
grid_q_sf <- t(as.matrix(sapply(seq(0, 2*pi, length.out=num_q), function(rad){c(cos(rad), sin(rad))})))
grid_q <- rbind(grid_q_sf, grid_q_sf)
grid_v <- data.matrix(cbind(c(rep(1, each=num_q), rep(0, each=num_q)), c(rep(0, each=num_q), rep(1, each=num_q))))

# directions (q, v) that yield coordinates of R and B
RB_q <- matrix(c(-1, 0, -1, 0, 0, -1, 0, -1), byrow=TRUE, ncol=2)
RB_v <- matrix(c(1, 0, 0, 1, 1, 0, 0, 1), byrow=TRUE, ncol=2)

q <- rbind(grid_q, RB_q)
v <- rbind(grid_v, RB_v)

## initialize vectors to collect results
# estimates of the feasible set
estR1 <- c()
estR2 <- c()
estB1 <- c()
estB2 <- c()
estF <- c()

estR1_ints <- c()
estR2_ints <- c()
estB1_ints <- c()
estB2_ints <- c()
estF_ints <- c()

# test 1 (weak group skew) result
test1_rej <- c()

# test 2 (LDA) result
ts_orig <- c()
ts_tc <- c()
ts_ac <- c()
ts_acc <- c()

cv_orig <- c()
cv_tc <- c()
cv_ac <- c()
cv_acc <- c()

rej_orig <- c()
rej_tc <- c()
rej_ac <- c()
rej_acc <- c()

# test 3 (distance-to-F) result
dist_orig <- c()
dist_tc <- c()
dist_ac <- c()
dist_acc <- c()

CSl_orig <- c()
CSl_tc <- c()
CSl_ac <- c()
CSl_acc <- c()

CSu_orig <- c()
CSu_tc <- c()
CSu_ac <- c()
CSu_acc <- c()

for (rep in 1:20){
  start <- Sys.time()
  print(paste0("Current rep=", rep))

  ## ESTIMATING THE FEASIBLE SET ---------------
  # estimate the nuisance parameters using grf
  est_nuisance <- nuisance(X=X, G=G, Y=Y, method="grf",
                           seed=rep, num.trees=5000)

  # estimate the support sets
  sf <- support_function(X=X, G=G, Y=Y,
                         q=q, v=v, est_nuisance=est_nuisance)

  # estimate the F point
  estimate_F45 <- est_F45(Y=Y, X=X, G=G, est_nuisance=est_nuisance, optimizer="SGD")
  F45 <- estimate_F45$F45
  q_F45 <- estimate_F45$q_F45

  # CONSTRUCTING CS FOR THE FA-FRONTIER
  kappa_n <- sqrt(log(n))/sqrt(n) # tolerance

  # extract the estimated support points
  ss <- data.frame(e_r = sf$est_sf[1:num_q],
                   e_b = sf$est_sf[(num_q+1):(2*num_q)])

  # halfspaces defined by q1*e_r + q2*e_b <= est_sf
  halfspaces <- data.frame(
    q1 = grid_q_sf[,1],
    q2 = grid_q_sf[,2],
    sf_q = grid_q_sf[,1]*ss[,1]+grid_q_sf[,2]*ss[,2]
  )

  hrep <- makeH(a1=grid_q_sf, b1=halfspaces$sf_q)
  vertices <- scdd(hrep)$output[, 3:4]

  # compute the vertices of the feasible region
  vertices <- data.frame(e_r=vertices[,1], e_b=vertices[,2])

  # hyperplanes defined by q1*e_r + q2*e_b = est_sf
  hyperplanes <- halfspaces %>%
    mutate(slope = ifelse(q2 != 0, -q1 / q2, NA),
           intercept = ifelse(q2 != 0, sf_q / q2, NA))

  # these are the estimated R, B and F points
  plot_RBF <- data.frame(e_r = c(sf$est_sf[(2*num_q)+1],
                                 sf$est_sf[(2*num_q)+3],
                                 F45),
                         e_b = c(sf$est_sf[(2*num_q)+2],
                                 sf$est_sf[(2*num_q)+4],
                                 F45),
                         type = c("est R", "est B", "est F"))

  # take the convex hull of estimated support points
  conv_hull <- slice(vertices, chull(e_r, e_b))

  # intersection between the convex hull and the 45-degree line
  inters <- find_intersections(conv_hull)

  # these are the corresponding R, B and F points of the intersection of estimated supporting halfspaces
  plot_RBFvert <- data.frame(e_r = c(vertices[which.min(vertices$e_r), 1],
                                     vertices[which.min(vertices$e_b), 1],
                                     inters[which.min(inters$e_r), 1]),
                             e_b = c(vertices[which.min(vertices$e_r), 2],
                                     vertices[which.min(vertices$e_b), 2],
                                     inters[which.min(inters$e_b), 2]),
                             type = c("est R", "est B", "est F"))

  estR1 <- append(estR1, plot_RBF[1,1])
  estR2 <- append(estR2, plot_RBF[1,2])
  estB1 <- append(estB1, plot_RBF[2,1])
  estB2 <- append(estB2, plot_RBF[2,2])
  estF <- append(estF, plot_RBF[3,1])

  estR1_ints <- append(estR1_ints, plot_RBFvert[1,1])
  estR2_ints <- append(estR2_ints, plot_RBFvert[1,2])
  estB1_ints <- append(estB1_ints, plot_RBFvert[2,1])
  estB2_ints <- append(estB2_ints, plot_RBFvert[2,2])
  estF_ints <- append(estF_ints, plot_RBFvert[3,1])

  ### TEST 1 -----------------------------------
  ## first, construct confidence sets for (R, B)
  # define parameter space of (R, B) by using the estimated R and B
  num_grid <- 100 # (num_grid)^2 is number of parameter values being tested
  # candidate values for R
  grid_R <- data.matrix(expand.grid(rnorm(num_grid,
                                          mean=plot_RBFvert[1,1],
                                          sd=kappa_n),
                                    rnorm(num_grid,
                                          mean=plot_RBFvert[1,2],
                                          sd=kappa_n)))

  grid_R <- rbind(c(plot_RBFvert[1,1], plot_RBFvert[1,2]),
                  grid_R)

  # candidate values for B
  grid_B <- data.matrix(expand.grid(rnorm(num_grid,
                                          mean=plot_RBFvert[2,1],
                                          sd=kappa_n),
                                    rnorm(num_grid,
                                          mean=plot_RBFvert[2,2],
                                          sd=kappa_n)))

  grid_B <- rbind(c(plot_RBFvert[2,1], plot_RBFvert[2,2]),
                  grid_B)

  # test without the no-kink assumption
  test1 <- CS_RB(X=X, G=G, Y=Y,
                 est_nuisance=est_nuisance,
                 R1=grid_R[,1],
                 R2=grid_R[,2],
                 B1=grid_B[,1],
                 B2=grid_B[,2],
                 optimizer="grid", grid_size=1000,
                 kink=TRUE, num_bstp_rep=1000)

  # set of indices of candidate (R,B) not rejected
  CS_index <- which(test1$rej==0)
  # result of the test, as per Step 2, Procedure 1 of Liu & Molinari (2024)
  test1_rej <- append(test1_rej, as.numeric(max((grid_R[CS_index,1]-grid_R[CS_index,2])*(grid_B[CS_index,1]-grid_B[CS_index,2])) < 0))

  R1=grid_R[,1]
  R2=grid_R[,2]
  B1=grid_B[,1]
  B2=grid_B[,2]

  ind <- which(test1$rej==0)

  ### TEST 2 --------------------------------
  ## import predictions from the 3 experimental algorithms trained in Obermeyer et al. (2019)
  # experimental algorithm that predicts total cost
  pred_tc <- read.csv("../../data/pred_log_cost_df.csv")
  alg_tc <- as.numeric(pred_tc$log_cost_t_hat > quantile(pred_tc$log_cost_t_hat, 0.97))
  # experimental algorithm that predicts avoidable cost
  pred_ac <- read.csv("../../data/pred_log_cost_avoidable_df.csv")
  alg_ac <- as.numeric(pred_ac$log_cost_avoidable_t_hat > quantile(pred_ac$log_cost_avoidable_t_hat, 0.97))
  # experimental algorithm that predicts number of active chronic conditions
  pred_acc <- read.csv("../../data/pred_gagne_sum_t_df.csv")
  alg_acc <- as.numeric(pred_acc$gagne_sum_t_hat > quantile(pred_acc$gagne_sum_t_hat, 0.97))

  # the original algorithm that returns risk scores
  alg_orig <- as.numeric(df$risk_score_t > quantile(df$risk_score_t, 0.97))

  test2 <- test_LDA(target_alg_preds=rbind(alg_orig, alg_tc, alg_ac, alg_acc),
                    X=X, G=G, Y=Y,
                    est_nuisance=est_nuisance,
                    num_bstp_rep=1000,
                    optimizer="grid",
                    kink=TRUE)

  ts_orig <- append(ts_orig, test2$LDA_stat[1])
  cv_orig <- append(cv_orig, test2$BScv[1])
  rej_orig <- as.numeric(ts_orig > cv_orig)

  ts_tc <- append(ts_tc, test2$LDA_stat[2])
  cv_tc <- append(cv_tc, test2$BScv[2])
  rej_tc <- as.numeric(ts_tc > cv_tc)

  ts_ac <- append(ts_ac, test2$LDA_stat[3])
  cv_ac <- append(cv_ac, test2$BScv[3])
  rej_ac <- as.numeric(ts_ac > cv_ac)

  ts_acc <- append(ts_acc, test2$LDA_stat[4])
  cv_acc <- append(cv_acc, test2$BScv[4])
  rej_acc <- as.numeric(ts_acc > cv_acc)

  ### TEST 3 -------------------
  # squared Euclidean distance
  rho <- function(a1, a2, b1, b2) {
    return((a1-b1)^2+(a2-b2)^2)
  }

  ## the original algorithm that returns risk scores ---
  # create parameter space of candidate (e, F)
  cand_eF_orig <- create_candidate_eF(X=X, G=G, Y=Y,
                                      est_nuisance=est_nuisance,
                                      est_F45_coord=F45,
                                      num_grid=10000,
                                      alg_preds=alg_orig,
                                      optimizer="BFGS",
                                      buffer=4*kappa_n)
  # only take non-negative candidate (e, F)
  nonneg_ind <- sample(which(cand_eF_orig$cand_F_cross[,1] > 0 & cand_eF_orig$cand_e_cross[,1] > 0 & cand_eF_orig$cand_e_cross[,2] > 0), size=3000)
  cand_eF_orig <- list("cand_F_cross"=cand_eF_orig$cand_F_cross[nonneg_ind, ],
                       "cand_e_cross"=cand_eF_orig$cand_e_cross[nonneg_ind, ])
  # only the cross case matters, as `create_candidate_eF` returns NULL for the above and below case after then kappa_n adjustment
  test3_orig <- CS_eF_cross(target_alg_preds=matrix(alg_orig, nrow=1),
                            X=X, G=G, Y=Y,
                            est_nuisance=est_nuisance,
                            F1=c(F45, cand_eF_orig$cand_F_cross[,1]),
                            F2=c(F45, cand_eF_orig$cand_F_cross[,2]),
                            e1=matrix(c(e_orig[1], cand_eF_orig$cand_e_cross[,1]), nrow=1),
                            e2=matrix(c(e_orig[2], cand_eF_orig$cand_e_cross[,2]), nrow=1),
                            num_bstp_rep=1000,
                            dist_func="euclidean-sq",
                            optimizer="BFGS",
                            kink=TRUE)

  # CS_n^{\rho(e^*, F)}
  CS_orig <- cbind(rbind(c(F45, F45), cand_eF_orig$cand_F_cross)[which(test3_orig$rej==0), , drop=FALSE],
                   rbind(e_orig, cand_eF_orig$cand_e_cross)[which(test3_orig$rej==0), , drop=FALSE])
  # retrieving distances
  CS_dist_orig <- c(min(rho(CS_orig[,1], CS_orig[,2], CS_orig[,3], CS_orig[,4])),
                    max(rho(CS_orig[,1], CS_orig[,2], CS_orig[,3], CS_orig[,4])))


  ## the experimental algorithm that predicts total cost ---
  # create parameter space of candidate (e, F)
  cand_eF_tc <- create_candidate_eF(X=X, G=G, Y=Y,
                                    est_nuisance=est_nuisance,
                                    est_F45_coord=F45,
                                    num_grid=10000,
                                    alg_preds=alg_tc,
                                    optimizer="grid",
                                    buffer=4*kappa_n)
  # only take non-negative candidate (e, F)
  nonneg_ind <- sample(which(cand_eF_tc$cand_F_cross[,1] > 0 & cand_eF_tc$cand_e_cross[,1] > 0 & cand_eF_tc$cand_e_cross[,2] > 0), size=3000)
  cand_eF_tc <- list("cand_F_cross"=cand_eF_tc$cand_F_cross[nonneg_ind, ],
                     "cand_e_cross"=cand_eF_tc$cand_e_cross[nonneg_ind, ])
  # only the cross case matters, as `create_candidate_eF` returns NULL for the above and below case after then kappa_n adjustment
  test3_tc <- CS_eF_cross(target_alg_preds=matrix(alg_tc, nrow=1),
                          X=X, G=G, Y=Y,
                          est_nuisance=est_nuisance,
                          F1=c(F45, cand_eF_tc$cand_F_cross[,1]),
                          F2=c(F45, cand_eF_tc$cand_F_cross[,2]),
                          e1=matrix(c(e_tc[1], cand_eF_tc$cand_e_cross[,1]), nrow=1),
                          e2=matrix(c(e_tc[2], cand_eF_tc$cand_e_cross[,2]), nrow=1),
                          num_bstp_rep=1000,
                          dist_func="euclidean-sq",
                          optimizer="BFGS",
                          kink=TRUE)

  # CS_n^{\rho(e^*, F)}
  CS_tc <- cbind(rbind(c(F45, F45), cand_eF_tc$cand_F_cross)[which(test3_tc$rej==0), , drop=FALSE],
                 rbind(e_tc, cand_eF_tc$cand_e_cross)[which(test3_tc$rej==0), , drop=FALSE])
  # retrieving distances
  CS_dist_tc <- c(min(rho(CS_tc[,1], CS_tc[,2], CS_tc[,3], CS_tc[,4])),
                  max(rho(CS_tc[,1], CS_tc[,2], CS_tc[,3], CS_tc[,4])))


  ## the experimental algorithm that predicts avoidable cost ---
  # create parameter space of candidate (e, F)
  cand_eF_ac <- create_candidate_eF(X=X, G=G, Y=Y,
                                    est_nuisance=est_nuisance,
                                    est_F45_coord=F45,
                                    num_grid=10000,
                                    alg_preds=alg_ac,
                                    optimizer="grid",
                                    buffer=4*kappa_n)
  # only take non-negative candidate (e, F)
  nonneg_ind <- sample(which(cand_eF_ac$cand_F_cross[,1] > 0 & cand_eF_ac$cand_e_cross[,1] > 0 & cand_eF_ac$cand_e_cross[,2] > 0), size=3000)
  cand_eF_ac <- list("cand_F_cross"=cand_eF_ac$cand_F_cross[nonneg_ind, ],
                     "cand_e_cross"=cand_eF_ac$cand_e_cross[nonneg_ind, ])

  # only the cross case matters, as `create_candidate_eF` returns NULL for the above and below case after then kappa_n adjustment
  test3_ac <- CS_eF_cross(target_alg_preds=matrix(alg_ac, nrow=1),
                          X=X, G=G, Y=Y,
                          est_nuisance=est_nuisance,
                          F1=c(F45, cand_eF_ac$cand_F_cross[,1]),
                          F2=c(F45, cand_eF_ac$cand_F_cross[,2]),
                          e1=matrix(c(e_ac[1], cand_eF_ac$cand_e_cross[,1]), nrow=1),
                          e2=matrix(c(e_ac[2], cand_eF_ac$cand_e_cross[,2]), nrow=1),
                          num_bstp_rep=1000,
                          dist_func="euclidean-sq",
                          optimizer="BFGS",
                          kink=TRUE)

  # CS_n^{\rho(e^*, F)}
  CS_ac <- cbind(rbind(c(F45, F45), cand_eF_ac$cand_F_cross)[which(test3_ac$rej==0), , drop=FALSE],
                 rbind(e_ac, cand_eF_ac$cand_e_cross)[which(test3_ac$rej==0), , drop=FALSE])
  # retrieving distances
  CS_dist_ac <- c(min(rho(CS_ac[,1], CS_ac[,2], CS_ac[,3], CS_ac[,4])),
                  max(rho(CS_ac[,1], CS_ac[,2], CS_ac[,3], CS_ac[,4])))

  ## the experimental algorithm that predicts active chronic conditions ---
  # create parameter space of candidate (e, F)
  cand_eF_acc <- create_candidate_eF(X=X, G=G, Y=Y,
                                     est_nuisance=est_nuisance,
                                     est_F45_coord=F45,
                                     num_grid=10000,
                                     alg_preds=alg_acc,
                                     optimizer="grid",
                                     buffer=4*kappa_n)
  # only take non-negative candidate (e, F)
  nonneg_ind <- sample(which(cand_eF_acc$cand_F_cross[,1] > 0 & cand_eF_acc$cand_e_cross[,1] > 0 & cand_eF_acc$cand_e_cross[,2] > 0), size=3000)
  cand_eF_acc <- list("cand_F_cross"=cand_eF_acc$cand_F_cross[nonneg_ind, ],
                      "cand_e_cross"=cand_eF_acc$cand_e_cross[nonneg_ind, ])
  # only the cross case matters, as `create_candidate_eF` returns NULL for the above and below case after then kappa_n adjustment
  test3_acc <- CS_eF_cross(target_alg_preds=matrix(alg_acc, nrow=1),
                           X=X, G=G, Y=Y,
                           est_nuisance=est_nuisance,
                           F1=c(F45, cand_eF_acc$cand_F_cross[,1]),
                           F2=c(F45, cand_eF_acc$cand_F_cross[,2]),
                           e1=matrix(c(e_acc[1], cand_eF_acc$cand_e_cross[,1]), nrow=1),
                           e2=matrix(c(e_acc[2], cand_eF_acc$cand_e_cross[,2]), nrow=1),
                           num_bstp_rep=1000,
                           dist_func="euclidean-sq",
                           optimizer="BFGS",
                           kink=TRUE)

  # CS_n^{\rho(e^*, F)}
  CS_acc <- cbind(rbind(c(F45, F45), cand_eF_acc$cand_F_cross)[which(test3_acc$rej==0), , drop=FALSE],
                  rbind(e_acc, cand_eF_acc$cand_e_cross)[which(test3_acc$rej==0), , drop=FALSE])
  # retrieving distances
  CS_dist_acc <- c(min(rho(CS_acc[,1], CS_acc[,2], CS_acc[,3], CS_acc[,4])),
                   max(rho(CS_acc[,1], CS_acc[,2], CS_acc[,3], CS_acc[,4])))

  dist_orig <- append(dist_orig, rho(F45, F45, e_orig[1], e_orig[2]))
  CSl_orig <- append(CSl_orig, CS_dist_orig[1])
  CSu_orig <- append(CSu_orig, CS_dist_orig[2])
  dist_tc <- append(dist_tc, rho(F45, F45, e_tc[1], e_tc[2]))
  CSl_tc <- append(CSl_tc, CS_dist_tc[1])
  CSu_tc <- append(CSu_tc, CS_dist_tc[2])
  dist_ac <- append(dist_ac, rho(F45, F45, e_ac[1], e_ac[2]))
  CSl_ac <- append(CSl_ac, CS_dist_ac[1])
  CSu_ac <- append(CSu_ac, CS_dist_ac[2])
  dist_acc <- append(dist_acc, rho(F45, F45, e_acc[1], e_acc[2]))
  CSl_acc <- append(CSl_acc, CS_dist_acc[1])
  CSu_acc <- append(CSu_acc, CS_dist_acc[2])

  write.csv(data.frame(cbind(estR1, estR2, estB1, estB2, estF,
                             estR1_ints, estR2_ints, estB1_ints, estB2_ints, estF_ints,
                             test1_rej,
                             ts_orig, cv_orig, rej_orig,
                             ts_tc, cv_tc, rej_tc,
                             ts_ac, cv_ac, rej_ac,
                             ts_acc, cv_acc, rej_acc,
                             dist_orig, dist_tc, dist_ac, dist_acc,
                             CSl_orig, CSl_tc, CSl_ac, CSl_acc,
                             CSu_orig, CSu_tc, CSu_ac, CSu_acc)),
            file="../../results/rep-results/rep-rf.csv", row.names = F)

  end <- Sys.time()
  print(end-start)
}
