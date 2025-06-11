library(tidyr)
library(dplyr)
library(EnvStats)
library(glmnet)
library(grf)
library(nnet)
library(torch)
torch_manual_seed(0)
# for parallelization
spec_cpu <- as.integer(Sys.getenv("OMP_NUM_THREADS"))
torch_set_num_threads(spec_cpu)
torch_set_num_interop_threads(ceiling(log(spec_cpu)))
library(doMC)
registerDoMC(cores=spec_cpu)

## Function that generates data for the simulation
testDGP <- function(n){
  X1 <- EnvStats::rnormTrunc(n, min=-3, max=3)
  X2 <- runif(n)
  X3 <- rbeta(n, 2, 2)
  X4 <- EnvStats::rnormTrunc(n, min=-3, max=3)

  X <- cbind(X1,X2,X3,X4)
  for (numvar in 5:20){
    assign(paste0("X", numvar),
           EnvStats::rnormTrunc(n, min=-3, max=3))
    X <- cbind(X, get(paste0("X", numvar)))
  }

  G <- rbinom(n,1,p=0.6)

  Y_balance <- G*sapply(1/(1+exp(-(X1+X2+0.5*X3))),
                        function(p){rbinom(1,1,p)})+
    (1-G)*sapply(1/(1+exp(-(-X1-0.5*X2+X4))),
                 function(p){rbinom(1,1,p)})

  Y_rskew <- G*sapply(1/(1+exp(-2*(X1+X2+X3))),
                      function(p){rbinom(1,1,p)})+
    (1-G)*sapply(1/(1+exp(-0.7*(X1+0.5*X2+0.6*X4))),
                 function(p){rbinom(1,1,p)})

  return(list("Y_balance"=Y_balance, "Y_rskew"=Y_rskew, "X"=X, "G"=G))
}

## Nuisance Parameter Estimation: Fits a vector of nuisance parameters needed to construct the estimator for the support function
nuisance <- function(Y, X, G,
                     l="classification",
                     method="lasso-logit",
                     fold=5,
                     seed=NULL,
                     sample_index=NULL, # used for bootstrapped samples
                     num.trees=10000, # for the grf package
                     parallel=TRUE # for multithreading in glmnet
){

  if (is.null(sample_index)){
    Y <- data.matrix(Y)
    X <- data.matrix(X)
    G <- data.matrix(G)
  } else {
    Y <- data.matrix(Y[sample_index])
    X <- data.matrix(X[sample_index,])
    G <- data.matrix(G[sample_index])
  }

  n <- nrow(Y) # sample size

  # validate input data
  if (anyNA(X) | anyNA(Y) | anyNA(G)){
    stop(paste0("Input data (X, Y, G) has at least one missing value."))
  }
  if (nrow(X) < ncol(X)){
    stop(paste0("Input data has more covariates than observations."))
  }
  if (!all(G %in% c(1,0))){
    stop(paste0("Group G shuold contain only binary values in c(1,0)."))
  }

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      l <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      l <- function(d,y) return((d-y)^2)
    }
    else{
      stop(paste0("Argument l is not in c('classification', 'square-loss') or an object of class 'function'."))
    }
  } else{
    if (!inherits(l,"function")){
      stop(paste0("Argument l is not in c('classification', 'square-loss') or an object of class 'function'."))
    }
    if (length(formals(l))!=2){
      stop(cat(paste0("The number of arguments in the loss function should be 2. For example, for classification loss the function l should be defined as: \n l <- function(d,y) return(as.numeric(d != y))\n Check your definition of l.")))
    }
  }

  # randomly split the data into folds
  split_data_ind <- sample(1:fold, size=n,
                           replace=TRUE,
                           prob=rep(1/fold, length.out=fold))

  # estimate mu_1
  est_mu_1 <- mean(G)

  # collect vectors of predictions evaluated at folds not used for estimation
  pred_diff_theta_r <- c()
  pred_diff_theta_b <- c()

  # first, for each fold, we fit the nuisance parameters using the other folds and evaluate the nuisance estimators at the current fold
  for (fd in 1:fold){
    Y_fd <- Y[split_data_ind!=fd]
    X_fd <- X[split_data_ind!=fd,]
    G_fd <- G[split_data_ind!=fd]

    for (g in c(1,0)){
      # create a data frame for regression
      label <- (l(1,Y_fd)-l(0,Y_fd))*as.numeric(G_fd==g) # the effective training label
      datause_dg <- data.frame(X_fd)
      datause_dg$label <- label

      group_name <- ifelse(g==1, "r", "b")

      # set up the machine learner
      if (is.character(method)){
        if (method=="grf"){
          if (is.null(seed)){
            seed <- 1
            warning(paste0("Argument `seed` is not provided for `regression_forest`; a random seed=", seed, " has been set for reproducibility on the same platform (grf version >= 2.4.0 required)."))
          }
          assign(paste0("est_diff_theta_", group_name),
                 regression_forest(X_fd, label,
                                   num.trees=num.trees,
                                   tune.parameters=c("sample.fraction",
                                                     "mtry",
                                                     "min.node.size",
                                                     "honesty.fraction",
                                                     "honesty.prune.leaves",
                                                     "alpha",
                                                     "imbalance.penalty"),
                                   seed=seed))
        } else if (method=="nnet"){
          assign(paste0("est_diff_theta_", group_name),
                 nnet(label ~.,
                      data=datause_dg,
                      size=8,
                      maxit=1000,
                      decay=0.01,
                      MaxNWts=10000,
                      trace=FALSE))
        } else if (method=="lasso-logit"){
          assign(paste0("est_diff_theta_", group_name),
                 cv.glmnet(x=X_fd,
                           y=factor(label, levels = c("-1", "0", "1")),
                           family="multinomial", alpha=1,
                           intercept=TRUE, parallel=parallel))
        } else if (method=="lasso"){
          assign(paste0("est_diff_theta_", group_name),
                 cv.glmnet(x=X_fd,
                           y=label,
                           family="gaussian", alpha=1,
                           intercept=TRUE, parallel=parallel))
        }
        else{
          stop(paste0("Argument 'method' should be a string in c('grf', 'nnet', 'lasso', 'lasso-logit')."))
        }
      }
      else{
        stop(paste0("Argument 'method' should be a string in c('grf', 'nnet', 'lasso', 'lasso-logit')."))
      }
    }

    # evaluate estimated theta at the current fold
    X_cur <- data.matrix(X[split_data_ind==fd,])

    if (method=="lasso") {
      pred_diff_theta_r <- append(pred_diff_theta_r,
                                  predict(est_diff_theta_r, newx=X_cur, s="lambda.min"))
      pred_diff_theta_b <- append(pred_diff_theta_b,
                                  predict(est_diff_theta_b, newx=X_cur, s="lambda.min"))

    } else if (method=="lasso-logit"){
      pred_probs_r <- predict(est_diff_theta_r, newx=X_cur,
                              s="lambda.min", type="response")[ , , 1]
      pred_diff_theta_r <- append(pred_diff_theta_r,
                                  -1*pred_probs_r[, 1]+1*pred_probs_r[, 3])

      pred_probs_b <- predict(est_diff_theta_b, newx=X_cur,
                              s="lambda.min", type="response")[ , , 1]
      pred_diff_theta_b <- append(pred_diff_theta_b,
                                  -1*pred_probs_b[, 1]+1*pred_probs_b[, 3])
    } else{
      pred_diff_theta_r <- append(pred_diff_theta_r,
                                  as.numeric(unlist(predict(est_diff_theta_r, X_cur))))
      pred_diff_theta_b <- append(pred_diff_theta_b,
                                  as.numeric(unlist(predict(est_diff_theta_b, X_cur))))
    }
  }

  return(list("pred_diff_theta_r"=pred_diff_theta_r,
              "pred_diff_theta_b"=pred_diff_theta_b,
              "est_mu_1"=est_mu_1,
              "split_data_ind"=split_data_ind))
}


## Function that estimates the support function of the feasible set given the direction q (if another set of directions v is provided, then this function returns the estimated support function of the SUPPORT SET of the feasible set)
support_function <- function(Y, X, G,
                             q=NULL, v=NULL,
                             est_nuisance=NULL, # pass the list returned by the `nuisance` function; if NULL, `support_function` will call `nuisance` to do the nuisance estimation.
                             l="classification",
                             method="lasso-logit",
                             fold=5,
                             sample_index=NULL, # used for bootstrap samples
                             return_score=FALSE, # if TRUE, will return a vector of score functions; used in the multiplier bootstrap inside of the hypothesis test functions
                             compute_cov=FALSE # if TRUE, will compute and return covariance matrix of dimension nrow(q) by nrow(q)
){

  # validate direction input q
  if (!inherits(q, "matrix") | (inherits(q, "matrix") & ncol(q) != 2)){
    stop(paste0("Direction input q must be of class matrix with ncol=2."))
  } else{
    num_direction <- nrow(q)
  }
  if (is.null(v)){
    v <- q
  } else if (!inherits(v, "matrix") | (inherits(v, "matrix") & ncol(v) != 2) |  (inherits(v, "matrix") & nrow(v) != nrow(q))){
    stop(paste0("Direction input v must be of class matrix with ncol=2 and same number of rows as the direction input q."))
  }

  # first we fit the nuisance parameters
  if (is.null(est_nuisance)){
    pred_theta <- nuisance(Y=Y, X=X, G=G, l=l,
                           method=method, fold=fold,
                           sample_index=sample_index)
  } else{
    pred_theta <- est_nuisance
  }

  diff_theta_r <- pred_theta[[1]]
  diff_theta_b <- pred_theta[[2]]
  est_mu_1 <- pred_theta[[3]]
  split_data_ind <- pred_theta[[4]]

  if (is.null(sample_index)){
    Y <- data.matrix(Y)
    G <- data.matrix(G)
  } else {
    Y <- data.matrix(Y[sample_index])
    G <- data.matrix(G[sample_index])
  }

  Y_ordered <- Y[order(split_data_ind)]
  G_ordered <- G[order(split_data_ind)]

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      loss <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      loss <- function(d,y) return((d-y)^2)
    }
  } else{ loss <- l }

  # next we evaluate the support function expression in the paper
  sf <- matrix(data=NA, ncol=1, nrow=num_direction) # initialize final output vector

  # labels L
  L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
  L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
  L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
  L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)
  n <- length(L_00) # sample size

  diff_L_r <- L_11-L_01
  diff_L_b <- L_10-L_00

  score <- matrix(NA, nrow=num_direction, ncol=n) # initialize vector to store the scores; not used if return_score=FALSE
  covmat <- matrix(NA, nrow=num_direction, ncol=num_direction) # initialize vector to store the covariance matrix; not used if compute_cov=FALSE

  for (i in 1:num_direction){
    # direction (q, v) in the current iteration
    q_cur <- q[i, ]
    v_cur <- v[i, ]

    inner_prod_q <- q_cur[1]*diff_theta_r/est_mu_1 + q_cur[2]*diff_theta_b/(1-est_mu_1) # k(\theta, Mq)
    inner_prod_v <- v_cur[1]*diff_L_r/est_mu_1 + v_cur[2]*diff_L_b/(1-est_mu_1) # Mv'(L1-L0)
    sign_inner_prod <- as.numeric(inner_prod_q > 0) # 1(k(\theta, Mq) > 0)

    # orthogonal score
    cur_score <- v_cur[1]*L_01/est_mu_1+v_cur[2]*L_00/(1-est_mu_1)+inner_prod_v*sign_inner_prod

    if (return_score | compute_cov){ score[i,] <- cur_score }

    # then the simulated support function in the current direction q_cur is the sample average of score over X
    sf[i, ] <- mean(cur_score)
  }

  if (compute_cov){
    demeaned_moments <- apply(score, 2, function(x){x-sf}) # num_direction by n
    if (num_direction==1){
      demeaned_moments <- matrix(demeaned_moments, nrow=1)
    }
    covmat <- (demeaned_moments %*% t(demeaned_moments))/n # num_direction by num_direction
  }

  return(list("est_sf"=sf,
              "score"=score,
              "split_data_ind"=split_data_ind,
              "covmat"=covmat))
}



## Function that constructs confidence set for (R, B)
CS_RB <- function(Y, X, G,
                  R1, R2, B1, B2, # candidate (R, B) for the test inversion; R1, R2, B1, B2 should be vectors of the same length.
                  est_nuisance=NULL, # pass the list returned by the `nuisance` function; if NULL, `nuisance` will be called to do the nuisance estimation.
                  num_bstp_rep=NULL, # number of bootstrap replications; if NULL, then just return the test statistic without bootstrapped critical values
                  optimizer=c("grid", "SGD", "BFGS", "optimize"),
                  grid_range=pi/4, # range of directions to check for the moment inequalities
                  grid_size=100, # size of the grid used to discretize the range of directions; used when optimizer="grid"
                  control=list(maxit=10000), gradient=FALSE, # for optimizer="BFGS"
                  lr=0.1, maxit=100, # for optimizer="SGD"
                  init_par=NULL, # for optimizer=c("SGD", "BFGS"); if NULL, will call `CS_RB_init_par` to find initial values via grid search; if specified, need to be a matrix of dimension nrow=2 and ncol=length(R1), where the first (second) row corresponds to the optimization problem for R (B) and each column corresponds to a candidate parameter value.
                  kink=FALSE, # whether the feasible set has kinks; if TRUE, then the numerical approximation from Fang & Santos (2019) is used for the directional derivative Default is FALSE.
                  alpha=0.05, # significance level; not used if num_bstp_rep=NULL
                  infntsml_adj=1e-6, # infinitesimal adjustment factor; not used if num_bstp_rep=NULL
                  l="classification",
                  method="lasso-logit",
                  fold=5){

  # estimate nuisance parameters
  if (is.null(est_nuisance)){
    est_nuisance <- nuisance(Y=Y, X=X, G=G, l=l,
                             method=method, fold=fold)
  }

  diff_theta_r <- est_nuisance[[1]]
  diff_theta_b <- est_nuisance[[2]]
  est_mu_1 <- est_nuisance[[3]]
  split_data_ind <- est_nuisance[[4]]

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      loss <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      loss <- function(d,y) return((d-y)^2)
    }
  } else{ loss <- l }

  Y_ordered <- data.matrix(Y)[order(split_data_ind)]
  G_ordered <- data.matrix(G)[order(split_data_ind)]

  # labels L
  L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
  L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
  L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
  L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)

  diff_L_r <- L_11-L_01
  diff_L_b <- L_10-L_00

  if (optimizer=="grid"){
    # create grid for the set of directions for the moment inequalities
    range_R <- seq(pi-grid_range, pi+grid_range, length.out=grid_size)
    grid_R_q <- t(as.matrix(sapply(range_R, function(rad){c(cos(rad), sin(rad))})))

    range_B <- seq(3*pi/2-grid_range, 3*pi/2+grid_range, length.out=grid_size)
    grid_B_q <- t(as.matrix(sapply(range_B, function(rad){c(cos(rad), sin(rad))})))

    # check whether u1=(-1, 0) is included in grid_R_q
    if (c(pi) %in% range_R){ # if so, replace it with something else
      noise <- runif(1, -1, 1)*1e-4
      grid_R_q[which(range_R %in% c(pi)), ] <- c(cos(pi+noise), sin(pi+noise))
    }

    # check whether u2=(0, -1) is included in grid_B_q
    if (c(3*pi/2) %in% range_B){ # if so, replace it with something else
      noise <- runif(1, -1, 1)*1e-4
      grid_B_q[which(range_B %in% c(3*pi/2)), ] <- c(cos(3*pi/2+noise), sin(3*pi/2+noise))
    }

    # the final set of directions contains u1, u2, and the grids for R and B.
    grid_q <- rbind(c(-1, 0), c(0, -1), grid_R_q, grid_B_q) # nrow=2*grid_size+2

    # estimate the support functions for the directions in grid_q
    est <- support_function(X=X, G=G, Y=Y,
                            q=grid_q, est_nuisance=est_nuisance)

    # estimated support functions
    est_sf <- est$est_sf
    est_sf_u1 <- est_sf[1] # estimated support function at u1=[-1 0]'
    est_sf_u2 <- est_sf[2] # estimated support function at u2=[0 -1]'
    est_sf_grid_R <- est_sf[3:(grid_size+2)] # estimated support function at q \in grid_R_q
    est_sf_grid_B <- est_sf[(grid_size+3):(2*grid_size+2)] # estimated support function at q \in grid_B_q
  } else { ### optimizer=c("SGD", "L-BFGS-B")
    # estimated support functions in directions u1 and u2
    est_sf <- apply(matrix(c(-1, 0, 0, -1), ncol=2, byrow=TRUE), 1,
                    function(q) {mean(q[1]*L_01/est_mu_1+q[2]*L_00/(1-est_mu_1)+
                                        (q[1]*diff_L_r/est_mu_1+q[2]*diff_L_b/(1-est_mu_1))*
                                        as.numeric(q[1]*diff_theta_r/est_mu_1+q[2]*diff_theta_b/(1-est_mu_1)>0))})

    est_sf_u1 <- est_sf[1] # estimated support function at u1=[-1 0]'
    est_sf_u2 <- est_sf[2] # estimated support function at u2=[0 -1]'

    # if init_par is not specified and optimizers other than optimize() are used
    if (is.null(init_par) & optimizer!="optimize"){
      set_init_par <- CS_RB_init_par(X=X, G=G, Y=Y,
                                     est_nuisance=est_nuisance,
                                     R1=R1, R2=R2, B1=B1, B2=B2,
                                     grid_size=grid_size,
                                     grid_range=grid_range)
      init_par_R <- set_init_par$arg_max_rad_R
      init_par_B <- set_init_par$arg_max_rad_B
    } else if (!is.null(init_par)){
      init_par_R <- init_par[1, ]
      init_par_B <- init_par[2, ]
    }
  }

  n <- length(Y)
  num_param <- length(R1) # number of candidate parameters to check
  test_stat <- c() # initialize vector to collect test stats
  cv <- c() # initialize vector to collect bootstrapped crit val

  if (kink){ # if kink=TRUE, need to store inequality constraints for bootstrap
    all_ineq_constr_R <- c()
    all_ineq_constr_B <- c()
  }

  # optimal q associated with the maximization problem
  all_opt_q_R <- c()
  all_opt_q_B <- c()

  for (param in 1:num_param){
    # current parameter value being tested
    cur_R1 <- R1[param]
    cur_R2 <- R2[param]
    cur_B1 <- B1[param]
    cur_B2 <- B2[param]

    # construct the test statistic
    cur_eq_constr_R <- -min((-1*cur_R1)-est_sf_u1, 0)   # -min{u1'\tilde{R}-h_E(u1), 0}
    cur_eq_constr_B <- -min((-1*cur_B2)-est_sf_u2, 0)   # -min{u2'\tilde{B}-h_E(u2), 0}

    if (optimizer=="grid"){
      # max{max_q (q'\tilde{R}-h_E(q)), 0}
      cur_ineq_constr_R <- max(max((grid_R_q[,1]*cur_R1+grid_R_q[,2]*cur_R2)-est_sf_grid_R), 0)
      # max{max_q (q'\tilde{B}-h_E(q)), 0}
      cur_ineq_constr_B <- max(max((grid_B_q[,1]*cur_B1+grid_B_q[,2]*cur_B2)-est_sf_grid_B), 0)
    } else{ ### optimizer=c("SGD", "L-BFGS-B")
      f_R <- function(rad){ # objective function
        q1 <- cos(rad)
        q2 <- sin(rad)
        # h_E(q) - q'\tilde{R} >= 0
        mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_R1-q2*cur_R2
      }

      f_B <- function(rad){ # objective function
        q1 <- cos(rad)
        q2 <- sin(rad)
        # h_E(q) - q'\tilde{B} >= 0
        mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_B1-q2*cur_B2
      }

      if (optimizer=="BFGS"){
        if (gradient){ ## if gradient=TRUE, closed-form gradient formula is used
          grad_f_R <- function(rad){ # gradient
            q1 <- cos(rad)
            q2 <- sin(rad)

            g_q1 <- -sin(rad)
            g_q2 <- cos(rad)

            mean(g_q1*L_01/est_mu_1+g_q2*L_00/(1-est_mu_1)+
                   (g_q1*diff_L_r/est_mu_1+g_q2*diff_L_b/(1-est_mu_1))*
                   as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-g_q1*cur_R1-g_q2*cur_R2
          }

          grad_f_B <- function(rad){ # gradient
            q1 <- cos(rad)
            q2 <- sin(rad)

            g_q1 <- -sin(rad)
            g_q2 <- cos(rad)

            mean(g_q1*L_01/est_mu_1+g_q2*L_00/(1-est_mu_1)+
                   (g_q1*diff_L_r/est_mu_1+g_q2*diff_L_b/(1-est_mu_1))*
                   as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-g_q1*cur_B1-g_q2*cur_B2
          }
        } else { ## otherwise let optim() numerically approximate the gradient
          grad_f_R <- NULL
          grad_f_B <- NULL
        }

        # min_q h_E(q)-q'\tilde{R}
        result_cur_R <- optim(par=init_par_R[param],
                              fn=f_R, gr=grad_f_R,
                              method="L-BFGS-B",
                              lower=pi-grid_range, upper=pi+grid_range,
                              control=control)

        result_cur_B <- optim(par=init_par_B[param],
                              fn=f_B, gr=grad_f_B,
                              method="L-BFGS-B",
                              lower=3*pi/2-grid_range, upper=3*pi/2+grid_range,
                              control=control)

        # max_q (q'\tilde{R} - h_E(q)) = - min_q (h_E(q)-q'\tilde{R)
        cur_ineq_constr_R <- max(-result_cur_R$value, 0)
        cur_ineq_constr_B <- max(-result_cur_B$value, 0)
        all_opt_q_R <- append(all_opt_q_R, result_cur_R$par)
        all_opt_q_B <- append(all_opt_q_B, result_cur_B$par)
      } else if (optimizer=="SGD") { ### optimizer="SGD"
        # initialize argument
        rad_R <- torch_tensor(init_par_R[param], requires_grad=TRUE)
        rad_B <- torch_tensor(init_par_B[param], requires_grad=TRUE)
        # initialize optimization
        optim_R <- optim_adam(params=list(rad_R), lr=lr)
        optim_B <- optim_adam(params=list(rad_B), lr=lr)
        # optimize
        for (iter in 1:maxit) {
          # min_q h_E(q)-q'\tilde{R}
          optim_R$zero_grad()  # initialize gradient
          l_fn_R <- f_R(rad_R) # loss at current rad
          l_fn_R$backward()    # backpropagation
          optim_R$step()       # perform descent

          # min_q h_E(q)-q'\tilde{B}
          optim_B$zero_grad()  # initialize gradient
          l_fn_B <- f_B(rad_B) # loss at current rad
          l_fn_B$backward()    # backpropagation
          optim_B$step()       # perform descent
        }

        # max_q (q'\tilde{R} - h_E(q)) = - min_q (h_E(q)-q'\tilde{R})
        cur_ineq_constr_R <- max(-l_fn_R$item(), 0)
        cur_ineq_constr_B <- max(-l_fn_B$item(), 0)

        all_opt_q_R <- append(all_opt_q_R, rad_R$item())
        all_opt_q_B <- append(all_opt_q_B, rad_B$item())
      } else if (optimizer=="optimize"){
        result_cur_R <- optimize(f_R, c(0, 2*pi), tol = 0.0001)
        result_cur_B <- optimize(f_B, c(0, 2*pi), tol = 0.0001)

        # max_q (q'\tilde{R} - h_E(q)) = - min_q (h_E(q)-q'\tilde{R)
        cur_ineq_constr_R <- max(-result_cur_R$objective, 0)
        cur_ineq_constr_B <- max(-result_cur_B$objective, 0)
        all_opt_q_R <- append(all_opt_q_R, result_cur_R$minimum)
        all_opt_q_B <- append(all_opt_q_B, result_cur_B$minimum)
      }
    }

    if (kink){ ## if kink=TRUE, need to store these optimized values for bootstrap
      all_ineq_constr_R <- append(all_ineq_constr_R, cur_ineq_constr_R)
      all_ineq_constr_B <- append(all_ineq_constr_B, cur_ineq_constr_B)
    }

    # test statistic
    cur_test_stat <- sqrt(n)*(cur_eq_constr_R+cur_eq_constr_B+cur_ineq_constr_R+cur_ineq_constr_B)
    test_stat <- append(test_stat, cur_test_stat) # length=num_param
  }

  if (!is.null(num_bstp_rep)){
    # initialize matrix to store bootstrapped test statistic
    if (kink){
      BS_test_stat <- matrix(NA, nrow=num_param, ncol=num_bstp_rep) # depends on which parameter we test
    } else { ## no kink
      BS_test_stat <- matrix(NA, nrow=1, ncol=num_bstp_rep) # doesn't depend on which parameter we test
    }

    # bootstrap once for all param
    ### ------- BOOTSTRAP STARTS -------
    for (b in 1:num_bstp_rep){
      # draw n exponential(1) weights
      b_W <- rexp(n, rate=1)

      # bootstrapped mu_1
      b_mu_1 <- mean(b_W*G_ordered)/mean(b_W)

      if (optimizer=="grid"){
        if (kink){
          # bootstrap the support function
          b_est_sf <- apply(grid_q, 1,
                            function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                     (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                     as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

          b_est_sf_u1 <- b_est_sf[1]
          b_est_sf_u2 <- b_est_sf[2]
          b_est_sf_grid_R <- b_est_sf[3:(grid_size+2)]
          b_est_sf_grid_B <- b_est_sf[(grid_size+3):(2*grid_size+2)]

          # equality constraint
          b_eq_constr <- -min(-sqrt(n)*(b_est_sf_u1-est_sf_u1), 0)-min(-sqrt(n)*(b_est_sf_u2-est_sf_u2), 0)

          # approximate the directional derivative as per Fang & Santos (2017)
          s_n <- sqrt(n)^{-1/2+0.01} # step size
          # direction at which we evaluate the directional derivative of \phi
          b_direc_grid_R <- s_n*sqrt(n)*(b_est_sf_grid_R-est_sf_grid_R) # length=grid_size
          b_direc_grid_B <- s_n*sqrt(n)*(b_est_sf_grid_B-est_sf_grid_B) # length=grid_size
          # evaluation point for \phi
          b_eval_grid_R <- est_sf_grid_R+b_direc_grid_R
          b_eval_grid_B <- est_sf_grid_B+b_direc_grid_B

          # \phi depends on the parameter at which we test
          for (param in 1:num_param){
            # current parameter value being tested
            cur_R1 <- R1[param]
            cur_R2 <- R2[param]
            cur_B1 <- B1[param]
            cur_B2 <- B2[param]

            # inequality constraint associated with the current parameter
            cur_ineq_constr_R <- all_ineq_constr_R[param]
            cur_ineq_constr_B <- all_ineq_constr_B[param]

            # apply the \phi transformation to the evaluation points
            b_cur_ineq_constr_R <- max(max((grid_R_q[,1]*cur_R1+grid_R_q[,2]*cur_R2)-b_eval_grid_R), 0)
            b_cur_ineq_constr_B <- max(max((grid_B_q[,1]*cur_B1+grid_B_q[,2]*cur_B2)-b_eval_grid_B), 0)
            BS_test_stat[param, b] <- b_eq_constr+
              (1/s_n)*(b_cur_ineq_constr_R-cur_ineq_constr_R)+
              (1/s_n)*(b_cur_ineq_constr_B-cur_ineq_constr_B)
          }
        } else { # no kink
          # bootstrap the support function; only the first two directions (u1 and u2) matter
          b_est_sf <- apply(grid_q[1:2,], 1,
                            function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                     (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                     as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

          b_est_sf_u1 <- b_est_sf[1]
          b_est_sf_u2 <- b_est_sf[2]

          # equality constraint
          b_eq_constr <- -min(-sqrt(n)*(b_est_sf_u1-est_sf_u1), 0)-min(-sqrt(n)*(b_est_sf_u2-est_sf_u2), 0)
          # inequality constraint
          b_ineq_constr <- max(-sqrt(n)*(b_est_sf_u1-est_sf_u1), 0)+max(-sqrt(n)*(b_est_sf_u2-est_sf_u2), 0)

          BS_test_stat[1, b] <- b_eq_constr+b_ineq_constr
        }
      } else{ # optimizer=c("SGD", "BFGS")
        # bootstrapped support functions in directions u1 and u2
        b_est_sf <- apply(matrix(c(-1, 0, 0, -1), ncol=2, byrow=TRUE), 1,
                          function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                   (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                   as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

        b_est_sf_u1 <- b_est_sf[1] # bootstrapped support function at u1=[-1 0]'
        b_est_sf_u2 <- b_est_sf[2] # bootstrapped support function at u2=[0 -1]'

        # equality constraint
        b_eq_constr <- -min(-sqrt(n)*(b_est_sf_u1-est_sf_u1), 0)-min(-sqrt(n)*(b_est_sf_u2-est_sf_u2), 0)

        if (kink){
          # approximate the directional derivative as per Fang & Santos (2017)
          s_n <- sqrt(n)^{-1/2+0.01} # step size
          for (param in 1:num_param){
            cur_R1 <- R1[param]
            cur_R2 <- R2[param]
            cur_B1 <- B1[param]
            cur_B2 <- B2[param]

            # inequality constraint associated with the current parameter
            cur_ineq_constr_R <- all_ineq_constr_R[param]
            cur_ineq_constr_B <- all_ineq_constr_B[param]

            b_cur_f_R <- function(rad) { # objective function
              q1 <- cos(rad)
              q2 <- sin(rad)
              # estimated support function
              est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
              # b-th bootstrapped support function
              b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                  (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                  as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
              # direction at which we evaluate \phi
              b_cur_direc <- s_n*sqrt(n)*(b_sf-est_sf)
              # evaluation point
              est_sf+b_cur_direc-q1*cur_R1-q2*cur_R2
            }

            b_cur_f_B <- function(rad) { # objective function
              q1 <- cos(rad)
              q2 <- sin(rad)
              # estimated support function
              est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
              # b-th bootstrapped support function
              b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                  (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                  as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
              # direction at which we evaluate \phi
              b_cur_direc <- s_n*sqrt(n)*(b_sf-est_sf)
              # evaluation point
              est_sf+b_cur_direc-q1*cur_B1-q2*cur_B2
            }

            if (optimizer=="BFGS"){
              if (gradient){ ## if gradient=TRUE, closed-form gradient formula is used
                b_grad_f_R <- function(rad){ # gradient
                  q1 <- cos(rad)
                  q2 <- sin(rad)

                  g_q1 <- -sin(rad)
                  g_q2 <- cos(rad)

                  # gradient of estimated support function
                  g_est_sf <- mean(g_q1*L_01/est_mu_1+g_q2*L_00/(1-est_mu_1)+
                                     (g_q1*diff_L_r/est_mu_1+g_q2*diff_L_b/(1-est_mu_1))*
                                     as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
                  # gradient of bootstrapped support function
                  g_b_sf <- mean(b_W*(g_q1*L_01/b_mu_1+g_q2*L_00/(1-b_mu_1)+
                                        (g_q1*diff_L_r/b_mu_1+g_q2*diff_L_b/(1-b_mu_1))*
                                        as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

                  # direction at which we evaluate \phi
                  g_b_cur_direc <- s_n*sqrt(n)*(g_b_sf-g_est_sf)
                  # evaluation point
                  g_est_sf+g_b_cur_direc-g_q1*cur_R1-g_q2*cur_R2
                }

                b_grad_f_B <- function(rad){ # gradient
                  q1 <- cos(rad)
                  q2 <- sin(rad)

                  g_q1 <- -sin(rad)
                  g_q2 <- cos(rad)

                  # gradient of estimated support function
                  g_est_sf <- mean(g_q1*L_01/est_mu_1+g_q2*L_00/(1-est_mu_1)+
                                     (g_q1*diff_L_r/est_mu_1+g_q2*diff_L_b/(1-est_mu_1))*
                                     as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
                  # gradient of bootstrapped support function
                  g_b_sf <- mean(b_W*(g_q1*L_01/b_mu_1+g_q2*L_00/(1-b_mu_1)+
                                        (g_q1*diff_L_r/b_mu_1+g_q2*diff_L_b/(1-b_mu_1))*
                                        as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

                  # direction at which we evaluate \phi
                  g_b_cur_direc <- s_n*sqrt(n)*(g_b_sf-g_est_sf)
                  # evaluation point
                  g_est_sf+g_b_cur_direc-g_q1*cur_B1-g_q2*cur_B2
                }
              } else { ## otherwise let optim() numerically approximate the gradient
                b_grad_f_R <- NULL
                b_grad_f_B <- NULL
              }

              # min_q (h_E(q)+s_n*sqrt(n)*{h*_E(q)-h_E(q)})-q'\tilde{R}
              b_result_cur_R <- optim(par=all_opt_q_R[param],
                                      fn=b_cur_f_R, gr=b_grad_f_R,
                                      method="L-BFGS-B",
                                      lower=pi-grid_range, upper=pi+grid_range,
                                      control=control)

              # min_q (h_E(q)+s_n*sqrt(n)*{h*_E(q)-h_E(q)})-q'\tilde{B}
              b_result_cur_B <- optim(par=all_opt_q_B[param],
                                      fn=b_cur_f_B, gr=b_grad_f_B,
                                      method="L-BFGS-B",
                                      lower=3*pi/2-grid_range, upper=3*pi/2+grid_range,
                                      control=control)

              b_ineq_constr_R <- max(-b_result_cur_R$value, 0)
              b_ineq_constr_B <- max(-b_result_cur_B$value, 0)
            } else if (optimizer=="SGD"){ ### optimizer="SGD"
              # initialize argument
              rad_R <- torch_tensor(all_opt_q_R[param,1], requires_grad = TRUE)
              rad_B <- torch_tensor(all_opt_q_B[param,1], requires_grad = TRUE)
              # initialize optimization
              optim_R <- optim_adam(params=list(rad_R), lr=lr)
              optim_B <- optim_adam(params=list(rad_B), lr=lr)
              # optimize
              for (iter in 1:maxit) {
                # min_q h_E(q)-q'\tilde{R}
                optim_R$zero_grad()          # initialize gradient
                b_l_fn_R <- b_cur_f_R(rad_R) # loss at current rad
                b_l_fn_R$backward()          # backpropagation
                optim_R$step()               # perform descent

                # min_q h_E(q)-q'\tilde{B}
                optim_B$zero_grad()          # initialize gradient
                b_l_fn_B <- b_cur_f_B(rad_B) # loss at current rad
                b_l_fn_B$backward()          # backpropagation
                optim_B$step()               # perform descent
              }

              # max_q (q'\tilde{R} - h_E(q)) = - min_q (h_E(q)-q'\tilde{R})
              b_ineq_constr_R <- max(-b_l_fn_R$item(), 0)
              b_ineq_constr_B <- max(-b_l_fn_B$item(), 0)
            } else if (optimizer=="optimize"){
              b_result_cur_R <- optimize(b_cur_f_R, c(0, 2*pi), tol = 0.0001)
              b_result_cur_B <- optimize(b_cur_f_B, c(0, 2*pi), tol = 0.0001)

              b_ineq_constr_R <- max(-b_result_cur_R$objective, 0)
              b_ineq_constr_B <- max(-b_result_cur_B$objective, 0)
            }

            b_cur_ineq_constr <- (1/s_n)*(b_ineq_constr_R-cur_ineq_constr_R)+
              (1/s_n)*(b_ineq_constr_B-cur_ineq_constr_B)

            BS_test_stat[param, b] <- b_eq_constr+b_cur_ineq_constr
          }
        } else{ ## no kink
          # inequality constraint
          b_ineq_constr <- max(-sqrt(n)*(b_est_sf_u1-est_sf_u1), 0) + max(-sqrt(n)*(b_est_sf_u2-est_sf_u2), 0)
          BS_test_stat[1, b] <- b_eq_constr+b_ineq_constr
        }
      }
    }
    ### ------- BOOTSTRAP ENDS -------
    cv <- apply(BS_test_stat, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj})
  }

  return(list("test_stat"=test_stat,
              "BScv"=cv, # will return cv=c() if num_bstp_rep=NULL
              "rej"=as.numeric(test_stat>cv) # will return rej=numeric(0) if num_bstp_rep=NULL
  ))
}

## Function that finds initial parameter values to pass to the optimizers in `CS_RB` by grid search
CS_RB_init_par <- function(X, G, Y,
                           est_nuisance=NULL,
                           R1, R2, B1, B2,
                           grid_size=1000,
                           grid_range=pi/4,
                           l="classification",
                           method="lasso-logit",
                           fold=5){
  num_param <- length(R1)

  # create grid for the set of directions for the moment inequalities
  range_R <- seq(pi-grid_range, pi+grid_range, length.out=grid_size)
  grid_R_q <- t(as.matrix(sapply(range_R, function(rad){c(cos(rad), sin(rad))})))

  range_B <- seq(3*pi/2-grid_range, 3*pi/2+grid_range, length.out=grid_size)
  grid_B_q <- t(as.matrix(sapply(range_B, function(rad){c(cos(rad), sin(rad))})))

  # the final set of directions contains u1, u2, and the grids for R and B.
  grid_q <- rbind(c(-1, 0), c(0, -1), grid_R_q, grid_B_q)

  # estimate the support functions for the directions in grid_q
  est <- support_function(X=X, G=G, Y=Y,
                          q=grid_q, est_nuisance=est_nuisance,
                          l=l, method=method, fold=fold)

  # estimated support functions
  est_sf <- est$est_sf
  est_sf_u1 <- est_sf[1] # estimated support function at u1=[-1 0]'
  est_sf_u2 <- est_sf[2] # estimated support function at u2=[0 -1]'
  est_sf_grid_R <- est_sf[3:(grid_size+2)] # estimated support function at q \in grid_R_q
  est_sf_grid_B <- est_sf[(grid_size+3):(2*grid_size+2)] # estimated support function at q \in grid_B_q

  arg_max_rad_R <- matrix(NA, nrow=1, ncol=num_param)
  arg_max_rad_B <- matrix(NA, nrow=1, ncol=num_param)

  for (param in 1:num_param){
    # current parameter value being tested
    cur_R1 <- R1[param]
    cur_R2 <- R2[param]
    cur_B1 <- B1[param]
    cur_B2 <- B2[param]

    arg_max_rad_R[1, param] <- range_R[which.max((grid_R_q[,1]*cur_R1+grid_R_q[,2]*cur_R2)-est_sf_grid_R)]
    arg_max_rad_B[1, param] <- range_B[which.max((grid_B_q[,1]*cur_B1+grid_B_q[,2]*cur_B2)-est_sf_grid_B)]
  }

  return(list("arg_max_rad_R"=arg_max_rad_R,
              "arg_max_rad_B"=arg_max_rad_B))
}


## Function that constructs the LDA test statistics and corresponding critical values
test_LDA <- function(Y, X, G,
                     est_nuisance=NULL, # pass the list returned by the `nuisance` function; if NULL, `nuisance` will be called to do the nuisance estimation.
                     target_alg_preds=NULL, # a matrix with n columns collecting the target algorithm's predicted value at each X, a^*(X). The rows stack different target algorithms. If target_alg_preds != NULL, then the `test_LDA` computes the estimated error allocation for each target algorithm, and the argument target_e is ignored.
                     target_e=NULL, # a matrix with ncol=2; used when the error allocation is taken as non-random, i.e, NOT estimated from a given target algorithm; this argument is ignored when target_alg_preds is specified
                     num_bstp_rep=NULL, # number of bootstrap replications; if NULL, then just return the test statistic without bootstrapped critical values
                     optimizer=c("grid", "SGD", "BFGS", "optimize"),
                     grid_size=1000, # size of the grid used to discretize the support of the function being optimized; used when optimizer="grid".
                     control=list(maxit=10000), num_try_init_par=NULL, gradient=FALSE, # for optimizer="BFGS"
                     lr=0.01, maxit=20000, # for optimizer="SGD"
                     init_par=NULL, # for optimizer=c("SGD", "BFGS"); if NULL, will call `CS_FAfrontier_init_par` to find initial values via grid search; if specified, need to be a matrix of dimension nrow=2 and ncol=nrow(target_alg_preds) or ncol=nrow(target_e), where the first (second) row corresponds to the optimization problem for testing feasibility (LDA) and each column corresponds to a target algorithm.
                     kink=FALSE, # whether the feasible set has kinks; if TRUE, then the numerical approximation from Fang & Santos (2019) is used for the directional derivative Default is FALSE.
                     alpha=0.05, # significance level; not used if num_bstp_rep=NULL
                     infntsml_adj=1e-6, # infinitesimal adjustment factor; not used if num_bstp_rep=NULL
                     l="classification",
                     method="lasso-logit",
                     fold=5){

  Y <- data.matrix(Y)
  X <- data.matrix(X)
  G <- data.matrix(G)
  n <- length(Y)   # sample size

  # estimate nuisance parameters
  if (is.null(est_nuisance)){
    est_nuisance <- nuisance(Y=Y, X=X, G=G, l=l,
                             method=method, fold=fold)
  }

  diff_theta_r <- est_nuisance[[1]]
  diff_theta_b <- est_nuisance[[2]]
  est_mu_1 <- est_nuisance[[3]]
  split_data_ind <- est_nuisance[[4]]

  Y_ordered <- data.matrix(Y)[order(split_data_ind)]
  G_ordered <- data.matrix(G)[order(split_data_ind)]

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      loss <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      loss <- function(d,y) return((d-y)^2)
    }
  } else {
    loss <- l
  }

  # labels L
  L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
  L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
  L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
  L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)

  diff_L_r <- L_11-L_01
  diff_L_b <- L_10-L_00

  # validate input target_alg_preds
  if (!is.null(target_alg_preds)){
    if (!inherits(target_alg_preds, "matrix") | (inherits(target_alg_preds, "matrix") & ncol(target_alg_preds) != n)){
      stop(paste0("Target algorithm target_alg_preds must be of class 'matrix' that contains n:=length(Y) columns."))
    } else {
      num_alg <- nrow(target_alg_preds) # number of algorithms

      # first we need the scores that yield estimated e_g^*
      score_e_r <- matrix(NA, nrow=num_alg, ncol=n)
      score_e_b <- matrix(NA, nrow=num_alg, ncol=n)

      for (alg in 1:num_alg){ # for each algorithm
        cur_alg_preds <- target_alg_preds[alg,] # retrieve current predictions a^*(X)
        cur_score <- cur_alg_preds*as.numeric(loss(1, Y))+(1-cur_alg_preds)*as.numeric(loss(0, Y))
        score_e_r[alg,] <- as.numeric(G==1)*cur_score/est_mu_1
        score_e_b[alg,] <- as.numeric(G==0)*cur_score/(1-est_mu_1)
      }
      # estimate e^*
      target_e_r <- apply(score_e_r, 1, FUN = mean)
      target_e_b <- apply(score_e_b, 1, FUN = mean)
      target_e <- data.matrix(cbind(target_e_r, target_e_b))
    }
  } else {
    if (!inherits(target_e, "matrix") | (inherits(target_e, "matrix") & ncol(target_e) != 2)){
      stop(paste0("Target error allocation target_e must be of class matrix with ncol=2."))
    } else{
      num_alg <- nrow(target_e)
    }
  }

  if (optimizer == "grid"){
    # the discretized set of directions to search over
    grid_q_LDA <- t(as.matrix(sapply(seq(3*pi/4, 7*pi/4, length.out=grid_size),
                                     function(rad){c(cos(rad), sin(rad))})))
    grid_q_feasible <- t(as.matrix(sapply(seq(0, 2*pi, length.out=grid_size),
                                          function(rad){c(cos(rad), sin(rad))})))

    ## estimate the support functions for the directions in grid_q
    # for the feasible set \mathcal{E} (with q)
    est_sf_E_val <- support_function(X=X, G=G, Y=Y,
                                     q=rbind(grid_q_LDA, grid_q_feasible),
                                     est_nuisance=est_nuisance)

    # estimated sf corresponding to the LDA directions
    est_sf_E_LDA <- est_sf_E_val$est_sf[1:grid_size]
    # estimated sf corresponding to the feasibility directions
    est_sf_E_feasible <- est_sf_E_val$est_sf[(grid_size+1):(2*grid_size)]

    # for the set of all FA-improvements C* (with -q)
    est_sf_Cstar <- matrix(NA, nrow=num_alg, ncol=grid_size)
    # test statistic corresponding to the constraint that e is feasible
    test_stat_feasible <- c()  # should have length=num_alg
    arg_max_q_feasible <- c()  # should have length=num_alg
    max_sf_diff <- c() # used to store max_q (q'e - h_\E(q)); length=num_alg
    for (cur in 1:num_alg){
      cur_e_r <- target_e[cur, 1]
      cur_e_b <- target_e[cur, 2]
      cur_e_diff <- cur_e_r-cur_e_b # (e_r^*-e_b^*)
      # support function of C* for the current e, h_{C^*}(-q); has length=grid_size
      est_sf_Cstar[cur, ] <- apply(-grid_q_LDA, 1,
                                   function(q){max(2*q[2]*cur_e_diff+(q[1]-q[2])*max(2*cur_e_diff,0), 0)+q[1]*cur_e_r+q[2]*cur_e_b-q[1]*max(2*cur_e_diff,0)})

      # check feasibility of cur_e
      cur_arg_max_q <- which.max((grid_q_feasible[,1]*cur_e_r+grid_q_feasible[,2]*cur_e_b)-est_sf_E_feasible)
      arg_max_q_feasible <- append(arg_max_q_feasible, cur_arg_max_q)
      max_sf_diff <- append(max_sf_diff, (grid_q_feasible[cur_arg_max_q,1]*cur_e_r+grid_q_feasible[cur_arg_max_q,2]*cur_e_b)-est_sf_E_feasible[cur_arg_max_q])
      test_stat_feasible <- append(test_stat_feasible,
                                   sqrt(n)*max(max_sf_diff[cur], 0))
    }

    # compute the LDA test statistic, one for each row of target_e
    # min_q (est_sf_Cstar(-q)+est_sf_E(q))
    min_sf_sum <- t(apply(est_sf_Cstar, 1, function(x){x+c(est_sf_E_LDA)})) %>% apply(1, FUN = min) # length=num_alg
    arg_min_q_LDA <- t(apply(est_sf_Cstar, 1, function(x){x+c(est_sf_E_LDA)})) %>% apply(1, FUN = which.min) # length=num_alg
    # max -(est_sf_Cstar+est_sf_E) = - min (est_sf_Cstar+est_sf_E), then apply -min{x, 0}
    test_stat_LDA <- sqrt(n)*(-pmin(-min_sf_sum, 0)) # length=num_param
    # final test statistic; length=num_param
    test_stat <- test_stat_feasible + test_stat_LDA
  } else { ### optimizer
    # set up the initial parameter value
    if (is.null(init_par) & optimizer!="optimize"){
      set_init_par <- CS_FAfrontier_init_par(X=X, G=G, Y=Y,
                                             est_nuisance=est_nuisance,
                                             e1=target_e[,1], e2=target_e[,2],
                                             grid_size=10000)
      init_par_feasible <- set_init_par$arg_max_rad_feasible
      init_par_LDA <- set_init_par$arg_min_rad_LDA
    } else if (!is.null(init_par)){
      init_par_feasible <- init_par[1,]
      init_par_LDA <- init_par[2,]
    }

    # initialize vectors to store minimized objective values, minimizers, and test stats
    min_sf_sum <- matrix(NA, nrow=1, ncol=num_alg)
    arg_min_rad_LDA <- matrix(NA, nrow=1, ncol=num_alg)
    test_stat_LDA <- matrix(NA, nrow=1, ncol=num_alg)

    max_sf_diff <- matrix(NA, nrow=1, ncol=num_alg)
    arg_max_rad_feasible <- matrix(NA, nrow=1, ncol=num_alg)
    test_stat_feasible <- matrix(NA, nrow=1, ncol=num_alg)

    for (alg in 1:num_alg){
      cur_e_r <- target_e[alg, 1]
      cur_e_b <- target_e[alg, 2]

      # test statistic (before taking the min/inf) as a function of q for the current algorithm
      alg_obj_LDA <- function(rad){
        q1 <- cos(rad)
        q2 <- sin(rad)

        # estimated support function of E at q
        est_sf_E <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+(q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))

        # e^*
        cur_e_diff <- cur_e_r-cur_e_b # (e_r^*-e_b^*)

        # support function of C* at -q
        neg_q1 <- -q1
        neg_q2 <- -q2
        est_sf_Cstar <- max(2*neg_q2*cur_e_diff+(neg_q1-neg_q2)*max(2*cur_e_diff,0), 0)+neg_q1*cur_e_r+neg_q2*cur_e_b-neg_q1*max(2*cur_e_diff,0) # h_{C^*}(-q)

        return(est_sf_Cstar+est_sf_E)
      }

      # feasibility test statistic
      alg_obj_feasible <- function(rad){
        q1 <- cos(rad)
        q2 <- sin(rad)

        # h_E(q) - q'e >= 0
        mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_e_r-q2*cur_e_b
      }

      if (optimizer=="BFGS"){### optimizer is BFGS
        # solving min_q (est_sf_Cstar+est_sf_E)
        result_LDA <- optim(par=init_par_LDA[alg],
                            fn=alg_obj_LDA,
                            method="L-BFGS-B",
                            lower=3*pi/4, upper=7*pi/4,
                            control=control)

        min_sf_sum[1, alg] <- result_LDA$value
        arg_min_rad_LDA[1, alg] <- result_LDA$par

        if (gradient){ ## if gradient=TRUE, closed-form gradient formula is used for the feasibility constraint
          g_alg_obj_feasible <- function(rad){
            q1 <- cos(rad)
            q2 <- sin(rad)

            g_q1 <- -sin(rad)
            g_q2 <- cos(rad)

            # h_E(q) - q'e >= 0
            mean(g_q1*L_01/est_mu_1+g_q2*L_00/(1-est_mu_1)+
                   (g_q1*diff_L_r/est_mu_1+g_q2*diff_L_b/(1-est_mu_1))*
                   as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-g_q1*cur_e_r-g_q2*cur_e_b
          }
        } else{
          g_alg_obj_feasible <- NULL
        }

        # feasibility: min_q (h_E(q)-q'e)
        result_feasible <- optim(par=init_par_feasible[alg],
                                 fn=alg_obj_feasible,
                                 gr=g_alg_obj_feasible,
                                 method="L-BFGS-B",
                                 lower=0, upper=2*pi,
                                 control=control)

        # max (q'e-h_E(q)) = - min_q (h_E(q)-q'e)
        max_sf_diff[1, alg] <- -result_feasible$value
        arg_max_rad_feasible[1, alg] <- result_feasible$par
      } else if (optimizer=="SGD"){ ### optimizer is SGD
        # solving min_q (est_sf_Cstar+est_sf_E)
        rad_LDA <- torch_tensor(init_par_LDA[alg], requires_grad=TRUE)
        optim <- optim_adam(params=list(rad_LDA), lr=lr)
        for (iter in 1:maxit) {
          optim$zero_grad()
          l_fn_LDA <- alg_obj_LDA(rad_LDA) # loss at current rad
          l_fn_LDA$backward()
          optim$step()
          # constrain the range of rad_LDA
          rad_LDA$data()$clamp_(3*pi/4, 7*pi/4)
        }

        min_sf_sum[1, alg] <- l_fn_LDA$item()
        arg_min_rad_LDA[1, alg] <- rad_LDA$item()

        # solving min_q (h_E(q)-q'e)
        rad_feasible <- torch_tensor(init_par_feasible[alg], requires_grad=TRUE)
        optim <- optim_adam(params=list(rad_feasible), lr=lr)
        for (iter in 1:maxit) {
          # min_q (h_E(q)-q'e)
          optim$zero_grad()
          l_fn_feasible <- alg_obj_feasible(rad_feasible) # loss at current rad
          l_fn_feasible$backward()
          optim$step()
        }

        # max (q'e-h_E(q)) = - min_q (h_E(q)-q'e)
        max_sf_diff[1, alg] <- -l_fn_feasible$item()
        arg_max_rad_feasible[1, alg] <- rad_feasible$item()
      } else if (optimizer=="optimize"){
        # solving min_q (est_sf_Cstar+est_sf_E)
        result_LDA <- optimize(alg_obj_LDA, c(3*pi/4, 7*pi/4), tol = 0.0001)
        min_sf_sum[1, alg] <- result_LDA$objective
        arg_min_rad_LDA[1, alg] <- result_LDA$minimum

        # feasibility: min_q (h_E(q)-q'e)
        result_feasible <- optimize(alg_obj_feasible, c(0, 2*pi), tol = 0.0001)
        # max (q'e-h_E(q)) = - min_q (h_E(q)-q'e)
        max_sf_diff[1, alg] <- -result_feasible$objective
        arg_max_rad_feasible[1, alg] <- result_feasible$minimum
      }

      # max -(est_sf_Cstar+est_sf_E) = - min (est_sf_Cstar+est_sf_E), then apply -min{x, 0}
      test_stat_LDA[alg] <- sqrt(n)*(-min(-min_sf_sum[alg], 0))
      test_stat_feasible[alg] <- sqrt(n)*(max(max_sf_diff[alg], 0))
    }

    test_stat <- test_stat_feasible + test_stat_LDA
  }

  # initialize vector to collect bootstrapped crit val; not used if num_bstp_rep=NULL
  btsrp_cv <- c()

  if (!is.null(num_bstp_rep)){
    # initialize matrix to store bootstrapped test statistic for each row of target_e
    BS_test_stat <- matrix(NA, nrow=num_alg, ncol=num_bstp_rep)

    ## step size for approximating the directional derivative of \phi, as per Step 2, Procedure 2 of Liu & Molinari (2025) if the feasible set has kinks or e^* is estimated
    s_n <- sqrt(n)^{-1/2+0.01} # step size

    if (!is.null(target_alg_preds)){
      score_e_r <- score_e_r[, order(split_data_ind), drop=FALSE]
      score_e_b <- score_e_b[, order(split_data_ind), drop=FALSE]
    }

    if (optimizer=="grid"){
      ## bootstrapped estimated support function of the feasible set, for each q in grid_q
      btsrp_sf_E_feasible <- matrix(NA, nrow=grid_size, ncol=num_bstp_rep)
      btsrp_sf_E_LDA <- matrix(NA, nrow=grid_size, ncol=num_bstp_rep)

      ## bootstrapped target_e (e^*), for each e^* in row of target_e
      btsrp_e_r <- matrix(NA, nrow=num_alg, ncol=num_bstp_rep)
      btsrp_e_b <- matrix(NA, nrow=num_alg, ncol=num_bstp_rep)

      # bootstrap once for all rows of target_e
      for (b in 1:num_bstp_rep){
        # draw n exponential(1) weights
        b_W <- rexp(n, rate=1)

        # bootstrapped mu_1
        b_mu_1 <- mean(b_W*G_ordered)/mean(b_W)

        # bootstrapped support function for the LDA directions
        btsrp_sf_E_LDA[, b] <- apply(grid_q_LDA, 1,
                                     function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                              (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                              as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

        # bootstrapped support function for the feasibility directions
        btsrp_sf_E_feasible[, b] <- apply(grid_q_feasible, 1,
                                          function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                                   (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                                   as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

        # bootstrap e^*
        if (!is.null(target_alg_preds)){
          btsrp_e_r[, b] <- apply(score_e_r, 1, function(x) mean(b_W*x*est_mu_1/b_mu_1)/mean(b_W))
          btsrp_e_b[, b] <- apply(score_e_b, 1, function(x) mean(b_W*x*(1-est_mu_1)/(1-b_mu_1))/mean(b_W))
        } else{ # if target_e is treated as not estimated, then no need to bootstrap e^*
          btsrp_e_r[, b] <- target_e[, 1]
          btsrp_e_b[, b] <- target_e[, 2]
        }
      }

      if (kink | !is.null(target_alg_preds)){ # if the feasible set has kinks or e is estimated
        # we need to use the Fang & Santos approximation for the LDA part
        for (b in 1:num_bstp_rep){
          # b-th bootstrapped sf of E; has length grid_size
          cur_btsrp_sf_E_LDA <- c(btsrp_sf_E_LDA[, b])

          # direction of estimated sf at which we evaluate the directional derivative of \phi
          cur_direc_sf_E_LDA <- s_n*sqrt(n)*(cur_btsrp_sf_E_LDA-c(est_sf_E_LDA)) # length=grid_size
          cur_sf_E_LDA <- c(est_sf_E_LDA)+cur_direc_sf_E_LDA # evaluation point

          for (alg in 1:num_alg){
            # direction of estimated e at which we evaluate the directional derivative of \phi
            b_e_r <- btsrp_e_r[alg, b] # b-th bootstrapped e_r
            b_e_b <- btsrp_e_b[alg, b] # b-th bootstrapped e_b
            cur_direc_e_r <- s_n*sqrt(n)*(b_e_r-target_e[alg, 1]) # length=1
            cur_direc_e_b <- s_n*sqrt(n)*(b_e_b-target_e[alg, 2]) # length=1

            # evaluate \phi
            cur_e_r <- target_e[alg, 1]+cur_direc_e_r
            cur_e_b <- target_e[alg, 2]+cur_direc_e_b
            cur_e_diff <- cur_e_r-cur_e_b
            cur_sf_Cstar <- apply(-grid_q_LDA, 1,
                                  function(q){max(2*q[2]*cur_e_diff+(q[1]-q[2])*max(2*cur_e_diff,0), 0)+q[1]*cur_e_r+q[2]*cur_e_b-q[1]*max(2*cur_e_diff,0)}) # h_{C^*}(-q); length=grid_size

            # bootstrapped LDA test stat
            b_cur_constr_LDA <- -min(cur_sf_Cstar+cur_sf_E_LDA)

            if (kink){ # if there is kink, need to numerically approximate the feasibility part too
              cur_btsrp_sf_E_feasible <- c(btsrp_sf_E_feasible[, b])
              cur_direc_sf_E_feasible <- s_n*sqrt(n)*(cur_btsrp_sf_E_feasible-c(est_sf_E_feasible)) # length=grid_size
              cur_sf_E_feasible <- c(est_sf_E_feasible)+cur_direc_sf_E_feasible # evaluation point
              b_cur_constr_feasible <- max((grid_q_feasible[,1]*cur_e_r+grid_q_feasible[,2]*cur_e_b)-cur_sf_E_feasible)
              BS_test_stat[alg, b] <- (1/s_n)*(-pmin(b_cur_constr_LDA-(-min_sf_sum[alg]), 0) + max(b_cur_constr_feasible-max_sf_diff[alg], 0))
            } else{ # if no kink but e is estimated
              opt_q_feasible_ind <- arg_max_q_feasible[alg]
              b_cur_constr_feasible <- sqrt(n)*(grid_q_feasible[opt_q_feasible_ind,1]*(b_e_r-target_e[alg, 1])+grid_q_feasible[opt_q_feasible_ind,2]*(b_e_b-target_e[alg, 2])-(btsrp_sf_E_feasible[opt_q_feasible_ind, b]-est_sf_E_feasible[opt_q_feasible_ind]))
              BS_test_stat[alg, b] <- (1/s_n)*(-pmin(b_cur_constr_LDA-(-min_sf_sum[alg]), 0))+max(b_cur_constr_feasible, 0)
            }
          }
        }
      } else { # if e is not estimated AND no kink
        for (alg in 1:num_alg){
          b_sf_opt_q_LDA <- btsrp_sf_E_LDA[arg_min_q_LDA[alg],]
          b_sf_opt_q_feasible <- btsrp_sf_E_feasible[arg_max_q_feasible[alg],]
          BS_test_stat[alg,] <- -pmin(-sqrt(n)*(b_sf_opt_q_LDA-est_sf_E_LDA[arg_min_q_LDA[alg]]), 0)+max(-sqrt(n)*(b_sf_opt_q_feasible-est_sf_E_feasible[arg_max_q_feasible[alg]]), 0)
        }
      }

      # compute the bootstrapped crit val, as per Step 3, Procedure 2
      btsrp_cv <- apply(BS_test_stat, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj})
    } else{ ## optimizer=c("SGD", "BFGS", "optimize")
      for (b in 1:num_bstp_rep){
        # draw n exponential(1) weights
        b_W <- rexp(n, rate=1)

        # bootstrapped mu_1
        b_mu_1 <- mean(b_W*G_ordered)/mean(b_W)

        if (kink | !is.null(target_alg_preds)){# if the feasible set has kinks or e is estimated
          # we need to use the Fang & Santos approximation for the LDA part
          for (alg in 1:num_alg){
            # min (\widehat{he^*}(\widehat{\btheta})+s_n*sqrt(n)*(\widetilde{he^*}(\widehat{\btheta})-\widehat{he^*}(\widehat{\btheta})))
            b_alg_obj_LDA <- function(rad){
              q1 <- cos(rad)
              q2 <- sin(rad)

              # estimated support function
              est_sf_E <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                                 (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                                 as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
              # b-th bootstrapped support function
              cur_btsrp_sf_E <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                            (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                            as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

              # bootstrap e^*
              if (!is.null(target_alg_preds)){
                cur_btsrp_e_r <- mean(b_W*score_e_r[alg, ]*est_mu_1/b_mu_1)/mean(b_W)
                cur_btsrp_e_b <- mean(b_W*score_e_b[alg, ]*(1-est_mu_1)/(1-b_mu_1))/mean(b_W)
              } else{# if target_e is treated as not estimated, then no need to bootstrap
                cur_btsrp_e_r <- target_e[alg, 1]
                cur_btsrp_e_b <- target_e[alg, 2]
              }

              # direction at which we evaluate the directional derivative of \phi
              cur_direc_sf_E <- s_n*sqrt(n)*(cur_btsrp_sf_E-est_sf_E)
              cur_direc_e_r <- s_n*sqrt(n)*(cur_btsrp_e_r-target_e[alg, 1])
              cur_direc_e_b <- s_n*sqrt(n)*(cur_btsrp_e_b-target_e[alg, 2])

              # evaluate \phi
              cur_sf_E <- est_sf_E+cur_direc_sf_E
              cur_e_r <- target_e[alg, 1]+cur_direc_e_r
              cur_e_b <- target_e[alg, 2]+cur_direc_e_b
              cur_e_diff <- cur_e_r-cur_e_b
              neg_q1 <- -q1
              neg_q2 <- -q2
              cur_sf_Cstar <- max(2*neg_q2*cur_e_diff+(neg_q1-neg_q2)*max(2*cur_e_diff,0), 0)+neg_q1*cur_e_r+neg_q2*cur_e_b-neg_q1*max(2*cur_e_diff,0) # h_{C^*}(-q)

              return(cur_sf_Cstar+cur_sf_E)
            }

            if (optimizer=="SGD"){
              # solving bootstrapped min (cur_sf_Cstar+cur_sf_E)
              b_rad_LDA <- torch_tensor(arg_min_rad_LDA[alg], requires_grad=TRUE)
              optim <- optim_adam(params=list(b_rad_LDA), lr=lr)
              for (iter in 1:maxit) {
                optim$zero_grad()
                b_l_fn_LDA <- b_alg_obj_LDA(b_rad_LDA) # loss at current rad
                b_l_fn_LDA$backward()
                optim$step()
                # constrain the range of b_rad_LDA
                b_rad_LDA$data()$clamp_(3*pi/4, 7*pi/4)
              }

              # bootstrapped LDA test stat
              b_cur_constr_LDA <- -b_l_fn_LDA$item()
            } else if (optimizer=="BFGS"){
              b_optim_LDA <- optim(par=arg_min_rad_LDA[alg],
                                   fn=b_alg_obj_LDA,
                                   method="L-BFGS-B",
                                   lower=3*pi/4, upper=7*pi/4,
                                   control=control)

              # multi-restart to avoid local minima
              if (!is.null(num_try_init_par)){
                for (num_try in 1:num_try_init_par) {
                  current_result <- optim(par=b_optim_LDA$par,
                                          fn=b_alg_obj_LDA,
                                          method="L-BFGS-B",
                                          lower=3*pi/4, upper=7*pi/4,
                                          control=control)

                  # if current is better, update best
                  if (current_result$value < b_optim_LDA$value) {
                    b_optim_LDA <- current_result
                  } else {
                    break
                  }
                }
              }

              # bootstrapped LDA test stat
              b_cur_constr_LDA <- -b_optim_LDA$value
            } else if (optimizer=="optimize"){
              b_optim_LDA <- optimize(b_alg_obj_LDA, c(3*pi/4, 7*pi/4), tol = 0.0001)
              # bootstrapped LDA test stat
              b_cur_constr_LDA <- -b_optim_LDA$objective
            }

            if (kink){ # if there is kink, need to numerically approximate the feasibility part too
              b_alg_obj_feasible <- function(rad) { # objective function
                q1 <- cos(rad)
                q2 <- sin(rad)

                # estimated support function
                est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                                 (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                                 as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
                # b-th bootstrapped support function
                b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                    (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                    as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

                # bootstrap e^*
                if (!is.null(target_alg_preds)){
                  b_e_r <- mean(b_W*score_e_r[alg, ]*est_mu_1/b_mu_1)/mean(b_W)
                  b_e_b <- mean(b_W*score_e_b[alg, ]*(1-est_mu_1)/(1-b_mu_1))/mean(b_W)
                } else{# if target_e is treated as not estimated, then no need to bootstrap
                  b_e_r <- target_e[alg, 1]
                  b_e_b <- target_e[alg, 2]
                }

                # direction at which we evaluate \phi
                b_direc_sf <- s_n*sqrt(n)*(b_sf-est_sf)
                b_direc_e_r <- s_n*sqrt(n)*(b_e_r-target_e[alg, 1])
                b_direc_e_b <- s_n*sqrt(n)*(b_e_b-target_e[alg, 2])

                # evaluation points
                eval_sf <- est_sf+b_direc_sf
                eval_e_r <- target_e[alg, 1]+b_direc_e_r
                eval_e_b <- target_e[alg, 2]+b_direc_e_b

                # evaluate \phi
                eval_sf-q1*eval_e_r-q2*eval_e_b
              }

              if (optimizer=="SGD"){
                # initialize argument
                b_rad_feasible <- torch_tensor(arg_max_rad_feasible[alg], requires_grad=TRUE)
                optim <- optim_adam(params=list(b_rad_feasible), lr=lr)
                # min_q (h_E(q)-q'e*)
                for (iter in 1:maxit) {
                  optim$zero_grad()             # initialize gradient
                  b_l_fn_feasible <- b_alg_obj_feasible(b_rad_feasible) # loss at current rad
                  b_l_fn_feasible$backward()    # backpropagation
                  optim$step()                  # descent
                }

                # max_q (q'e* - h_E(q)) = - min_q (h_E(q)-q'e*)
                b_cur_constr_feasible <- -b_l_fn_feasible$item()
              } else if (optimizer=="BFGS"){
                if (gradient){ ## if gradient=TRUE, closed-form gradient formula is used
                  b_grad_obj_feasible <-function(rad){ # gradient
                    q1 <- cos(rad)
                    q2 <- sin(rad)

                    g_q1 <- -sin(rad)
                    g_q2 <- cos(rad)

                    # estimated support function
                    g_est_sf <- mean(g_q1*L_01/est_mu_1+g_q2*L_00/(1-est_mu_1)+
                                       (g_q1*diff_L_r/est_mu_1+g_q2*diff_L_b/(1-est_mu_1))*
                                       as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
                    # b-th bootstrapped support function
                    g_b_sf <- mean(b_W*(g_q1*L_01/b_mu_1+g_q2*L_00/(1-b_mu_1)+
                                          (g_q1*diff_L_r/b_mu_1+g_q2*diff_L_b/(1-b_mu_1))*
                                          as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

                    # bootstrap e^*
                    if (!is.null(target_alg_preds)){
                      b_e_r <- mean(b_W*score_e_r[alg, ]*est_mu_1/b_mu_1)/mean(b_W)
                      b_e_b <- mean(b_W*score_e_b[alg, ]*(1-est_mu_1)/(1-b_mu_1))/mean(b_W)
                    } else{# if target_e is treated as not estimated, then no need to bootstrap
                      b_e_r <- target_e[alg, 1]
                      b_e_b <- target_e[alg, 2]
                    }

                    # direction at which we evaluate \phi
                    g_b_direc_sf <- s_n*sqrt(n)*(g_b_sf-g_est_sf)
                    b_direc_e_r <- s_n*sqrt(n)*(b_e_r-target_e[alg, 1])
                    b_direc_e_b <- s_n*sqrt(n)*(b_e_b-target_e[alg, 2])

                    # evaluation points
                    g_eval_sf <- g_est_sf+g_b_direc_sf
                    eval_e_r <- target_e[alg, 1]+b_direc_e_r
                    eval_e_b <- target_e[alg, 2]+b_direc_e_b

                    # evaluate \phi
                    g_eval_sf-g_q1*eval_e_r-g_q2*eval_e_b
                  }
                } else{
                  b_grad_obj_feasible <- NULL
                }

                b_optim_feasible <- optim(par=arg_max_rad_feasible[alg],
                                          fn=b_alg_obj_feasible,
                                          gr=b_grad_obj_feasible,
                                          method="L-BFGS-B",
                                          lower=0, upper=2*pi,
                                          control=control)

                # multi-restart to avoid local minima
                if (!is.null(num_try_init_par)){
                  for (num_try in 1:num_try_init_par) {
                    current_result <- optim(par=b_optim_feasible$par,
                                            fn=b_alg_obj_feasible,
                                            gr=b_grad_obj_feasible,
                                            method="L-BFGS-B",
                                            lower=0, upper=2*pi,
                                            control=control)

                    # if current is better, update best
                    if (current_result$value < b_optim_feasible$value) {
                      b_optim_feasible <- current_result
                    } else {
                      break
                    }
                  }
                }

                # max_q (q'e* - h_E(q)) = - min_q (h_E(q)-q'e*)
                b_cur_constr_feasible <- -b_optim_feasible$value
              } else if (optimizer=="optimize"){
                b_optim_feasible <- optimize(b_alg_obj_feasible, c(0, 2*pi), tol = 0.0001)
                # max_q (q'e* - h_E(q)) = - min_q (h_E(q)-q'e*)
                b_cur_constr_feasible <- -b_optim_feasible$objective
              }

              BS_test_stat[alg, b] <- (1/s_n)*(-pmin(b_cur_constr_LDA-(-min_sf_sum[alg]), 0) + max(b_cur_constr_feasible-max_sf_diff[alg], 0))
            } else{ # if no kink but e is estimated
              opt_q_feasible <- matrix(c(cos(arg_max_rad_feasible[alg]),
                                         sin(arg_max_rad_feasible[alg])), ncol=2)

              # estimated support function at optimal q
              est_sf_opt_feasible <- mean(opt_q_feasible[1]*L_01/est_mu_1+opt_q_feasible[2]*L_00/(1-est_mu_1)+(opt_q_feasible[1]*diff_L_r/est_mu_1+opt_q_feasible[2]*diff_L_b/(1-est_mu_1))*as.numeric(opt_q_feasible[1]*diff_theta_r/est_mu_1+opt_q_feasible[2]*diff_theta_b/(1-est_mu_1)>0))
              # b-th bootstrapped support function at optimal q
              b_sf_opt_feasible <- mean(b_W*(opt_q_feasible[1]*L_01/b_mu_1+opt_q_feasible[2]*L_00/(1-b_mu_1)+(opt_q_feasible[1]*diff_L_r/b_mu_1+opt_q_feasible[2]*diff_L_b/(1-b_mu_1))*as.numeric(opt_q_feasible[1]*diff_theta_r/b_mu_1+opt_q_feasible[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

              # b-th bootstrapped e^*
              b_e_r <- mean(b_W*score_e_r[alg, ]*est_mu_1/b_mu_1)/mean(b_W)
              b_e_b <- mean(b_W*score_e_b[alg, ]*(1-est_mu_1)/(1-b_mu_1))/mean(b_W)

              # b-th bootstrapped limit distribution for the feasibility part
              b_cur_constr_feasible <- sqrt(n)*(opt_q_feasible[1]*(b_e_r-target_e[alg, 1])+opt_q_feasible[2]*(b_e_b-target_e[alg, 2])-(b_sf_opt_feasible-est_sf_opt_feasible))

              BS_test_stat[alg, b] <- (1/s_n)*(-pmin(b_cur_constr_LDA-(-min_sf_sum[alg]), 0))+max(b_cur_constr_feasible, 0)
            }
          }
        } else { # if e is not estimated AND no kink
          for (alg in 1:num_alg){
            ## for the feasibility part ----
            opt_q_feasible <- matrix(c(cos(arg_max_rad_feasible[alg]),
                                       sin(arg_max_rad_feasible[alg])), ncol=2)

            # estimated support function at optimal q
            est_sf_opt_feasible <- mean(opt_q_feasible[1]*L_01/est_mu_1+opt_q_feasible[2]*L_00/(1-est_mu_1)+(opt_q_feasible[1]*diff_L_r/est_mu_1+opt_q_feasible[2]*diff_L_b/(1-est_mu_1))*as.numeric(opt_q_feasible[1]*diff_theta_r/est_mu_1+opt_q_feasible[2]*diff_theta_b/(1-est_mu_1)>0))

            # b-th bootstrapped support function at optimal q
            b_sf_opt_feasible <- mean(b_W*(opt_q_feasible[1]*L_01/b_mu_1+opt_q_feasible[2]*L_00/(1-b_mu_1)+(opt_q_feasible[1]*diff_L_r/b_mu_1+opt_q_feasible[2]*diff_L_b/(1-b_mu_1))*as.numeric(opt_q_feasible[1]*diff_theta_r/b_mu_1+opt_q_feasible[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

            # b-th bootstrapped limit distribution for the feasibility part
            b_cur_constr_feasible <- sqrt(n)*(-(b_sf_opt_feasible-est_sf_opt_feasible))

            ## for the LDA part ----
            opt_q_LDA <- matrix(c(cos(arg_min_rad_LDA[alg]),
                                  sin(arg_min_rad_LDA[alg])), ncol=2)

            # estimated support function at optimal q
            est_sf_opt_LDA <- mean(opt_q_LDA[1]*L_01/est_mu_1+opt_q_LDA[2]*L_00/(1-est_mu_1)+(opt_q_LDA[1]*diff_L_r/est_mu_1+opt_q_LDA[2]*diff_L_b/(1-est_mu_1))*as.numeric(opt_q_LDA[1]*diff_theta_r/est_mu_1+opt_q_LDA[2]*diff_theta_b/(1-est_mu_1)>0))

            # b-th bootstrapped support function at optimal q
            b_sf_opt_LDA <- mean(b_W*(opt_q_LDA[1]*L_01/b_mu_1+opt_q_LDA[2]*L_00/(1-b_mu_1)+(opt_q_LDA[1]*diff_L_r/b_mu_1+opt_q_LDA[2]*diff_L_b/(1-b_mu_1))*as.numeric(opt_q_LDA[1]*diff_theta_r/b_mu_1+opt_q_LDA[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

            # b-th bootstrapped limit distribution for the feasibility part
            b_cur_constr_LDA <- sqrt(n)*(-(b_sf_opt_LDA-est_sf_opt_LDA))

            BS_test_stat[alg, b] <- max(b_cur_constr_feasible, 0)-min(b_cur_constr_LDA, 0)
          }
        }
      }
      # compute the bootstrapped crit val, as per Step 3, Procedure 2
      btsrp_cv <- apply(BS_test_stat, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj})
    }
  }

  return(list("LDA_stat"=test_stat,
              "BScv"=btsrp_cv, # will return BScv=c() if num_bstp_rep=NULL
              "rej"=as.numeric(test_stat>btsrp_cv) # will return rej=numeric(0) if num_bstp_rep=NULL
  ))
}


## Function that constructs CS_n^+ and CS_n^- for the distance-to-F test
CS_eF_disjoint <- function(Y, X, G,
                           est_nuisance=NULL, # pass the list returned by the `nuisance` function; if NULL, `nuisance` will be called to do the nuisance estimation.
                           F1_above=NULL, F2_above=NULL, e1_above=NULL, e2_above=NULL, # candidate (e, F) to check for CS_n^+; if is.null(target_alg_preds)=FALSE and `e1_above` and `e2_above` are specified, they should be matrices with nrow=nrow(target_alg_preds) and ncol=length(F1_above)=length(F2_above). If `target_e` is specified, `e1_above` and `e2_above` are ignored.
                           F1_below=NULL, F2_below=NULL, e1_below=NULL, e2_below=NULL, # candidate (e, F) to check for CS_n^-; if is.null(target_alg_preds)=FALSE and`e1_below` and `e2_below` are specified, they should be matrices with nrow=nrow(target_alg_preds) and ncol=length(F1_below)=length(F2_below). If `target_e` is specified, `e1_below` and `e2_below` are ignored.
                           target_alg_preds=NULL, # a matrix with n columns collecting the target algorithm's predicted value at each X, a^*(X). The rows stack different target algorithms. If is.null(target_alg_preds)=FALSE, then `CS_eF` computes the estimated error allocation for each target algorithm, the argument `target_e` is ignored, and arguments `e1_above`, `e2_above`, `e1_below`, and `e2_below` must be all specified or all NULL.
                           target_e=NULL, # a matrix with ncol=2; used when the error allocation is taken as non-random, i.e, NOT estimated from a given target algorithm; this argument is ignored when target_alg_preds is specified. If `target_alg_preds` is NULL and `target_e` is specified, then arguments `e1_above`, `e2_above`, `e1_below`, and `e2_below` are ignored.
                           num_bstp_rep=NULL, # number of bootstrap replications; if NULL, then just return the test statistic without bootstrapped critical values
                           optimizer=c("grid", "SGD", "BFGS"),
                           grid_range=pi/4, # range of directions to check for the moment inequalities
                           grid_size=100, # size of the grid used to discretize the range of directions; used when optimizer="grid"
                           control=list(maxit=10000), # for optimizer="BFGS"
                           lr=0.1, maxit=100, # for optimizer="SGD"
                           init_par=NULL, # for optimizer=c("SGD", "BFGS"); if NULL, will call `CS_eF_disjoint_init_par` to find initial values via grid search; if specified, need to be a list of two vectors, the first having length=length(F1_above) and the second having length=length(F1_below).
                           kink=FALSE, # whether the feasible set has kinks; if TRUE, then the numerical approximation from Fang & Santos (2019) is used for the directional derivative Default is FALSE.
                           alpha=0.05, # significance level; not used if num_bstp_rep=NULL
                           infntsml_adj=1e-6, # infinitesimal adjustment factor; not used if num_bstp_rep=NULL
                           l="classification",
                           method="lasso-logit",
                           fold=5){

  # if the candidate values of (e, F) are all NULL, then return NULL values
  allNULL_above <- is.null(F1_above) & is.null(F2_above) & is.null(e1_above) & is.null(e2_above)
  allNULL_below <- is.null(F1_below) & is.null(F2_below) & is.null(e1_below) & is.null(e2_below)
  if (allNULL_above & allNULL_below){
    return(list("test_stat_above"=NULL,
                "test_stat_below"=NULL,
                "cv_above"=NULL,
                "cv_below"=NULL,
                "rej_above"=NULL,
                "rej_below"=NULL))
  }

  # estimate nuisance parameters
  if (is.null(est_nuisance)){
    est_nuisance <- nuisance(Y=Y, X=X, G=G, l=l,
                             method=method, fold=fold)
  }

  diff_theta_r <- est_nuisance[[1]]
  diff_theta_b <- est_nuisance[[2]]
  est_mu_1 <- est_nuisance[[3]]
  split_data_ind <- est_nuisance[[4]]

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      loss <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      loss <- function(d,y) return((d-y)^2)
    }
  } else{ loss <- l }

  Y_ordered <- data.matrix(Y)[order(split_data_ind)]
  G_ordered <- data.matrix(G)[order(split_data_ind)]

  # labels L
  L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
  L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
  L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
  L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)

  diff_L_r <- L_11-L_01
  diff_L_b <- L_10-L_00

  if (optimizer=="grid"){
    ## create grids for the set of directions indexing the moment inequalities
    # when the feasible set is ABOVE the 45-degree line
    range_above <- seq(7*pi/4-grid_range, 7*pi/4+grid_range, length.out=grid_size)
    grid_F_above <- t(as.matrix(sapply(range_above,
                                       function(rad){c(cos(rad), sin(rad))})))

    range_below <- seq(3*pi/4-grid_range, 3*pi/4+grid_range, length.out=grid_size)
    grid_F_below <- t(as.matrix(sapply(range_below,
                                       function(rad){c(cos(rad), sin(rad))}))) # when the feasible set is BELOW the 45-degree line

    # check whether (u2-u1)/sqrt(2) is included in grid_F_above
    if (c(7*pi/4) %in% range_above){ # if so, replace it with something else
      noise <- runif(1, -1, 1)*1e-4
      grid_F_above[which(range_above %in% c(7*pi/4)), ] <- c(cos(7*pi/4+noise), sin(7*pi/4+noise))
    }

    # check whether (u1-u2)/sqrt(2) is included in grid_F_below
    if (c(3*pi/4) %in% range_below){ # if so, replace it with something else
      noise <- runif(1, -1, 1)*1e-4
      grid_F_below[which(range_below %in% c(3*pi/4)), ] <- c(cos(3*pi/4+noise), sin(3*pi/4+noise))
    }

    # the final set of directions contains (u2-u1)/sqrt(2), (u1-u2)/sqrt(2), and the grids for F when the feasible is above/below the 45-degree line.
    grid_q <- rbind(c(cos(7*pi/4), sin(7*pi/4)), # (u2-u1)/sqrt(2)
                    c(cos(3*pi/4), sin(3*pi/4)), # (u1-u2)/sqrt(2)
                    grid_F_above,
                    grid_F_below)

    # estimate the support functions for the directions in grid_q
    est <- support_function(X=X, G=G, Y=Y,
                            q=grid_q, est_nuisance=est_nuisance)

    # estimated support functions
    est_sf <- est$est_sf
    est_sf_F_above <- est_sf[1] # estimated sf in direction (u2-u1)/sqrt(2)
    est_sf_F_below <- est_sf[2] # estimated sf in direction (u1-u2)/sqrt(2)
    est_sf_grid_above <- est_sf[3:(grid_size+2)] # estimated sf for q \in grid_F_above
    est_sf_grid_below <- est_sf[(grid_size+3):(2*grid_size+2)] # estimated sf for q \in grid_F_below
  } else{ ### optimizer=c("SGD", "BFGS")
    # estimated support functions in directions (u2-u1)/sqrt(2) and (u1-u2)/sqrt(2)
    est_sf <- apply(matrix(c(1/sqrt(2), -1/sqrt(2),
                             -1/sqrt(2), 1/sqrt(2)), ncol=2, byrow=TRUE), 1,
                    function(q) {mean(q[1]*L_01/est_mu_1+q[2]*L_00/(1-est_mu_1)+
                                        (q[1]*diff_L_r/est_mu_1+q[2]*diff_L_b/(1-est_mu_1))*
                                        as.numeric(q[1]*diff_theta_r/est_mu_1+q[2]*diff_theta_b/(1-est_mu_1)>0))})

    est_sf_F_above <- est_sf[1] # estimated sf in direction (u2-u1)/sqrt(2)
    est_sf_F_below <- est_sf[2] # estimated sf in direction (u1-u2)/sqrt(2)

    # set initial parameter values for the optimizers
    if (is.null(init_par)){
      set_init_par <- CS_eF_disjoint_init_par(X=X, G=G, Y=Y,
                                              est_nuisance=est_nuisance,
                                              F1_above=F1_above, F2_above=F2_above,
                                              F1_below=F1_below, F2_below=F2_below,
                                              grid_size=grid_size,
                                              grid_range=grid_range)
      init_par_above <- set_init_par$arg_max_rad_above
      init_par_below <- set_init_par$arg_max_rad_below
    } else{
      init_par_above <- init_par[[1]]
      init_par_below <- init_par[[2]]
    }
  }

  n <- length(Y)
  # validate input target_alg_preds
  if (!is.null(target_alg_preds)){
    if (!inherits(target_alg_preds, "matrix") | (inherits(target_alg_preds, "matrix") & ncol(target_alg_preds) != n)){
      stop(paste0("Target algorithm `target_alg_preds` must be of class 'matrix' that contains n:=length(Y) columns."))
    } else if (length(unique(c(length(F1_above), length(F2_above), ncol(e1_above), ncol(e2_above))))>1 | length(unique(c(length(F1_below), length(F2_below), ncol(e1_below), ncol(e2_below))))>1 | (!allNULL_above & length(unique(c(nrow(e1_above), nrow(e2_above), nrow(target_alg_preds))))!=1) | (!allNULL_below & length(unique(c(nrow(e1_below), nrow(e2_below), nrow(target_alg_preds))))!=1)) {
      stop(paste0("If specified, (1) `e1_above` and `e2_above` should have ncol=length(F1_above)=length(F2_above); (2)`e1_below` and `e2_below` should have ncol=length(F1_below)=length(F2_below); (3) when `target_alg_preds` is specified, `e1_above`, `e2_above`, `e1_below`, and `e2_below` should be matrices with nrow=nrow(target_alg_preds)."))
    } else {
      num_alg <- nrow(target_alg_preds) # number of algorithms

      # first we need the scores that yield estimated e_g^*
      score_e_r <- matrix(NA, nrow=num_alg, ncol=n)
      score_e_b <- matrix(NA, nrow=num_alg, ncol=n)

      for (alg in 1:num_alg){ # for each algorithm
        cur_alg_preds <- target_alg_preds[alg,] # retrieve current predictions a^*(X)
        cur_score <- cur_alg_preds*as.numeric(loss(1, Y))+(1-cur_alg_preds)*as.numeric(loss(0, Y))
        score_e_r[alg,] <- as.numeric(G==1)*cur_score/est_mu_1
        score_e_b[alg,] <- as.numeric(G==0)*cur_score/(1-est_mu_1)
      }

      # estimate e^*
      target_e_r <- apply(score_e_r, 1, FUN = mean)
      target_e_b <- apply(score_e_b, 1, FUN = mean)
      target_e <- data.matrix(cbind(target_e_r, target_e_b))

      # equality constraints for candidate e=e^*
      if (!is.null(e1_above)){
        all_eq_constr_e1_above <- matrix(apply(e1_above, 2, function(x) abs(x-target_e_r)), nrow=num_alg) # num_alg by num_param_above
        all_eq_constr_e2_above <- matrix(apply(e2_above, 2, function(x) abs(x-target_e_b)), nrow=num_alg) # num_alg by num_param_above
      }

      if (!is.null(e1_below)){
        all_eq_constr_e1_below <- matrix(apply(e1_below, 2, function(x) abs(x-target_e_r)), nrow=num_alg) # num_alg by num_param_below
        all_eq_constr_e2_below <- matrix(apply(e2_below, 2, function(x) abs(x-target_e_b)), nrow=num_alg) # num_alg by num_param_below
      }
    }
  } else {
    if (!inherits(target_e, "matrix") | (inherits(target_e, "matrix") & ncol(target_e) != 2)){
      stop(paste0("Target error allocation `target_e` must be of class matrix with ncol=2."))
    } else if (length(unique(c(length(F1_above), length(F2_above))))>1 | length(unique(c(length(F1_below), length(F2_below))))>1 ){
      stop(paste0("If specified, `F1_above` and `F2_above` (resp., `F1_below` and `F2_below`) should have the same length."))
    } else{ # if e is not estimated, then CS_n^+/CS_n^_ depends on F only
      num_alg <- 1
    }
  }

  test_stat_above <- c()
  cv_above <- c()
  test_stat_below <- c()
  cv_below <- c()

  ## Constructing CS_n^+
  if (!allNULL_above){
    num_param_above <- length(F1_above) # number of candidate parameters to check for CS_n^+

    # store constraints for F
    all_eq_constr_F_above <- c()
    all_ineq_constr_F_above <- c()

    ## Constructing test statistics for candidate parameter F values
    for (param in 1:num_param_above){
      # current candidate value of F being tested
      cur_F1 <- F1_above[param]
      cur_F2 <- F2_above[param]

      # equality constraints for F being the support set
      cur_eq_constr_F_above <- -min((1/sqrt(2)*cur_F1-1/sqrt(2)*cur_F2)-est_sf_F_above, 0) # -min{(u2-u1)/sqrt(2)'\tilde{F}-h_E((u2-u1)/sqrt(2)), 0}
      all_eq_constr_F_above <- append(all_eq_constr_F_above, cur_eq_constr_F_above)

      if (optimizer=="grid"){
        # inequality constraints for F being feasible
        cur_ineq_constr_F_above <- max(max((grid_F_above[,1]*cur_F1+grid_F_above[,2]*cur_F2)-est_sf_grid_above), 0) # max{max_q (q'\tilde{F}-h_E(q)), 0}
      } else { ### optimizer=c("SGD", "BFGS")
        if (optimizer=="BFGS"){
          f_above <- function(q){ # objective function
            q1 <- q[1]
            q2 <- q[2]
            # h_E(q) - q'\tilde{F} >= 0
            mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+(q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_F1-q2*cur_F2
          }

          # min_q h_E(q)-q'\tilde{F}
          result_cur_above <- optim(par=c(cos(init_par_above[param]), sin(init_par_above[param])),
                                    fn=f_above,
                                    method="L-BFGS-B",
                                    lower=c(cos(7*pi/4-grid_range), sin(7*pi/4-grid_range)),
                                    upper=c(cos(7*pi/4+grid_range), sin(7*pi/4+grid_range)),
                                    control=control)

          # max_q (q'\tilde{F}-h_E(q)) = - min_q (h_E(q)-q'\tilde{F})
          cur_ineq_constr_F_above <- max(-result_cur_above$value, 0)   # max{max_q (q'\tilde{F}-h_E(q)), 0}
        } else { ### optimizer="SGD"
          f_above <- function(rad){ # objective function
            q1 <- cos(rad)
            q2 <- sin(rad)
            # h_E(q) - q'\tilde{F} >= 0
            mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+(q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_F1-q2*cur_F2
          }

          # initialize argument
          rad_F_above <- torch_tensor(init_par_above[param], requires_grad=TRUE)
          # initialize optimization
          optim_F_above <- optim_adam(params=list(rad_F_above), lr=lr)
          # optimize
          for (iter in 1:maxit) {
            # min_q h_E(q)-q'\tilde{R}
            optim_F_above$zero_grad()      # initialize gradient
            l_fn_F <- f_above(rad_F_above) # loss at current rad
            l_fn_F$backward()              # backpropagation
            optim_F_above$step()           # descent
          }

          # max_q (q'\tilde{F} - h_E(q)) = - min_q (h_E(q)-q'\tilde{F})
          cur_ineq_constr_F_above <- max(-l_fn_F$item(), 0)  # max{max_q (q'\tilde{F}-h_E(q)), 0}
        }
      }
      all_ineq_constr_F_above <- append(all_ineq_constr_F_above, cur_ineq_constr_F_above)
    }

    if (!is.null(target_alg_preds)){ ## if e^* is estimated
      # num_alg by num_param_above
      test_stat_above <- sqrt(n)*sweep(all_eq_constr_e1_above+all_eq_constr_e2_above, 2,
                                       all_eq_constr_F_above+all_ineq_constr_F_above, "+")
    } else{ ## if e^* is not estimated
      # 1 by num_param_above
      test_stat_above <- matrix(sqrt(n)*(all_eq_constr_F_above+all_ineq_constr_F_above), nrow=1)
    }
  }

  ## Constructing CS_n^-
  if (!allNULL_below){
    num_param_below <- length(F1_below) # number of candidate parameters to check for CS_n^-

    # store constraints for F
    all_eq_constr_F_below <- c()
    all_ineq_constr_F_below <- c()

    ## Constructing test statistics for candidate parameter F values
    for (param in 1:num_param_below){
      # current candidate value of F being tested
      cur_F1 <- F1_below[param]
      cur_F2 <- F2_below[param]

      # equality constraints for F being the support set
      cur_eq_constr_F_below <- -min((-1/sqrt(2)*cur_F1+1/sqrt(2)*cur_F2)-est_sf_F_below, 0) # -min{(u1-u2)/sqrt(2)'\tilde{F}-h_E((u1-u2)/sqrt(2)), 0}
      all_eq_constr_F_below <- append(all_eq_constr_F_below, cur_eq_constr_F_below)

      if (optimizer=="grid"){
        # inequality constraints for F being feasible
        cur_ineq_constr_F_below <- max(max((grid_F_below[,1]*cur_F1+grid_F_below[,2]*cur_F2)-est_sf_grid_below), 0) # max{max_q (q'\tilde{F}-h_E(q)), 0}
      } else { ### optimizer=c("SGD", "BFGS")
        if (optimizer=="BFGS"){
          f_below <- function(q){ # objective function
            q1 <- q[1]
            q2 <- q[2]
            # h_E(q) - q'\tilde{F} >= 0
            mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+(q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_F1-q2*cur_F2
          }

          # min_q h_E(q)-q'\tilde{F}
          result_cur_below <- optim(par=c(cos(init_par_below[param]), sin(init_par_below[param])),
                                    fn=f_below,
                                    method="L-BFGS-B",
                                    lower=c(cos(3*pi/4+grid_range), sin(3*pi/4+grid_range)),
                                    upper=c(cos(3*pi/4-grid_range), sin(3*pi/4-grid_range)),
                                    control=control)

          # max_q (q'\tilde{F}-h_E(q)) = - min_q (h_E(q)-q'\tilde{F})
          cur_ineq_constr_F_below <- max(-result_cur_below$value, 0)   # max{max_q (q'\tilde{F}-h_E(q)), 0}
        } else { ### optimizer="SGD"
          f_below <- function(rad){ # objective function
            q1 <- cos(rad)
            q2 <- sin(rad)
            # h_E(q) - q'\tilde{F} >= 0
            mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+(q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_F1-q2*cur_F2
          }

          # initialize argument
          rad_F_below <- torch_tensor(init_par_below[param], requires_grad=TRUE)
          # initialize optimization
          optim_F_below <- optim_adam(params=list(rad_F_below), lr=lr)
          # optimize
          for (iter in 1:maxit) {
            # min_q h_E(q)-q'\tilde{R}
            optim_F_below$zero_grad()      # initialize gradient
            l_fn_F <- f_below(rad_F_below) # loss at current rad
            l_fn_F$backward()              # backpropagation
            optim_F_below$step()           # descent
          }

          # max_q (q'\tilde{F} - h_E(q)) = - min_q (h_E(q)-q'\tilde{F})
          cur_ineq_constr_F_below <- max(-l_fn_F$item(), 0)  # max{max_q (q'\tilde{F}-h_E(q)), 0}
        }
      }
      all_ineq_constr_F_below <- append(all_ineq_constr_F_below, cur_ineq_constr_F_below)
    }

    if (!is.null(target_alg_preds)){ ## if e^* is estimated
      # num_alg by num_param_below
      test_stat_below <- sqrt(n)*sweep(all_eq_constr_e1_below+all_eq_constr_e2_below, 2,
                                       all_eq_constr_F_below+all_ineq_constr_F_below, "+")
    } else{ ## if e^* is not estimated
      # 1 by num_param_below
      test_stat_below <- matrix(sqrt(n)*(all_eq_constr_F_below+all_ineq_constr_F_below), nrow=1)
    }
  }

  # bootstrap once for all param
  if (!is.null(num_bstp_rep)){
    # initialize matrix to store bootstrapped test statistic; we bootstrap separately e and F
    if (!is.null(target_alg_preds)){ # if e^* is estimated, we need to bootstrap for it;
      # initialize matrices to store bootstrapped moments for e^*
      BS_constr_e_r <- matrix(NA, nrow=num_alg, ncol=num_bstp_rep)
      BS_constr_e_b <- matrix(NA, nrow=num_alg, ncol=num_bstp_rep)
    }

    if (!allNULL_above){
      if (kink){
        BS_constr_F_above <- matrix(NA, nrow=num_param_above, ncol=num_bstp_rep) # depends on which parameter we test
      } else { ## no kink
        BS_constr_F_above <- matrix(NA, nrow=1, ncol=num_bstp_rep) # doesn't depend on which parameter we test
      }
    }

    if (!allNULL_below){
      if (kink){
        BS_constr_F_below <- matrix(NA, nrow=num_param_below, ncol=num_bstp_rep) # depends on which parameter we test
      } else { ## no kink
        BS_constr_F_below <- matrix(NA, nrow=1, ncol=num_bstp_rep) # doesn't depend on which parameter we test
      }
    }

    ### ------- BOOTSTRAP STARTS -------
    for (b in 1:num_bstp_rep){
      # draw n exponential(1) weights
      b_W <- rexp(n, rate=1)

      # bootstrapped mu_1
      b_mu_1 <- mean(b_W*G_ordered)/mean(b_W)

      if (!is.null(target_alg_preds)){ # need to bootstrap for e^* if estimated
        # reshuffle the obs indices by split_data_ind
        score_e_r <- score_e_r[, order(split_data_ind), drop=FALSE]
        score_e_b <- score_e_b[, order(split_data_ind), drop=FALSE]
        # multiplier bootstrap for e^*
        b_e_r <- apply(score_e_r, 1, function(x) mean(b_W*x*est_mu_1/b_mu_1)/mean(b_W))
        b_e_b <- apply(score_e_b, 1, function(x) mean(b_W*x*(1-est_mu_1)/(1-b_mu_1))/mean(b_W))
        # bootstrapped distribution of |sqrt(n)(est_e_g-e_g^*)|
        b_e_r_distr <- abs(sqrt(n)*(b_e_r-target_e[, 1])) # length=num_alg
        b_e_b_distr <- abs(sqrt(n)*(b_e_b-target_e[, 2])) # length=num_alg
        BS_constr_e_r[, b] <- b_e_r_distr
        BS_constr_e_b[, b] <- b_e_b_distr
      }

      if (optimizer=="grid"){
        if (kink){
          b_est_sf <- apply(grid_q, 1,
                            function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                     (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                     as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

          b_est_sf_F_above <- b_est_sf[1]
          b_est_sf_F_below <- b_est_sf[2]
          b_est_sf_grid_above <- b_est_sf[3:(grid_size+2)]
          b_est_sf_grid_below <- b_est_sf[(grid_size+3):(2*grid_size+2)]

          # approximate the directional derivative as per Fang & Santos (2017)
          s_n <- sqrt(n)^{-1/2+0.01} # step size
          if (!allNULL_above){ # bootstrap for the above case
            # direction at which we evaluate the directional derivative of \phi
            b_direc_grid_above <- s_n*sqrt(n)*(b_est_sf_grid_above-est_sf_grid_above) # length=grid_size
            # evaluation point for \phi
            b_eval_grid_above <- est_sf_grid_above+b_direc_grid_above
            # \phi depends on the parameter at which we test
            for (param in 1:num_param_above){ # for the above case
              # current parameter value being tested
              cur_F1 <- F1_above[param]
              cur_F2 <- F2_above[param]

              # estimated inequality constraint associated with the current parameter
              cur_ineq_constr_F_above <- all_ineq_constr_F_above[param]

              # apply the \phi transformation to the evaluation points
              b_cur_ineq_constr_above <- max(max((grid_F_above[,1]*cur_F1+grid_F_above[,2]*cur_F2)-b_eval_grid_above), 0)

              # equality constraint for F being the support point
              b_eq_constr_F_above <- -min(-sqrt(n)*(b_est_sf_F_above-est_sf_F_above), 0)

              BS_constr_F_above[param, b] <- b_eq_constr_F_above+
                (1/s_n)*(b_cur_ineq_constr_above-cur_ineq_constr_F_above)
            }
          }

          if (!allNULL_below){ # bootstrap for the below case
            # direction at which we evaluate the directional derivative of \phi
            b_direc_grid_below <- s_n*sqrt(n)*(b_est_sf_grid_below-est_sf_grid_below) # length=grid_size
            # evaluation point for \phi
            b_eval_grid_below <- est_sf_grid_below+b_direc_grid_below
            # \phi depends on the parameter at which we test
            for (param in 1:num_param_below){ # for the below case
              # current parameter value being tested
              cur_F1 <- F1_below[param]
              cur_F2 <- F2_below[param]

              # estimated inequality constraint associated with the current parameter
              cur_ineq_constr_F_below <- all_ineq_constr_F_below[param]

              # apply the \phi transformation to the evaluation points
              b_cur_ineq_constr_below <- max(max((grid_F_below[,1]*cur_F1+grid_F_below[,2]*cur_F2)-b_eval_grid_below), 0)

              # equality constraint for F being the support point
              b_eq_constr_F_below <- -min(-sqrt(n)*(b_est_sf_F_below-est_sf_F_below), 0)

              BS_constr_F_below[param, b] <- b_eq_constr_F_below+
                (1/s_n)*(b_cur_ineq_constr_below-cur_ineq_constr_F_below)
            }
          }
        } else { ### no kink
          # only the first two directions ((u2-u1)/sqrt(2) and (u1-u2)/sqrt(2)) matter
          b_est_sf <- apply(grid_q[1:2, ], 1,
                            function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                     (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                     as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

          b_est_sf_F_above <- b_est_sf[1]
          b_est_sf_F_below <- b_est_sf[2]

          if (!allNULL_above){
            # equality constraint for F being the support point
            b_eq_constr_F_above <- -min(-sqrt(n)*(b_est_sf_F_above-est_sf_F_above), 0)
            # inequality constraint for F being feasible
            b_ineq_constr_F_above <- max(-sqrt(n)*(b_est_sf_F_above-est_sf_F_above), 0)
            BS_constr_F_above[1, b] <- b_eq_constr_F_above+b_ineq_constr_F_above
          }

          if (!allNULL_below){
            # equality constraint for F being the support point
            b_eq_constr_F_below <- -min(-sqrt(n)*(b_est_sf_F_below-est_sf_F_below), 0)
            # inequality constraint for F being feasible
            b_ineq_constr_F_below <- max(-sqrt(n)*(b_est_sf_F_below-est_sf_F_below), 0)
            BS_constr_F_below[1, b] <- b_eq_constr_F_below+b_ineq_constr_F_below
          }
        }
      } else{ # optimizer=c("SGD", "BFGS")
        # bootstrapped support functions in directions (u2-u1)/sqrt(2) and (u1-u2)/sqrt(2)
        b_est_sf <- apply(matrix(c(1/sqrt(2), -1/sqrt(2),
                                   -1/sqrt(2), 1/sqrt(2)), ncol=2, byrow=TRUE), 1,
                          function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                   (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                   as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})

        b_est_sf_F_above <- b_est_sf[1] # bootstrapped support function at (u2-u1)/sqrt(2)
        b_est_sf_F_below <- b_est_sf[2] # bootstrapped support function at (u1-u2)/sqrt(2)

        if (kink){
          # approximate the directional derivative as per Fang & Santos (2017)
          s_n <- sqrt(n)^{-1/2+0.01} # step size
          if (!allNULL_above){ ## for the above case
            for (param in 1:num_param_above){
              cur_F1 <- F1_above[param]
              cur_F2 <- F2_above[param]

              # estimated inequality constraint associated with the current parameter
              cur_ineq_constr_F_above <- all_ineq_constr_F_above[param]

              if (optimizer=="BFGS"){
                b_cur_f_above <- function(q) { # objective function
                  q1 <- q[1]
                  q2 <- q[2]
                  # estimated support function
                  est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                                   (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                                   as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))

                  # b-th bootstrapped support function
                  b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                      (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                      as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

                  # direction at which we evaluate \phi
                  b_cur_direc <- s_n*sqrt(n)*(b_sf-est_sf)
                  # evaluation point
                  est_sf+b_cur_direc-q1*cur_F1-q2*cur_F2
                }

                # min_q (h_E(q)+s_n*sqrt(n)*{h*_E(q)-h_E(q)})-q'\tilde{F}
                b_result_cur_above <- optim(par=c(cos(7*pi/4), sin(7*pi/4)),
                                            fn=b_cur_f_above,
                                            method="L-BFGS-B",
                                            lower=c(cos(7*pi/4-grid_range), sin(7*pi/4-grid_range)),
                                            upper=c(cos(7*pi/4+grid_range), sin(7*pi/4+grid_range)),
                                            control=control)

                # apply the \phi transformation to the evaluation points
                b_cur_ineq_constr_above <- max(-b_result_cur_above$value, 0)
              } else{ ### optimizer="SGD"
                b_cur_f_above <- function(rad) { # objective function
                  q1 <- cos(rad)
                  q2 <- sin(rad)
                  # estimated support function
                  est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                                   (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                                   as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
                  # b-th bootstrapped support function
                  b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                      (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                      as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
                  # direction at which we evaluate \phi
                  b_cur_direc <- s_n*sqrt(n)*(b_sf-est_sf)
                  # evaluation point
                  est_sf+b_cur_direc-q1*cur_F1-q2*cur_F2
                }

                # initialize argument
                b_rad_above <- torch_tensor(7*pi/4, requires_grad=TRUE)
                # initialize optimization
                b_optim_above <- optim_adam(params=list(b_rad_above), lr=lr)
                # optimize
                for (iter in 1:maxit) {
                  # min_q (h_E(q)+s_n*sqrt(n)*{h*_E(q)-h_E(q)})-q'\tilde{F}
                  b_optim_above$zero_grad()  # initialize gradient
                  b_l_fn_above <- b_cur_f_above(b_rad_above) # loss at current rad
                  b_l_fn_above$backward()    # backpropagation
                  b_optim_above$step()       # descent
                }

                # apply the \phi transformation to the evaluation points
                b_cur_ineq_constr_above <- max(-b_l_fn_above$item(), 0)
              }

              # equality constraint for F being the support point
              b_eq_constr_F_above <- -min(-sqrt(n)*(b_est_sf_F_above-est_sf_F_above), 0)

              BS_constr_F_above[param, b] <- b_eq_constr_F_above+
                (1/s_n)*(b_cur_ineq_constr_above-cur_ineq_constr_F_above)
            }
          }

          if (!allNULL_below){ ## for the below case
            for (param in 1:num_param_below){
              cur_F1 <- F1_below[param]
              cur_F2 <- F2_below[param]

              # estimated inequality constraint associated with the current parameter
              cur_ineq_constr_F_below <- all_ineq_constr_F_below[param]

              if (optimizer=="BFGS"){
                b_cur_f_below <- function(q) { # objective function
                  q1 <- q[1]
                  q2 <- q[2]
                  # estimated support function
                  est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                                   (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                                   as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
                  # b-th bootstrapped support function
                  b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                      (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                      as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
                  # direction at which we evaluate \phi
                  b_cur_direc <- s_n*sqrt(n)*(b_sf-est_sf)
                  # evaluation point
                  est_sf+b_cur_direc-q1*cur_F1-q2*cur_F2
                }

                # min_q (h_E(q)+s_n*sqrt(n)*{h*_E(q)-h_E(q)})-q'\tilde{F}
                b_result_cur_below <- optim(par=c(cos(7*pi/4), sin(7*pi/4)),
                                            fn=b_cur_f_below,
                                            method="L-BFGS-B",
                                            lower=c(cos(7*pi/4-grid_range), sin(7*pi/4-grid_range)),
                                            upper=c(cos(7*pi/4+grid_range), sin(7*pi/4+grid_range)),
                                            control=control)

                # apply the \phi transformation to the evaluation points
                b_cur_ineq_constr_below <- max(-b_result_cur_below$value, 0)
              } else{ ### optimizer="SGD"
                b_cur_f_below <- function(rad) { # objective function
                  q1 <- cos(rad)
                  q2 <- sin(rad)
                  # estimated support function
                  est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                                   (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                                   as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
                  # b-th bootstrapped support function
                  b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                      (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                      as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
                  # direction at which we evaluate \phi
                  b_cur_direc <- s_n*sqrt(n)*(b_sf-est_sf)
                  # evaluation point
                  est_sf+b_cur_direc-q1*cur_F1-q2*cur_F2
                }

                # initialize argument
                b_rad_below <- torch_tensor(7*pi/4, requires_grad=TRUE)
                # initialize optimization
                b_optim_below <- optim_adam(params=list(b_rad_below), lr=lr)
                # optimize
                for (iter in 1:maxit) {
                  # min_q (h_E(q)+s_n*sqrt(n)*{h*_E(q)-h_E(q)})-q'\tilde{F}
                  b_optim_below$zero_grad()  # initialize gradient
                  b_l_fn_below <- b_cur_f_below(b_rad_below) # loss at current rad
                  b_l_fn_below$backward()    # backpropagation
                  b_optim_below$step()       # descent
                }

                # apply the \phi transformation to the evaluation points
                b_cur_ineq_constr_below <- max(-b_l_fn_below$item(), 0)
              }

              # equality constraint for F being the support point
              b_eq_constr_F_below <- -min(-sqrt(n)*(b_est_sf_F_below-est_sf_F_below), 0)

              BS_constr_F_below[param, b] <- b_eq_constr_F_below+
                (1/s_n)*(b_cur_ineq_constr_below-cur_ineq_constr_F_below)
            }
          }
        } else{ ## no kink
          if (!allNULL_above){
            # equality constraint for F being the support point
            b_eq_constr_F_above <- -min(-sqrt(n)*(b_est_sf_F_above-est_sf_F_above), 0)
            # inequality constraint for F being feasible
            b_ineq_constr_F_above <- max(-sqrt(n)*(b_est_sf_F_above-est_sf_F_above), 0)
            BS_constr_F_above[1, b] <- b_eq_constr_F_above+b_ineq_constr_F_above
          }

          if (!allNULL_below){
            # equality constraint for F being the support point
            b_eq_constr_F_below <- -min(-sqrt(n)*(b_est_sf_F_below-est_sf_F_below), 0)
            # inequality constraint for F being feasible
            b_ineq_constr_F_below <- max(-sqrt(n)*(b_est_sf_F_below-est_sf_F_below), 0)
            BS_constr_F_below[1, b] <- b_eq_constr_F_below+b_ineq_constr_F_below
          }
        }
      }
    }
    ### ------- BOOTSTRAP ENDS -------
  }
  ## compute critical values
  # if est e & kink: cv is num_alg by num_param, ts is num_alg by num_param
  # if est e & no kink: cv is num_alg by 1, ts is num_alg by num_param
  # if cst e & kink: cv is 1 by num_param, ts is 1 by num_param
  # if cst e & no kink: cv is 1x1, ts is 1 by num_param
  rej_above <- c()
  rej_below <- c()
  if (!is.null(target_alg_preds)){ # if e^* is estimated
    if (!allNULL_above){
      if (kink){
        cv_above <- matrix(NA, nrow=num_alg, ncol=num_param_above)
      } else {
        cv_above <- matrix(NA, nrow=num_alg, ncol=1)
      }

      for (alg in 1:num_alg){
        alg_BS_constr_e <- BS_constr_e_r[alg, ]+BS_constr_e_b[alg, ] # length=num_bstp_rep
        alg_constr_above <- sweep(BS_constr_F_above, 2, alg_BS_constr_e, "+") # num_param_above (if kink) or 1 (if no kink) by num_bstp_rep
        cv_above[alg, ] <- apply(alg_constr_above, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj}) # length=num_param_above (if kink) or 1 (if no kink)
      }
      rej_above <- apply(test_stat_above, 2, function(col) as.numeric(col > cv_above))
    }

    if (!allNULL_below){
      if (kink){
        cv_below <- matrix(NA, nrow=num_alg, ncol=num_param_below)
      } else {
        cv_below <- matrix(NA, nrow=num_alg, ncol=1)
      }

      for (alg in 1:num_alg){
        alg_BS_constr_e <- BS_constr_e_r[alg, ]+BS_constr_e_b[alg, ] # length=num_bstp_rep
        alg_constr_below <- sweep(BS_constr_F_below, 2, alg_BS_constr_e, "+") # num_param_below (if kink) or 1 (if no kink) by num_bstp_rep
        cv_below[alg, ] <- apply(alg_constr_below, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj}) # length=num_param_below (if kink) or 1 (if no kink)
      }

      rej_below <- apply(test_stat_below, 2, function(col) as.numeric(col > cv_below))
    }
  } else{ ## if e^* is not estimated
    if (!allNULL_above){
      cv_above <- apply(BS_constr_F_above, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj})
      rej_above <- as.numeric(test_stat_above > cv_above)
    }

    if (!allNULL_below){
      cv_below <- apply(BS_constr_F_below, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj})
      rej_below <- as.numeric(test_stat_below > cv_below)
    }
  }

  return(list("test_stat_above"=test_stat_above,
              "test_stat_below"=test_stat_below,
              "cv_above"=cv_above, # will return NA matrix if is.null(num_bstp_rep)
              "cv_below"=cv_below, # will return NA matrix if is.null(num_bstp_rep)
              "rej_above"=rej_above, # will return c() if is.null(num_bstp_rep)
              "rej_below"=rej_below # will return c matrix if is.null(num_bstp_rep)
  ))
}



## Function that returns a list of initial parameter values by grid search
CS_eF_disjoint_init_par <- function(X, G, Y,
                                    est_nuisance=NULL,
                                    F1_above, F2_above,
                                    F1_below, F2_below,
                                    grid_size=1000,
                                    grid_range=pi/4,
                                    l="classification",
                                    method="lasso-logit",
                                    fold=5){
  num_param_above <- length(F1_above)
  num_param_below <- length(F1_below)

  ## create grids for the set of directions indexing the moment inequalities
  # when the feasible set is ABOVE the 45-degree line
  range_above <- seq(7*pi/4-grid_range, 7*pi/4+grid_range, length.out=grid_size)
  grid_F_above <- t(as.matrix(sapply(range_above,
                                     function(rad){c(cos(rad), sin(rad))})))

  range_below <- seq(3*pi/4-grid_range, 3*pi/4+grid_range, length.out=grid_size)
  grid_F_below <- t(as.matrix(sapply(range_below,
                                     function(rad){c(cos(rad), sin(rad))}))) # when the feasible set is BELOW the 45-degree line

  grid_q <- rbind(c(cos(7*pi/4), sin(7*pi/4)), # (u2-u1)/sqrt(2)
                  c(cos(3*pi/4), sin(3*pi/4)), # (u1-u2)/sqrt(2)
                  grid_F_above,
                  grid_F_below)

  # estimate the support functions for the directions in grid_q
  est <- support_function(X=X, G=G, Y=Y,
                          q=grid_q, est_nuisance=est_nuisance,
                          l=l, method=method, fold=fold)
  # estimated support functions
  est_sf <- est$est_sf
  est_sf_F_above <- est_sf[1] # estimated sf in direction (u2-u1)/sqrt(2)
  est_sf_F_below <- est_sf[2] # estimated sf in direction (u1-u2)/sqrt(2)
  est_sf_grid_above <- est_sf[3:(grid_size+2)] # estimated sf for q \in grid_F_above
  est_sf_grid_below <- est_sf[(grid_size+3):(2*grid_size+2)] # estimated sf for q \in grid_F_below

  arg_max_rad_above <- matrix(NA, nrow=1, ncol=num_param_above)
  arg_max_rad_below <- matrix(NA, nrow=1, ncol=num_param_below)

  for (param in 1:num_param_above){
    # current candidate value of F being tested
    cur_F1 <- F1_above[param]
    cur_F2 <- F2_above[param]

    # inequality constraints for F being feasible
    arg_max_rad_above[1,param] <- range_above[which.max((grid_F_above[,1]*cur_F1+grid_F_above[,2]*cur_F2)-est_sf_grid_above)] # max_q (q'\tilde{F}-h_E(q))
  }

  for (param in 1:num_param_below){
    # current candidate value of F being tested
    cur_F1 <- F1_below[param]
    cur_F2 <- F2_below[param]

    # inequality constraints for F being feasible
    arg_max_rad_below[1,param] <- range_below[which.max((grid_F_below[,1]*cur_F1+grid_F_below[,2]*cur_F2)-est_sf_grid_below)] # max_q (q'\tilde{F}-h_E(q))
  }

  return(list("arg_max_rad_above"=arg_max_rad_above,
              "arg_max_rad_below"=arg_max_rad_below))
}



## Function that constructs CS_n^{45} for the distance-to-F test
CS_eF_cross <- function(Y, X, G,
                        est_nuisance=NULL, # pass the list returned by the `nuisance` function; if NULL, `nuisance` will be called to do the nuisance estimation.
                        F1, F2, e1=NULL, e2=NULL, # candidate (e, F) for the test inversion; if is.null(target_alg_preds)=FALSE, `e1` and `e2` should be matrices with nrow=nrow(target_alg_preds) and ncol=length(F1)=length(F2). If `target_e` is specified, `e1` and `e2` are ignored.
                        target_alg_preds=NULL, # a matrix with n columns collecting the target algorithm's predicted value at each X, a^*(X). The rows stack different target algorithms. If !is.null(target_alg_preds), then `CS_eF_cross` computes the estimated error allocation for each target algorithm, and the argument `target_e` is ignored.
                        target_e=NULL, # a matrix with ncol=2; used when the error allocation is taken as non-random, i.e, NOT estimated from a given target algorithm; this argument is ignored when `target_alg_preds` is specified.
                        dist_func=c("euclidean", "euclidean-sq", "max"), # distance function rho; choose either from c("euclidean", "euclidean-sq", "max") or specify a function
                        num_bstp_rep=NULL, # number of bootstrap replications; if NULL, then `CS_eF_cross` returns the test statistic without bootstrapped critical values.
                        optimizer=c("grid", "SGD", "BFGS", "optimize"),
                        grid_size=10000, # size of the grid used to discretize the support of the function being optimized; used when optimizer="grid".
                        control=list(maxit=10000), gradient=TRUE, # for optimizer="BFGS"
                        lr=0.01, maxit=20000, # for optimizer="SGD"
                        init_par=NULL, # for optimizer=c("SGD", "BFGS"); if NULL, will call `F45_init_par` to find initial values via grid search; if specified, should be one number.
                        kink=FALSE, # whether the feasible set has kinks; if TRUE, then the numerical approximation from Fang & Santos (2019) is used for the directional derivation. Default is FALSE.
                        alpha=0.05, # significance level; not used if `num_bstp_rep` is NULL.
                        infntsml_adj=1e-6, # infinitesimal adjustment factor; not used if `num_bstp_rep` is NULL
                        l="classification",
                        method="lasso-logit",
                        fold=5){
  Y <- data.matrix(Y)
  X <- data.matrix(X)
  G <- data.matrix(G)
  n <- length(Y)   # sample size

  ## validate distance function
  rho_sqE <- FALSE
  if (is.character(dist_func)){
    if (dist_func=="euclidean"){
      rho <- function(a1, a2, b1, b2) return(sqrt((a1-b1)^2+(a2-b2)^2))
    } else if (dist_func=="euclidean-sq"){
      rho <- function(a1, a2, b1, b2) return((a1-b1)^2+(a2-b2)^2)
      rho_sqE <- TRUE
    } else if (dist_func=="max") {
      rho <- function(a1, a2, b1, b2) return(pmax(abs(a1-b1), abs(a2-b2)))
    } else{
      stop(paste0("Argument `dist_func` is not in c('euclidean', 'euclidean-sq', 'max') or an object of class 'function'."))
    }
  } else{
    if (!inherits(dist_func, "function")){
      stop(paste0("Argument `dist_func` is not in c('euclidean', 'euclidean-sq', 'max') or an object of class 'function'."))
    }
    if (length(formals(dist_func))!=4){
      stop(cat(paste0("The number of arguments in the distance function should be 4. For example, for Euclidean distance, `dist_func` should be defined as: \n dist_func <- function(a1, a2, b1, b2) return(sqrt((a1-b1)^2+(a2-b2)^2))\nCheck your definition of `dist_func`.")))
    }
    rho <- dist_func
  }

  # estimate nuisance parameters
  if (is.null(est_nuisance)){
    est_nuisance <- nuisance(Y=Y, X=X, G=G, l=l,
                             method=method, fold=fold)
  }

  diff_theta_r <- est_nuisance[[1]]
  diff_theta_b <- est_nuisance[[2]]
  est_mu_1 <- est_nuisance[[3]]
  split_data_ind <- est_nuisance[[4]]

  Y_ordered <- data.matrix(Y)[order(split_data_ind)]
  G_ordered <- data.matrix(G)[order(split_data_ind)]

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      loss <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      loss <- function(d,y) return((d-y)^2)
    }
  } else {
    loss <- l
  }

  # labels L
  L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
  L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
  L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
  L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)

  diff_L_r <- L_11-L_01
  diff_L_b <- L_10-L_00

  if (optimizer=="grid"){ ### grid search to estimate F_45
    # discretize the support of the scalar c that index the support function of the intersection
    grid_c <- runif(grid_size, min = -1e3, max = 1e3)
    # the grid of directions to optimize for is u1-c[1, -1]'=[-1-c, c]' for c in grid_c
    grid_q <- data.matrix(cbind(-1-grid_c, grid_c))

    ## estimate the support functions for the directions in grid_q
    est_sf_E_val <- support_function(X=X, G=G, Y=Y,
                                     q=grid_q,
                                     est_nuisance=est_nuisance)

    est_sf_E <- est_sf_E_val$est_sf
    est_F45_coord <- -min(est_sf_E) # estimated coordinates of F_45
    arg_inf_index <- which.min(est_sf_E)
  } else { ### optimizer to estimate F_45
    if (is.null(init_par) & optimizer!="optimize"){ # set initial opt parameter values
      init_par <- F45_init_par(X=X, G=G, Y=Y,
                               est_nuisance=est_nuisance,
                               grid_size=grid_size)
    }

    f <- function(c){ # objective function
      mean((-1-c)*L_01/est_mu_1+c*L_00/(1-est_mu_1)+
             ((-1-c)*diff_L_r/est_mu_1+c*diff_L_b/(1-est_mu_1))*
             as.numeric((-1-c)*diff_theta_r/est_mu_1+c*diff_theta_b/(1-est_mu_1)>0))
    }

    if (optimizer=="BFGS"){### optimizer is BFGS
      if (gradient){ ## if gradient=TRUE, closed-form gradient formula is used
        gradient <- function(c){ # gradient
          mean(-1*L_01/est_mu_1+L_00/(1-est_mu_1)+
                 (-1*diff_L_r/est_mu_1+diff_L_b/(1-est_mu_1))*
                 as.numeric((-1-c)*diff_theta_r/est_mu_1+c*diff_theta_b/(1-est_mu_1)>0))
        }
      } else { ## otherwise let optim() numerically approximate the gradient
        gradient <- NULL
      }

      result <- optim(par=init_par,
                      fn=f, gr=gradient,
                      method="BFGS",
                      control=control)
      est_F45_coord <- -result$value
      arg_inf_c <- result$par
    } else if (optimizer=="SGD") { ### optimizer is SGD
      c <- torch_tensor(init_par, requires_grad=TRUE)
      optim <- optim_adam(params=list(c), lr=lr)
      for (iter in 1:maxit) {
        optim$zero_grad()
        l_fn <- f(c) # loss at current c
        l_fn$backward()
        optim$step()
      }
      est_F45_coord <- -l_fn$item()
      arg_inf_c <- c$item()
    } else if (optimizer=="optimize"){
      result <- optimize(f, c(-1e3, 1e3), tol = 0.0001)
      est_F45_coord <- -result$objective
      arg_inf_c <- result$minimum
    }
  }

  ## validate input target_alg_preds
  if (!is.null(target_alg_preds)){
    if (!inherits(target_alg_preds, "matrix") | (inherits(target_alg_preds, "matrix") & ncol(target_alg_preds) != n)){
      stop(paste0("Target algorithm `target_alg_preds` must be of class 'matrix' that contains n:=length(Y) columns."))
    } else if (length(unique(c(nrow(e1), nrow(e2), nrow(target_alg_preds))))!=1 | length(unique(c(length(F1), length(F2), ncol(e1), ncol(e2))))!=1) {
      stop(paste0("When `target_alg_preds` is specified, `e1` and `e2` should be matrices with nrow=nrow(target_alg_preds) and ncol=length(F1)=length(F2)."))
    } else {
      num_alg <- nrow(target_alg_preds) # number of algorithms

      # first we need the scores that yield estimated e_g^*
      score_e_r <- matrix(NA, nrow=num_alg, ncol=n)
      score_e_b <- matrix(NA, nrow=num_alg, ncol=n)

      for (alg in 1:num_alg){ # for each algorithm
        cur_alg_preds <- target_alg_preds[alg,] # retrieve current predictions a^*(X)
        cur_score <- cur_alg_preds*as.numeric(loss(1, Y))+(1-cur_alg_preds)*as.numeric(loss(0, Y))
        score_e_r[alg,] <- as.numeric(G==1)*cur_score/est_mu_1
        score_e_b[alg,] <- as.numeric(G==0)*cur_score/(1-est_mu_1)
      }
      # estimate e^*
      target_e_r <- apply(score_e_r, 1, FUN = mean)
      target_e_b <- apply(score_e_b, 1, FUN = mean)
      target_e <- data.matrix(cbind(target_e_r, target_e_b))
    }
  } else {
    if (!inherits(target_e, "matrix") | (inherits(target_e, "matrix") & ncol(target_e) != 2)){
      stop(paste0("Target error allocation `target_e` must be of class matrix with ncol=2."))
    } else{
      num_alg <- nrow(target_e)
    }
  }

  num_param <- length(F1) # number of candidate parameters to test

  # estimated distance for each algorithm; length=num_alg
  est_dist <- apply(target_e, 1,
                    function(x) rho(x[1], x[2], est_F45_coord, est_F45_coord))
  # distance for each candidate (e, F); num_alg by num_param
  cand_dist <- matrix(NA, nrow=num_alg, ncol=num_param)
  if (!is.null(target_alg_preds)){ # if e^* is estimated
    for (alg in 1:num_alg){
      cand_dist[alg, ] <- rho(e1[alg, ], e2[alg, ], F1, F2)
    }
  } else{ # if e^* is not estimated
    cand_dist <- t(apply(target_e, 1, function(x) rho(x[1], x[2], F1, F2)))
  }

  # test statistic, num_alg by num_param
  test_stat <- matrix(NA, nrow=num_alg, ncol=num_param)
  for (alg in 1:num_alg){
    test_stat[alg,] <- abs(sqrt(n)*(est_dist[alg]-cand_dist[alg,]))
  }

  ## initialize vector to collect bootstrapped crit val; not used if num_bstp_rep=NULL
  btsrp_cv <- c()

  if (!is.null(num_bstp_rep)){
    # initialize matrix to store bootstrapped test statistic for each row of target_e
    BS_test_stat <- matrix(NA, nrow=num_alg, ncol=num_bstp_rep)

    if (!is.null(target_alg_preds)){ # if e^* is estimated, needs to reshuffle the obs indices by split_data_ind
      score_e_r <- score_e_r[, order(split_data_ind), drop=FALSE]
      score_e_b <- score_e_b[, order(split_data_ind), drop=FALSE]
    }

    if (kink){
      s_n <- sqrt(n)^{-1/2+0.01} # step size for approximating the directional derivative
    }

    ## bootstrap once for all rows of target_e
    # this part doesn't depend on which candidate (e, F) is being tested
    # ----- BOOTSTRAP STARTS -----
    for (b in 1:num_bstp_rep){
      # draw n exponential(1) weights
      b_W <- rexp(n, rate=1)
      # bootstrapped mu_1
      b_mu_1 <- mean(b_W*G_ordered)/mean(b_W)

      if (kink | !rho_sqE){ # if there's kink or rho is not squared-Euclidean
        if (optimizer=="grid"){
          # b-th bootstrapped sf of E; has length grid_size
          b_sf_E <- apply(grid_q, 1,
                          function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                   (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                   as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)})
          # direction at which we evaluate the directional derivative of \phi
          b_direc_sf_E <- s_n*sqrt(n)*(b_sf_E-c(est_sf_E)) # length=grid_size
          # evaluate the inf part of \phi
          b_eval_sf_E <- c(est_sf_E)+b_direc_sf_E # evaluation point = est + direc
          b_F45_coord <- -min(b_eval_sf_E) # apply the inf function
        } else{ ## optimizer=c("SGD", "BFGS")
          b_f <- function(c){ # objective function
            # estimated h_E(u1(c))
            est_sf <- mean((-1-c)*L_01/est_mu_1+c*L_00/(1-est_mu_1)+
                             ((-1-c)*diff_L_r/est_mu_1+c*diff_L_b/(1-est_mu_1))*
                             as.numeric((-1-c)*diff_theta_r/est_mu_1+c*diff_theta_b/(1-est_mu_1)>0))

            # b-th bootstrapped h_E(u1(c))
            b_sf <- mean(b_W*((-1-c)*L_01/b_mu_1+c*L_00/(1-b_mu_1)+
                                ((-1-c)*diff_L_r/b_mu_1+c*diff_L_b/(1-b_mu_1))*
                                as.numeric((-1-c)*diff_theta_r/b_mu_1+c*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

            # direction at which we evaluate phi
            b_direc <- s_n*sqrt(n)*(b_sf-est_sf)

            # evaluation point
            est_sf+b_direc
          }

          if (optimizer=="SGD"){
            c <- torch_tensor(arg_inf_c, requires_grad=TRUE)
            # min_c (h_E(u1(c))+s_n*sqrt(n)*{h*_E(u1(c))-h_E(u1(c))})
            optim <- optim_adam(params=list(c), lr=lr)
            for (iter in 1:maxit) {
              optim$zero_grad()
              b_loss <- b_f(c)
              b_loss$backward()
              optim$step()
            }
            # b-th bootstrapped F_45
            b_F45_coord <- -b_loss$item()
          } else if (optimizer=="BFGS"){ ## optimizer=="BFGS"
            if (!is.null(gradient)){ ## if gradient=TRUE, closed-form gradient formula is used
              b_gradient <- function(c){ # gradient
                g_est_sf <- mean(-1*L_01/est_mu_1+L_00/(1-est_mu_1)+
                                   (-1*diff_L_r/est_mu_1+diff_L_b/(1-est_mu_1))*
                                   as.numeric((-1-c)*diff_theta_r/est_mu_1+c*diff_theta_b/(1-est_mu_1)>0))

                # b-th bootstrapped h_E(u1(c))
                g_b_sf <- mean(b_W*(-1*L_01/b_mu_1+L_00/(1-b_mu_1)+
                                      (-1*diff_L_r/b_mu_1+diff_L_b/(1-b_mu_1))*
                                      as.numeric((-1-c)*diff_theta_r/b_mu_1+c*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

                # direction at which we evaluate phi
                g_b_direc <- s_n*sqrt(n)*(g_b_sf-g_est_sf)

                # evaluation point
                g_est_sf+g_b_direc
              }
            } else { ## otherwise let optim() numerically approximate the gradient
              b_gradient <- NULL
            }

            b_result <- optim(par=arg_inf_c, fn=b_f, gr=b_gradient,
                              method="BFGS", control=control)
            b_F45_coord <- -b_result$value
          } else if (optimizer=="optimize"){
            b_result <- optimize(b_f, c(-1e3, 1e3), tol = 0.0001)
            b_F45_coord <- -b_result$objective
          }
        }

        # b-th bootstrapped distribution of F45
        b_F45_distr <- (1/s_n)*(b_F45_coord-est_F45_coord)

        # b-th bootstrapped e^*
        if (!is.null(target_alg_preds)){
          b_e_r <- apply(score_e_r, 1, function(x) mean(b_W*x*est_mu_1/b_mu_1)/mean(b_W))
          b_e_b <- apply(score_e_b, 1, function(x) mean(b_W*x*(1-est_mu_1)/(1-b_mu_1))/mean(b_W))

          if (rho_sqE){ # if rho is the Euclidean distance
            # bootstrapped distribution of sqrt(n)(est_e_g-e_g^*)
            b_e_r_distr <-sqrt(n)*(b_e_r-target_e[, 1]) # length=num_alg
            b_e_b_distr <-sqrt(n)*(b_e_b-target_e[, 2])

            # apply delta method and CMP for abs.val.
            BS_test_stat[, b] <- abs((2*(est_F45_coord-target_e[, 1])+2*(est_F45_coord-target_e[, 2]))*b_F45_distr-
                                       2*(est_F45_coord-target_e[, 1])*b_e_r_distr-
                                       2*(est_F45_coord-target_e[, 2])*b_e_b_distr)
          } else{
            # direction at which we evaluate the directional derivative of \phi
            b_direc_e_r <- s_n*sqrt(n)*(b_e_r-target_e[, 1]) # length=num_alg
            b_direc_e_b <- s_n*sqrt(n)*(b_e_b-target_e[, 2]) # length=num_alg

            # evaluation point for e^*; length=num_alg
            b_eval_e_r <- target_e[, 1] + b_direc_e_r # evaluation point = est + direc
            b_eval_e_b <- target_e[, 2] + b_direc_e_b # evaluation point = est + direc
            # evaluate \phi; length=num_alg
            b_dist <- apply(cbind(b_eval_e_r, b_eval_e_b), 1,
                            function(x) rho(x[1], x[2], b_F45_coord, b_F45_coord))

            BS_test_stat[, b] <- abs((1/s_n)*(b_dist-est_dist))
          }
        } else{ # if e^* is not estimated,
          if (rho_sqE){ # if rho is the Euclidean distance
            # apply delta method (when rho is squared-Euclidean) and CMP for abs.val.
            BS_test_stat[, b] <- abs((2*(est_F45_coord-target_e[, 1])+2*(est_F45_coord-target_e[, 2]))*b_F45_distr)
          } else {
            # evaluate \phi; length=num_alg
            b_dist <- apply(cbind(target_e[, 1], target_e[, 2]), 1,
                            function(x) rho(x[1], x[2], b_F45_coord, b_F45_coord))
            BS_test_stat[, b] <- abs((1/s_n)*(b_dist-est_dist))
          }
        }
      } else{ ## if there's no kink AND rho is squared-Euclidean
        if (optimizer=="grid"){
          opt_q <- grid_q[arg_inf_index, ]
          b_F45_coord <- -mean(b_W*(opt_q[1]*L_01/b_mu_1+opt_q[2]*L_00/(1-b_mu_1)+
                                      (opt_q[1]*diff_L_r/b_mu_1+opt_q[2]*diff_L_b/(1-b_mu_1))*
                                      as.numeric(opt_q[1]*diff_theta_r/b_mu_1+opt_q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
        } else{ ## optimizer=c("SGD", "BFGS")
          b_F45_coord <- -mean(b_W*((-1-arg_inf_c)*L_01/b_mu_1+arg_inf_c*L_00/(1-b_mu_1)+
                                      ((-1-arg_inf_c)*diff_L_r/b_mu_1+arg_inf_c*diff_L_b/(1-b_mu_1))*
                                      as.numeric((-1-arg_inf_c)*diff_theta_r/b_mu_1+arg_inf_c*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
        }

        # bootstrapped distribution of sqrt(n)(est_F45_coord-true_F)
        b_F45_distr <- sqrt(n)*(b_F45_coord-est_F45_coord)

        # b-th bootstrapped e^*
        if (!is.null(target_alg_preds)){
          b_e_r <- apply(score_e_r, 1, function(x) mean(b_W*x*est_mu_1/b_mu_1)/mean(b_W)) # length=num_alg
          b_e_b <- apply(score_e_b, 1, function(x) mean(b_W*x*(1-est_mu_1)/(1-b_mu_1))/mean(b_W))

          # bootstrapped distribution of sqrt(n)(est_e_g-e_g^*)
          b_e_r_distr <-sqrt(n)*(b_e_r-target_e[, 1]) # length=num_alg
          b_e_b_distr <-sqrt(n)*(b_e_b-target_e[, 2])

          # apply delta method (when rho is squared-Euclidean) and CMP for abs.val.
          BS_test_stat[, b] <- abs((2*(est_F45_coord-target_e[, 1])+2*(est_F45_coord-target_e[, 2]))*b_F45_distr-
                                     2*(est_F45_coord-target_e[, 1])*b_e_r_distr-
                                     2*(est_F45_coord-target_e[, 2])*b_e_b_distr)
        } else{ # if e^* is not estimated
          # apply delta method (when rho is squared-Euclidean) and CMP for abs.val.
          BS_test_stat[, b] <- abs((2*(est_F45_coord-target_e[, 1])+2*(est_F45_coord-target_e[, 2]))*b_F45_distr)
        }
      }
    }
    # ----- BOOTSTRAP ENDS -----

    # compute the bootstrapped crit val, as per Step 3, Procedure 2
    btsrp_cv <- apply(BS_test_stat, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj}) # length=num_alg
  }

  return(list("test_stat"=test_stat,
              "BScv"=btsrp_cv, # will return BScv=c() if num_bstp_rep=NULL
              "rej"=apply(test_stat, 2, function(x) as.numeric(x > btsrp_cv)) # will return rej=numeric(0) if num_bstp_rep=NULL
  ))
}


## Function that finds initial values for the optimizers in `CS_eF_cross` using grid search
F45_init_par <- function(X, G, Y,
                         est_nuisance=NULL,
                         grid_size=10000,
                         l="classification",
                         method="lasso-logit",
                         fold=5){
  # pre-estimate arg_inf_c by grid search
  grid_c <- runif(grid_size, min=-1e3, max=1e3)
  grid_q <- data.matrix(cbind(-1-grid_c, grid_c))
  est_sf_E_val <- support_function(X=X, G=G, Y=Y,
                                   est_nuisance=est_nuisance,
                                   q=grid_q,
                                   l=l, method=method, fold=fold)
  return(grid_c[which.min(est_sf_E_val$est_sf)])
}


## Function that creates a grid of candidate values of (e, F) for constructing the confidence set for the distance-to-F test
create_candidate_eF <- function(X, G, Y, alg_preds,
                                est_nuisance=NULL,
                                est_F45_coord=NULL,
                                num_grid=2000,
                                buffer=sqrt(log(n))/sqrt(n),
                                optimizer=c("SGD", "grid"),
                                lr=0.1, maxit=10000, init_par=0,
                                l="classification",
                                method="lasso-logit",
                                fold=5){
  # estimate the smallest octagon containing the feasible set
  oct <- support_function(X=X, G=G, Y=Y,
                          est_nuisance=est_nuisance,
                          q=matrix(c(-1, 0,
                                     1, 0,
                                     0, 1,
                                     0, -1,
                                     -1/sqrt(2), -1/sqrt(2),
                                     1/sqrt(2), 1/sqrt(2),
                                     1/sqrt(2), -1/sqrt(2),
                                     -1/sqrt(2), 1/sqrt(2)), byrow=TRUE, ncol=2),
                          l=l, method=method, fold=fold)

  # sides of this octagon, enlarged by buffer
  a <- oct$est_sf[1]+buffer
  b <- oct$est_sf[2]+buffer
  c <- oct$est_sf[3]+buffer
  d <- oct$est_sf[4]+buffer
  e <- oct$est_sf[5]+buffer
  f <- oct$est_sf[6]+buffer
  g <- oct$est_sf[7]+buffer
  h <- oct$est_sf[8]+buffer

  ## create grid for e^*
  est_mu_1 <- mean(G)
  score <- alg_preds*as.numeric(Y!=1)+(1-alg_preds)*as.numeric(Y!=0)
  est_e_r <- mean(as.numeric(G==1)*score)/est_mu_1
  est_e_b <- mean(as.numeric(G==0)*score)/(1-est_mu_1)

  size <- round(sqrt(num_grid))

  # Gaussian grid centered at e^*
  grid_e <- data.matrix(
    expand.grid(rnorm(4*size,
                      mean=est_e_r,
                      sd=buffer),
                rnorm(4*size,
                      mean=est_e_b,
                      sd=buffer))
  )

  # check if the octagon (minus buffer) intersects with the 45-degree line such that the h side
  # and the g side are at least buffer away from the 45-degree
  if (h-buffer > buffer & g-buffer > buffer){ # if so we only test for the case of intersection
    if (is.null(est_F45_coord)){
      if (optimizer=="SGD"){
        ### ESTIMATE F_45
        if (is.null(est_nuisance)){
          est_nuisance <- nuisance(Y=Y, X=X, G=G, l=l, method=method, fold=fold)
        }

        diff_theta_r <- est_nuisance[[1]]
        diff_theta_b <- est_nuisance[[2]]
        est_mu_1 <- est_nuisance[[3]]
        split_data_ind <- est_nuisance[[4]]

        # set up the loss function
        if (is.character(l)){
          if (l=="classification"){
            loss <- function(d,y) return(as.numeric(d != y))
          } else if (l=="square-loss") {
            loss <- function(d,y) return((d-y)^2)
          }
        } else{ loss <- l }

        Y_ordered <- data.matrix(Y)[order(split_data_ind)]
        G_ordered <- data.matrix(G)[order(split_data_ind)]

        # labels L
        L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
        L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
        L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
        L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)

        diff_L_r <- L_11-L_01
        diff_L_b <- L_10-L_00

        ### stochastic gradient descent
        f <- function(c){ # objective function
          mean((-1-c)*L_01/est_mu_1+c*L_00/(1-est_mu_1)+
                 ((-1-c)*diff_L_r/est_mu_1+c*diff_L_b/(1-est_mu_1))*
                 as.numeric((-1-c)*diff_theta_r/est_mu_1+c*diff_theta_b/(1-est_mu_1)>0))
        }

        c <- torch_tensor(init_par, requires_grad=TRUE)
        optim <- optim_adam(params=list(c), lr=lr)
        for (i in 1:maxit) {
          optim$zero_grad()
          loss <- f(c)
          loss$backward()
          optim$step()
        }
        est_F45_coord <- -loss$item()
      } else {
        # the grid of directions to optimize for is u1-c[1, -1]'=[-1-c, c]' for c in grid_c
        grid_c <- runif(50000, min=-1e3, max=1e3)
        grid_q <- data.matrix(cbind(-1-grid_c, grid_c))

        ## estimate the support functions for the directions in grid_q
        est_sf_E_val <- support_function(X=X, G=G, Y=Y,
                                         q=grid_q,
                                         est_nuisance=est_nuisance,
                                         l=l, method=method, fold=fold)
        est_F45_coord <- -min(est_sf_E_val$est_sf)
      }
    }

    sample_F45 <- rnorm(num_grid, mean=est_F45_coord, sd=buffer)
    grid_F_cross <- cbind(sample_F45, sample_F45)

    grid_e_cross <- grid_e[sample(1:nrow(grid_e), size=nrow(grid_F_cross), replace=FALSE), ]

    return(list("cand_F_above"=NULL,
                "cand_e_above"=NULL,
                "cand_F_below"=NULL,
                "cand_e_below"=NULL,
                "cand_F_cross"=grid_F_cross,
                "cand_e_cross"=grid_e_cross))

  } else if (g-buffer < -buffer){ # if the octagon (minus buffer) is completely above the 45-degree line and at least -buffer away from the 45-degree line, we only test for CS_n^+
    est_F_above <- support_function(X=X, G=G, Y=Y,
                                    est_nuisance=est_nuisance,
                                    q=matrix(c(1/sqrt(2), -1/sqrt(2),
                                               1/sqrt(2), -1/sqrt(2)), ncol=2),
                                    v=matrix(c(1, 0,
                                               0, 1), ncol=2),
                                    l=l, method=method, fold=fold)
    est_F_above_r <- est_F_above$est_sf[1] # estimated first coord of F
    est_F_above_b <- est_F_above$est_sf[2] # estimated second coord of F

    # Gaussian grid centered at est_F_above
    grid_F_above <- data.matrix(
      expand.grid(rnorm(4*size,
                        mean=est_F_above_r,
                        sd=buffer),
                  rnorm(4*size,
                        mean=est_F_above_b,
                        sd=buffer))
    )

    # subset to points above the 45-degree line
    grid_F_above <- grid_F_above[which(grid_F_above[,1]<=grid_F_above[,2]),]
    grid_e_above <- grid_e[which(grid_e[,1]<=grid_e[,2]),]

    size_above <- min(nrow(grid_e_above), nrow(grid_F_above), num_grid)

    if (size_above > 0){
      grid_e_above <- grid_e_above[sample(1:nrow(grid_e_above), size=size_above, replace = FALSE), ]
      grid_F_above <- grid_F_above[sample(1:nrow(grid_F_above), size=size_above, replace = FALSE), ]
    } else{
      grid_e_above <- NULL
      grid_F_above <- NULL
    }

    return(list("cand_F_above"=grid_F_above,
                "cand_e_above"=grid_e_above,
                "cand_F_below"=NULL,
                "cand_e_below"=NULL,
                "cand_F_cross"=NULL,
                "cand_e_cross"=NULL))
  } else if (h-buffer < -buffer){ # if the octagon (minus buffer) is completely below the 45-degree line and at least -buffer away from the 45-degree line, we only test for CS_n^-
    est_F_below <- support_function(X=X, G=G, Y=Y,
                                    est_nuisance=est_nuisance,
                                    q=matrix(c(-1/sqrt(2), 1/sqrt(2),
                                               -1/sqrt(2), 1/sqrt(2)), ncol=2),
                                    v=matrix(c(1, 0,
                                               0, 1), ncol=2),
                                    l=l, method=method, fold=fold)
    est_F_below_r <- est_F_below$est_sf[1] # estimated first coord of F
    est_F_below_b <- est_F_below$est_sf[2] # estimated second coord of F

    # Gaussian grid centered at est_F_above
    grid_F_below <- data.matrix(
      expand.grid(rnorm(4*size,
                        mean=est_F_below_r,
                        sd=buffer),
                  rnorm(4*size,
                        mean=est_F_below_b,
                        sd=buffer))
    )

    # subset to points above the 45-degree line
    grid_F_below <- grid_F_below[which(grid_F_below[,1]>=grid_F_below[,2]),]
    grid_e_below <- grid_e[which(grid_e[,1]>=grid_e[,2]),]

    size_below <- min(nrow(grid_e_below), nrow(grid_F_below), num_grid)

    if (size_below > 0){
      grid_e_below <- grid_e_below[sample(1:nrow(grid_e_below), size=size_below, replace = FALSE), ]
      grid_F_below <- grid_F_below[sample(1:nrow(grid_F_below), size=size_below, replace = FALSE), ]
    } else{
      grid_e_below <- NULL
      grid_F_below <- NULL
    }

    return(list("cand_F_above"=NULL,
                "cand_e_above"=NULL,
                "cand_F_below"=grid_F_below,
                "cand_e_below"=grid_e_below,
                "cand_F_cross"=NULL,
                "cand_e_cross"=NULL))

  } else{
    # create equal-spaced grid within the smallest rectangle containing all intersections of the 8 sides
    grid_rect <- data.matrix(expand.grid(seq(min(-sqrt(2)*(e+h)/2, -a),
                                             max(sqrt(2)*(f+g)/2, b), length.out=4*size),
                                         seq(min(sqrt(2)*(g-e)/2, -d),
                                             max(sqrt(2)*(f+h)/2, c), length.out=4*size)))

    # create grid within the octagon by subsetting the points in the rectangle
    grid_oct <- grid_rect[which(-1*grid_rect[,1]<=a &
                                  1*grid_rect[,1]<=b &
                                  1*grid_rect[,2]<=c &
                                  -1*grid_rect[,2]<=d &
                                  -grid_rect[,1]/sqrt(2)-grid_rect[,2]/sqrt(2)<=e &
                                  grid_rect[,1]/sqrt(2)+grid_rect[,2]/sqrt(2)<=f &
                                  grid_rect[,1]/sqrt(2)-grid_rect[,2]/sqrt(2)<=g &
                                  -grid_rect[,1]/sqrt(2)+grid_rect[,2]/sqrt(2)<=h),]


    # subsetting oct_grid into points above/below the 45-degree line (could be empty set)
    grid_F_above <- grid_oct[which(grid_oct[,1]<=grid_oct[,2]), ]
    grid_F_below <- grid_oct[which(grid_oct[,1]>=grid_oct[,2]), ]

    # subsetting grid into points above/below the 45-degree line (could be empty set)
    grid_e_above <- grid_e[which(grid_e[,1]<=grid_e[,2]),]
    grid_e_below <- grid_e[which(grid_e[,1]>=grid_e[,2]),]

    size_above <- min(nrow(grid_e_above), nrow(grid_F_above), num_grid)
    size_below <- min(nrow(grid_e_below), nrow(grid_F_below), num_grid)

    if (size_above > 0){
      grid_e_above <- grid_e_above[sample(1:nrow(grid_e_above), size=size_above, replace = FALSE), ]
      grid_F_above <- grid_F_above[sample(1:nrow(grid_F_above), size=size_above, replace = FALSE), ]
    } else{
      grid_e_above <- NULL
      grid_F_above <- NULL
    }

    if (size_below > 0){
      grid_e_below <- grid_e_below[sample(1:nrow(grid_e_below), size=size_below, replace = FALSE), ]
      grid_F_below <- grid_F_below[sample(1:nrow(grid_F_below), size=size_below, replace = FALSE), ]
    } else{
      grid_e_below <- NULL
      grid_F_below <- NULL
    }

    if (h > 0 & g > 0){ # when the octagon intersects with the 45-degree line
      sample_F_cross <- seq(-sqrt(2)*e/2, sqrt(2)*f/2, length.out=num_grid)
      grid_F_cross <- cbind(sample_F_cross, sample_F_cross)
      grid_e_cross <- grid_e[sample(1:nrow(grid_e), size=nrow(grid_F_cross), replace = FALSE), ]
    } else{
      grid_F_cross <- NULL
      grid_e_cross <- NULL
    }

    return(list("cand_F_above"=grid_F_above,
                "cand_e_above"=grid_e_above,
                "cand_F_below"=grid_F_below,
                "cand_e_below"=grid_e_below,
                "cand_F_cross"=grid_F_cross,
                "cand_e_cross"=grid_e_cross))
  }
}


## Function that estimates F45
est_F45 <- function(Y, X, G,
                    est_nuisance=NULL, # pass the list returned by the `nuisance` function; if NULL, `nuisance` will be called to do the nuisance estimation.
                    optimizer=c("grid", "SGD", "BFGS"),
                    grid_size=20000, # size of the grid used to discretize the support of the function being optimized; used when optimizer="grid".
                    control=list(maxit=10000), # for optimizer="BFGS"
                    lr=0.08, maxit=20000, # for optimizer="SGD"
                    init_par=NULL, # for optimizer=c("SGD", "BFGS"); if NULL, will call `F45_init_par` to find initial values via grid search; if specified, should be one number.
                    l="classification",
                    method="lasso-logit",
                    fold=5){

  if (optimizer=="grid"){ ### grid search to estimate F_45
    warning(paste0("Grid search can be imprecise for estimating F_45. Try using `BFGS` or `SGD` (preferred)."))
    # discretize the support of the scalar c that index the support function of the intersection
    grid_c <- runif(grid_size, min = -1e3, max = 1e3)
    # the grid of directions to optimize for is u1-c[1, -1]'=[-1-c, c]' for c in grid_c
    grid_q <- data.matrix(cbind(-1-grid_c, grid_c))

    ## estimate the support functions for the directions in grid_q
    est_sf_E_val <- support_function(X=X, G=G, Y=Y,
                                     q=grid_q,
                                     est_nuisance=est_nuisance,
                                     l=l, method=method, fold=fold,
                                     return_score = !is.null(num_bstp_rep))

    est_sf_E <- est_sf_E_val$est_sf
    est_F45_coord <- -min(est_sf_E) # estimated coordinates of F_45
    q_F45 <- grid_q[which.min(est_sf_E), ]
  } else if (optimizer=="SGD" | optimizer=="BFGS"){ ### optimizer to estimate F_45
    if (is.null(est_nuisance)){
      pred_theta <- nuisance(Y=Y, X=X, G=G, l=l,
                             method=method, fold=fold)
    } else{
      pred_theta <- est_nuisance
    }

    diff_theta_r <- pred_theta[[1]]
    diff_theta_b <- pred_theta[[2]]
    est_mu_1 <- pred_theta[[3]]
    split_data_ind <- pred_theta[[4]]

    # set up the loss function
    if (is.character(l)){
      if (l=="classification"){
        loss <- function(d,y) return(as.numeric(d != y))
      } else if (l=="square-loss") {
        loss <- function(d,y) return((d-y)^2)
      }
    } else{ loss <- l }

    Y_ordered <- data.matrix(Y)[order(split_data_ind)]
    G_ordered <- data.matrix(G)[order(split_data_ind)]

    # labels L
    L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
    L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
    L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
    L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)

    diff_L_r <- L_11-L_01
    diff_L_b <- L_10-L_00

    if (is.null(init_par)){
      init_par <- F45_init_par(X=X, G=G, Y=Y,
                               est_nuisance=est_nuisance,
                               grid_size=grid_size,
                               l=l, method=method, fold=fold)
    }

    f <- function(c){ # objective function
      mean((-1-c)*L_01/est_mu_1+c*L_00/(1-est_mu_1)+
             ((-1-c)*diff_L_r/est_mu_1+c*diff_L_b/(1-est_mu_1))*
             as.numeric((-1-c)*diff_theta_r/est_mu_1+c*diff_theta_b/(1-est_mu_1)>0))
    }

    if (optimizer=="BFGS"){### optimizer is BFGS
      result <- optim(par=init_par, fn=f,
                      method="BFGS",
                      control=control)
      est_F45_coord <- -result$value
      q_F45 <- c(-1-result$par, result$par)
    } else { ### optimizer is SGD
      c <- torch_tensor(init_par, requires_grad=TRUE)
      optim <- optim_adam(params=list(c), lr=lr)
      for (iter in 1:maxit) {
        optim$zero_grad()
        l_fn <- f(c) # loss at current c
        l_fn$backward()
        optim$step()
      }
      est_F45_coord <- -l_fn$item()
      q_F45 <- c(-1-c$item(), c$item())
    }
  }

  return(list("F45"=est_F45_coord, "q_F45"=q_F45))
}


## Function that creates a grid of candidate values of e for constructing the confidence set for the FA-frontier
create_candidate_e_frontier <- function(Y, X, G,
                                        est_nuisance=NULL,
                                        num_grid=5000,
                                        buffer=sqrt(log(n))/sqrt(n),
                                        l="classification",
                                        method="lasso-logit",
                                        fold=5){
  # estimate the smallest octagon containing the feasible set
  oct <- support_function(X=X, G=G, Y=Y,
                          est_nuisance=est_nuisance,
                          q=matrix(c(-1, 0,
                                     1, 0,
                                     0, 1,
                                     0, -1,
                                     -1/sqrt(2), -1/sqrt(2),
                                     1/sqrt(2), 1/sqrt(2),
                                     1/sqrt(2), -1/sqrt(2),
                                     -1/sqrt(2), 1/sqrt(2)), byrow=TRUE, ncol=2),
                          l=l, method=method, fold=fold)

  # sides of this octagon, enlarged by buffer
  a <- oct$est_sf[1]+buffer
  b <- oct$est_sf[2]+buffer
  c <- oct$est_sf[3]+buffer
  d <- oct$est_sf[4]+buffer
  e <- oct$est_sf[5]+buffer
  f <- oct$est_sf[6]+buffer
  g <- oct$est_sf[7]+buffer
  h <- oct$est_sf[8]+buffer

  size <- round(sqrt(num_grid))

  # create equal-spaced grid within the smallest rectangle containing all intersections of the 8 sides
  grid_rect <- data.matrix(expand.grid(seq(min(-sqrt(2)*(e+h)/2, -a),
                                           max(sqrt(2)*(f+g)/2, b), length.out=4*size),
                                       seq(min(sqrt(2)*(g-e)/2, -d),
                                           max(sqrt(2)*(f+h)/2, c), length.out=4*size)))

  # create grid within the octagon by subsetting the points in the rectangle
  grid_oct <- grid_rect[which(-1*grid_rect[,1]<=a &
                                1*grid_rect[,1]<=b &
                                1*grid_rect[,2]<=c &
                                -1*grid_rect[,2]<=d &
                                -grid_rect[,1]/sqrt(2)-grid_rect[,2]/sqrt(2)<=e &
                                grid_rect[,1]/sqrt(2)+grid_rect[,2]/sqrt(2)<=f &
                                grid_rect[,1]/sqrt(2)-grid_rect[,2]/sqrt(2)<=g &
                                -grid_rect[,1]/sqrt(2)+grid_rect[,2]/sqrt(2)<=h &
                                (c-b+g*sqrt(2))*grid_rect[,1]-(c-b-h*sqrt(2))*grid_rect[,2] <= b*h*sqrt(2)+g*c*sqrt(2)-2*g*h &
                                (-1*grid_rect[,1]>=a-2*buffer | 1*grid_rect[,1]>=b-2*buffer | 1*grid_rect[,2]>=c-2*buffer | -1*grid_rect[,2]>=d-2*buffer | -grid_rect[,1]/sqrt(2)-grid_rect[,2]/sqrt(2)>=e-2*buffer | grid_rect[,1]/sqrt(2)+grid_rect[,2]/sqrt(2)>=f-2*buffer | grid_rect[,1]/sqrt(2)-grid_rect[,2]/sqrt(2)>=g-2*buffer | -grid_rect[,1]/sqrt(2)+grid_rect[,2]/sqrt(2)>=h-2*buffer)), ]

  grid_e <- grid_oct[sample(1:nrow(grid_oct), size=num_grid, replace = FALSE), ]

  return(grid_e)
}

## Function that constructs confidence set for the FA-frontier
CS_FAfrontier <- function(Y, X, G,
                          e1, e2, # candidate values of e to check
                          est_nuisance=NULL, # pass the list returned by the `nuisance` function; if NULL, `nuisance` will be called to do the nuisance estimation.
                          num_bstp_rep=NULL, # number of bootstrap replications; if NULL, then just return the test statistic without bootstrapped critical values
                          optimizer=c("grid", "SGD", "BFGS"), # optimization method
                          grid_size=1000, # size of the grid used to discretize the range of directions
                          lr=0.08, maxit=20000, # for optimizer="SGD"
                          control=list(maxit=10000), # for optimizer="BFGS"
                          init_par=NULL, # for optimizer=c("SGD", "BFGS"); if NULL, will call `CS_FAfrontier_init_par` to find initial values via grid search; if specified, need to be a matrix of dimension nrow=2 and ncol=length(e1), where the first (second) row corresponds to the optimization problem for testing feasibility (LDA) and each column corresponds to a candidate parameter value.
                          kink=FALSE, # whether the feasible set has kinks; if TRUE, then the numerical approximation from Fang & Santos (2019) is used for the directional derivative Default is FALSE.
                          alpha=0.05, # significance level; not used if num_bstp_rep=NULL
                          infntsml_adj=1e-6, # infinitesimal adjustment factor; not used if num_bstp_rep=NULL
                          l="classification",
                          method="lasso-logit",
                          fold=5){
  Y <- data.matrix(Y)
  X <- data.matrix(X)
  G <- data.matrix(G)
  n <- length(Y)   # sample size
  num_param <- length(e1) # number of parameter values to test

  if (length(e2)!=num_param){
    stop(paste0("Candidate vectors e1 and e2 should have the same length."))
  }

  # estimate nuisance parameters
  if (is.null(est_nuisance)){
    est_nuisance <- nuisance(Y=Y, X=X, G=G, l=l,
                             method=method, fold=fold)
  }

  diff_theta_r <- est_nuisance[[1]]
  diff_theta_b <- est_nuisance[[2]]
  est_mu_1 <- est_nuisance[[3]]
  split_data_ind <- est_nuisance[[4]]

  Y_ordered <- data.matrix(Y)[order(split_data_ind)]
  G_ordered <- data.matrix(G)[order(split_data_ind)]

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      loss <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      loss <- function(d,y) return((d-y)^2)
    }
  } else {
    loss <- l
  }

  # labels L
  L_00 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==0)
  L_10 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==0)
  L_01 <- as.numeric(loss(0,Y_ordered))*as.numeric(G_ordered==1)
  L_11 <- as.numeric(loss(1,Y_ordered))*as.numeric(G_ordered==1)

  diff_L_r <- L_11-L_01
  diff_L_b <- L_10-L_00

  if (optimizer == "grid"){
    # the discretized set of directions to search over
    grid_q_LDA <- t(as.matrix(sapply(seq(3*pi/4, 7*pi/4, length.out=grid_size),
                                     function(rad){c(cos(rad), sin(rad))})))
    grid_q_feasible <- t(as.matrix(sapply(seq(0, 2*pi, length.out=grid_size),
                                          function(rad){c(cos(rad), sin(rad))})))

    ## estimate the support functions for the directions in grid_q
    # for the feasible set \mathcal{E} (with q)
    est_sf_E_val <- support_function(X=X, G=G, Y=Y,
                                     q=rbind(grid_q_LDA, grid_q_feasible),
                                     est_nuisance=est_nuisance)

    # estimated sf corresponding to the LDA directions
    est_sf_E_LDA <- est_sf_E_val$est_sf[1:grid_size]
    # estimated sf corresponding to the feasibility directions
    est_sf_E_feasible <- est_sf_E_val$est_sf[(grid_size+1):(2*grid_size)]

    # for the set of all FA-improvements C* (with -q)
    est_sf_Cstar <- matrix(NA, nrow=num_param, ncol=grid_size)
    # test statistic corresponding to the constraint that e is feasible
    test_stat_feasible <- c()  # should have length=num_param
    arg_max_q_feasible <- c()  # should have length=num_param
    max_sf_diff <- c()
    for (param in 1:num_param){
      cur_e_r <- e1[param]
      cur_e_b <- e2[param]
      cur_e_diff <- cur_e_r-cur_e_b # (e_r^*-e_b^*)
      # support function of C* for the current e, h_{C^*}(-q)
      est_sf_Cstar[param, ] <- apply(-grid_q_LDA, 1,
                                     function(q){max(2*q[2]*cur_e_diff+(q[1]-q[2])*max(2*cur_e_diff,0), 0)+q[1]*cur_e_r+q[2]*cur_e_b-q[1]*max(2*cur_e_diff,0)}) # length=grid_size

      # check feasibility of cur_e
      cur_arg_max_q <- which.max((grid_q_feasible[,1]*cur_e_r+grid_q_feasible[,2]*cur_e_b)-est_sf_E_feasible)
      arg_max_q_feasible <- append(arg_max_q_feasible, cur_arg_max_q)
      max_sf_diff <- append(max_sf_diff, (grid_q_feasible[cur_arg_max_q,1]*cur_e_r+grid_q_feasible[cur_arg_max_q,2]*cur_e_b)-est_sf_E_feasible[cur_arg_max_q])
      test_stat_feasible <- append(test_stat_feasible,
                                   sqrt(n)*max(max_sf_diff[param], 0))

    }

    # compute the LDA test statistic, one for each element of (e1, e2)
    # min (est_sf_Cstar+est_sf_E)
    min_sf_sum <- t(apply(est_sf_Cstar, 1, function(x){x+c(est_sf_E_LDA)})) %>% apply(1, FUN = min) # length=num_param
    arg_min_q_LDA <- t(apply(est_sf_Cstar, 1, function(x){x+c(est_sf_E_LDA)})) %>% apply(1, FUN = which.min) # length=num_param
    # max -(est_sf_Cstar+est_sf_E) = - min (est_sf_Cstar+est_sf_E), then apply -min{x, 0}
    test_stat_LDA <- sqrt(n)*(-pmin(-min_sf_sum, 0)) # length=num_param
    # final test statistic; length=num_param
    test_stat <- test_stat_feasible + test_stat_LDA
  } else { ### optimizer=c("SGD", "L-BFGS-B")
    # set up the initial parameter value
    if (is.null(init_par)){
      set_init_par <- CS_FAfrontier_init_par(X=X, G=G, Y=Y,
                                             est_nuisance=est_nuisance,
                                             e1=e1, e2=e2,
                                             grid_size=grid_size)
      init_par_feasible <- set_init_par$arg_max_rad_feasible
      init_par_LDA <- set_init_par$arg_min_rad_LDA
    } else{
      init_par_feasible <- init_par[1,]
      init_par_LDA <- init_par[2,]
    }

    min_sf_sum <- matrix(NA, nrow=1, ncol=num_param)
    arg_min_rad_LDA <- matrix(NA, nrow=1, ncol=num_param)
    test_stat_LDA <- matrix(NA, nrow=1, ncol=num_param)

    max_sf_diff <- matrix(NA, nrow=1, ncol=num_param)
    arg_max_rad_feasible <- matrix(NA, nrow=1, ncol=num_param)
    test_stat_feasible <- matrix(NA, nrow=1, ncol=num_param)

    for (param in 1:num_param){
      # current parameter value being tested
      cur_e_r <- e1[param]
      cur_e_b <- e2[param]

      # LDA test statistic (before taking the min/inf) as a function of q for the current e
      param_obj_LDA <- function(rad){
        q1 <- cos(rad)
        q2 <- sin(rad)

        # orthogonal score of the support function of E
        cur_score <- q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+(q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0)

        # estimated support function of E at q
        est_sf_E <- mean(cur_score)

        # e^*
        cur_e_diff <- cur_e_r-cur_e_b # (e_r^*-e_b^*)

        # support function of C* at -q
        neg_q1 <- -q1
        neg_q2 <- -q2
        est_sf_Cstar <- max(2*neg_q2*cur_e_diff+(neg_q1-neg_q2)*max(2*cur_e_diff,0), 0)+neg_q1*cur_e_r+neg_q2*cur_e_b-neg_q1*max(2*cur_e_diff,0) # h_{C^*}(-q)

        return(est_sf_Cstar+est_sf_E)
      }

      # feasibility test statistic
      param_obj_feasible <- function(rad){
        q1 <- cos(rad)
        q2 <- sin(rad)

        # h_E(q) - q'\tilde{R} >= 0
        mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))-q1*cur_e_r-q2*cur_e_b
      }

      if (optimizer=="SGD"){
        # solving min_q (est_sf_Cstar+est_sf_E)
        rad_LDA <- torch_tensor(init_par_LDA[param], requires_grad=TRUE)
        optim <- optim_adam(params=list(rad_LDA), lr=lr)
        for (iter in 1:maxit) {
          optim$zero_grad()
          l_fn_LDA <- param_obj_LDA(rad_LDA) # loss at current rad
          l_fn_LDA$backward()
          optim$step()
          # constrain the range of rad_LDA
          rad_LDA$data()$clamp_(3*pi/4, 7*pi/4)
        }

        min_sf_sum[param] <- l_fn_LDA$item()
        arg_min_rad_LDA[param] <- rad_LDA$item()

        # solving min_q (h_E(q)-q'\tilde{R})
        rad_feasible <- torch_tensor(init_par_feasible[param], requires_grad=TRUE)
        optim <- optim_adam(params=list(rad_feasible), lr=lr)
        for (iter in 1:maxit) {
          # min_q (h_E(q)-q'e)
          optim$zero_grad()
          l_fn_feasible <- param_obj_feasible(rad_feasible) # loss at current rad
          l_fn_feasible$backward()
          optim$step()
        }

        # max (q'e-h_E(q)) = - min_q (h_E(q)-q'e)
        max_sf_diff[param] <- -l_fn_feasible$item()
        arg_max_rad_feasible[param] <- rad_feasible$item()
      } else if (optimizer=="BFGS"){
        # feasibility: min_q (h_E(q)-q'e)
        result_feasible <- optim(par=init_par_feasible[param],
                                 fn=param_obj_feasible,
                                 method="L-BFGS-B",
                                 lower=0, upper=2*pi,
                                 control=control)

        # max (q'e-h_E(q)) = - min_q (h_E(q)-q'e)
        max_sf_diff[param] <- -result_feasible$value
        arg_max_rad_feasible[param] <- result_feasible$par

        result_LDA <- optim(par=init_par_LDA[param],
                            fn=param_obj_LDA,
                            method="L-BFGS-B",
                            lower=3*pi/4, upper=7*pi/4,
                            control=control)

        min_sf_sum[param] <- result_LDA$value
        arg_min_rad_LDA[param] <- result_LDA$par
      }

      # max -(est_sf_Cstar+est_sf_E) = - min (est_sf_Cstar+est_sf_E), then apply -min{x, 0}
      test_stat_LDA[param] <- sqrt(n)*(-min(-min_sf_sum[param], 0))
      test_stat_feasible[param] <- sqrt(n)*(max(max_sf_diff[param], 0))
    }

    test_stat <- test_stat_feasible + test_stat_LDA
  }

  # initialize vector to collect bootstrapped crit val; not used if num_bstp_rep=NULL
  cv <- c()

  if (!is.null(num_bstp_rep)){
    # initialize matrix to store bootstrapped test statistic
    BS_test_stat <- matrix(NA, nrow=num_param, ncol=num_bstp_rep) # depends on which parameter we test

    # bootstrap once for all param
    ### ------- BOOTSTRAP STARTS -------
    for (b in 1:num_bstp_rep){
      # draw n exponential(1) weights
      b_W <- rexp(n, rate=1)
      # bootstrapped mu_1
      b_mu_1 <- mean(b_W*G_ordered)/mean(b_W)

      if (optimizer=="grid"){
        if (kink){
          # bootstrapped support function
          b_est_sf_feasible <- apply(grid_q_feasible, 1,
                                     function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                              (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                              as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)}) # length=grid_size

          b_est_sf_LDA <- apply(grid_q_LDA, 1,
                                function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                         (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                         as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)}) # length=grid_size

          # approximate the directional derivative as per Fang & Santos (2017)
          s_n <- sqrt(n)^{-1/2+0.01} # step size
          # direction at which we evaluate the directional derivative of \phi
          b_direc_grid_sf_feasible <- s_n*sqrt(n)*(b_est_sf_feasible-est_sf_E_feasible) # length=grid_size
          b_direc_grid_sf_LDA <- s_n*sqrt(n)*(b_est_sf_LDA-est_sf_E_LDA) # length=grid_size

          # evaluation point for \phi
          b_eval_grid_sf_feasible <- est_sf_E_feasible+b_direc_grid_sf_feasible
          b_eval_grid_sf_LDA <- est_sf_E_LDA+b_direc_grid_sf_LDA

          # \phi depends on the parameter at which we test
          for (param in 1:num_param){
            # current parameter value being tested
            cur_e_r <- e1[param]
            cur_e_b <- e2[param]

            # feasibility constraint associated with the current parameter
            cur_constr_feasible <- max_sf_diff[param] # max_q (q'e - h(q))
            # LDA constraint associated with the current parameter
            cur_constr_LDA <- -min_sf_sum[param] # max -(est_sf_Cstar+est_sf_E)

            # apply the \phi transformation to the evaluation points
            b_cur_constr_feasible <- max((grid_q_feasible[,1]*cur_e_r+grid_q_feasible[,2]*cur_e_b)-b_eval_grid_sf_feasible)
            b_cur_constr_LDA <- -min(est_sf_Cstar[param, ]+b_eval_grid_sf_LDA)

            BS_test_stat[param, b] <- (1/s_n)*(max(b_cur_constr_feasible-cur_constr_feasible, 0))+
              (1/s_n)*(-min(b_cur_constr_LDA-cur_constr_LDA, 0))
          }
        } else { # no kink
          b_est_sf_feasible <- apply(grid_q_feasible[arg_max_q_feasible, ], 1,
                                     function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                              (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                              as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)}) # length=num_param

          b_est_sf_LDA <- apply(grid_q_LDA[arg_min_q_LDA,], 1,
                                function(q) {mean(b_W*(q[1]*L_01/b_mu_1+q[2]*L_00/(1-b_mu_1)+
                                                         (q[1]*diff_L_r/b_mu_1+q[2]*diff_L_b/(1-b_mu_1))*
                                                         as.numeric(q[1]*diff_theta_r/b_mu_1+q[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)}) # length=num_param

          # feasibility constraint; length=num_param
          b_constr_feasible <- pmax(-sqrt(n)*(b_est_sf_feasible-est_sf_E_feasible[arg_max_q_feasible]), 0)
          # LDA constraint; length=num_param
          b_constr_LDA <- -pmin(-sqrt(n)*(b_est_sf_LDA-est_sf_E_LDA[arg_min_q_LDA]), 0)

          BS_test_stat[, b] <- b_constr_feasible+b_constr_LDA
        }
      } else{ # optimizer=c("SGD", "BFGS")
        if (kink){
          # approximate the directional derivative as per Fang & Santos (2017)
          s_n <- sqrt(n)^{-1/2+0.01} # step size
          for (param in 1:num_param){
            # current parameter value being tested
            cur_e_r <- e1[param]
            cur_e_b <- e2[param]

            b_obj_feasible <- function(rad) { # objective function
              q1 <- cos(rad)
              q2 <- sin(rad)

              # estimated support function
              est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
              # b-th bootstrapped support function
              b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                  (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                  as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)
              # direction at which we evaluate \phi
              b_cur_direc <- s_n*sqrt(n)*(b_sf-est_sf)
              # evaluation point
              est_sf+b_cur_direc-q1*cur_e_r-q2*cur_e_b
            }

            b_obj_LDA <- function(rad) {
              q1 <- cos(rad)
              q2 <- sin(rad)

              # estimated support function
              est_sf <- mean(q1*L_01/est_mu_1+q2*L_00/(1-est_mu_1)+
                               (q1*diff_L_r/est_mu_1+q2*diff_L_b/(1-est_mu_1))*
                               as.numeric(q1*diff_theta_r/est_mu_1+q2*diff_theta_b/(1-est_mu_1)>0))
              # b-th bootstrapped support function
              b_sf <- mean(b_W*(q1*L_01/b_mu_1+q2*L_00/(1-b_mu_1)+
                                  (q1*diff_L_r/b_mu_1+q2*diff_L_b/(1-b_mu_1))*
                                  as.numeric(q1*diff_theta_r/b_mu_1+q2*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

              # direction at which we evaluate the directional derivative of \phi
              cur_direc_sf_E <- s_n*sqrt(n)*(cur_btsrp_sf_E-est_sf_E)

              # evaluate \phi
              cur_eval_sf_E <- est_sf_E+cur_direc_sf_E
              cur_e_diff <- cur_e_r-cur_e_b
              neg_q1 <- -q1
              neg_q2 <- -q2
              cur_sf_Cstar <- max(2*neg_q2*cur_e_diff+(neg_q1-neg_q2)*max(2*cur_e_diff,0), 0)+neg_q1*cur_e_r+neg_q2*cur_e_b-neg_q1*max(2*cur_e_diff,0) # h_{C^*}(-q)

              return(cur_sf_Cstar+cur_eval_sf_E)
            }

            if (optimizer=="SGD"){
              # initialize argument
              rad_feasible <- torch_tensor(arg_max_rad_feasible[param], requires_grad=TRUE)
              rad_LDA <- torch_tensor(arg_min_rad_LDA[param], requires_grad=TRUE)
              # initialize optimization
              optim_feasible <- optim_adam(params=list(rad_feasible), lr=lr)
              optim_LDA <- optim_adam(params=list(rad_LDA), lr=lr)
              # optimize
              for (iter in 1:maxit) {
                # min_q h_E(q)-q'e
                optim_feasible$zero_grad()  # initialize gradient
                b_l_fn_feasible <- b_obj_feasible(rad_feasible) # loss at current rad
                b_l_fn_feasible$backward()    # backpropagation
                optim_feasible$step()         # descent
                # constrain the range of rad_LDA
                rad_LDA$data()$clamp_(3*pi/4, 7*pi/4)

                # min_q (h_{C*}(-q)+h_E(q))
                optim_LDA$zero_grad()  # initialize gradient
                b_l_fn_LDA <- b_obj_LDA(rad_LDA) # loss at current rad
                b_l_fn_LDA$backward()    # backpropagation
                optim_LDA$step()         # descent
              }

              # max_q (q'e - h_E(q)) = - min_q (h_E(q)-q'e)
              b_cur_constr_feasible <- -b_l_fn_feasible$item()
              # max_q - (h_{C*}(-q)+h_E(q)) = - min_q (h_{C*}(-q)+h_E(q))
              b_cur_constr_LDA <- -b_l_fn_LDA$item()
            } else if (optimizer=="BFGS"){
              optim_feasible <- optim(par=arg_max_rad_feasible[param],
                                      fn=b_obj_feasible,
                                      method="L-BFGS-B",
                                      lower=0, upper=2*pi,
                                      control=control)

              optim_LDA <- optim(par=arg_min_rad_LDA[param],
                                 fn=b_obj_LDA,
                                 method="L-BFGS-B",
                                 lower=3*pi/4, upper=7*pi/4,
                                 control=control)

              # max_q (q'\tilde{R} - h_E(q)) = - min_q (h_E(q)-q'\tilde{R})
              b_cur_constr_feasible <- -optim_feasible$value
              # max_q - (h_{C*}(-q)+h_E(q)) = - min_q (h_{C*}(-q)+h_E(q))
              b_cur_constr_LDA <- -optim_LDA$value
            }

            # feasibility constraint associated with the current parameter
            cur_constr_feasible <- max_sf_diff[param] # max_q (q'e - h(q))
            # LDA constraint associated with the current parameter
            cur_constr_LDA <- -min_sf_sum[param] # max -(est_sf_Cstar+est_sf)

            BS_test_stat[param, b] <- (1/s_n)*(max(b_cur_constr_feasible-cur_constr_feasible, 0))+
              (1/s_n)*(-min(b_cur_constr_LDA-cur_constr_LDA, 0))
          }
        } else{ ## no kink and e^* is nonstochastic
          for (param in 1:num_param){
            ## for the feasibility part ----
            # optimal q^* for feasibility
            opt_q_feasible <- matrix(c(cos(arg_max_rad_feasible[param]),
                                       sin(arg_max_rad_feasible[param])), ncol=2)

            # estimated support function at optimal q
            est_sf_opt_q_feasible <- mean(opt_q_feasible[1]*L_01/est_mu_1+opt_q_feasible[2]*L_00/(1-est_mu_1)+(opt_q_feasible[1]*diff_L_r/est_mu_1+opt_q_feasible[2]*diff_L_b/(1-est_mu_1))*as.numeric(opt_q_feasible[1]*diff_theta_r/est_mu_1+opt_q_feasible[2]*diff_theta_b/(1-est_mu_1)>0))

            # b-th bootstrapped support function at optimal q
            b_sf_opt_q_feasible <- mean(b_W*(opt_q_feasible[1]*L_01/b_mu_1+opt_q_feasible[2]*L_00/(1-b_mu_1)+(opt_q_feasible[1]*diff_L_r/b_mu_1+opt_q_feasible[2]*diff_L_b/(1-b_mu_1))*as.numeric(opt_q_feasible[1]*diff_theta_r/b_mu_1+opt_q_feasible[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

            ## for the LDA part ----
            # optimal q^* for LDA
            opt_q_LDA <- matrix(c(cos(arg_min_rad_LDA[param]),
                                  sin(arg_min_rad_LDA[param])), ncol=2)

            # estimated support function at optimal q
            est_sf_opt_q_LDA <- mean(opt_q_LDA[1]*L_01/est_mu_1+opt_q_LDA[2]*L_00/(1-est_mu_1)+(opt_q_LDA[1]*diff_L_r/est_mu_1+opt_q_LDA[2]*diff_L_b/(1-est_mu_1))*as.numeric(opt_q_LDA[1]*diff_theta_r/est_mu_1+opt_q_LDA[2]*diff_theta_b/(1-est_mu_1)>0))

            # b-th bootstrapped support function at optimal q
            b_sf_opt_q_LDA <- mean(b_W*(opt_q_LDA[1]*L_01/b_mu_1+opt_q_LDA[2]*L_00/(1-b_mu_1)+(opt_q_LDA[1]*diff_L_r/b_mu_1+opt_q_LDA[2]*diff_L_b/(1-b_mu_1))*as.numeric(opt_q_LDA[1]*diff_theta_r/b_mu_1+opt_q_LDA[2]*diff_theta_b/(1-b_mu_1)>0)))/mean(b_W)

            BS_test_stat[param, b] <- max(-sqrt(n)*(b_sf_opt_q_feasible-est_sf_opt_q_feasible), 0)-
              min(-sqrt(n)*(b_sf_opt_q_LDA-est_sf_opt_q_LDA),0)
          }
        }
      }
    }
    ### ------- BOOTSTRAP ENDS -------
    cv <- apply(BS_test_stat, 1, function(x) {quantile(x, 1-alpha+infntsml_adj)+infntsml_adj})
  }

  return(list("test_stat"=test_stat,
              "cv"=cv, # will return cv=c() if num_bstp_rep=NULL
              "rej"=as.numeric(test_stat>cv) # will return rej=numeric(0) if num_bstp_rep=NULL
  ))
}


## Function that finds initial values for the optimizers in `CS_FAfrontier` using grid search
CS_FAfrontier_init_par <- function(X, G, Y,
                                   est_nuisance=NULL,
                                   e1, e2,
                                   grid_size=1000,
                                   l="classification",
                                   method="lasso-logit",
                                   fold=5){
  num_param <- length(e1)

  # the discretized set of directions to search over
  grid_rad_feasible <- seq(0, 2*pi, length.out=grid_size)
  grid_rad_LDA <- seq(3*pi/4, 7*pi/4, length.out=grid_size)
  grid_q_feasible <- t(as.matrix(sapply(grid_rad_feasible,
                                        function(rad){c(cos(rad), sin(rad))})))
  grid_q_LDA <- t(as.matrix(sapply(grid_rad_LDA,
                                   function(rad){c(cos(rad), sin(rad))})))

  ## estimate the support functions for the directions in grid_q
  # for the feasible set \mathcal{E} (with q)
  est_sf_E_val <- support_function(X=X, G=G, Y=Y,
                                   q=rbind(grid_q_LDA, grid_q_feasible),
                                   est_nuisance=est_nuisance,
                                   l=l, method=method, fold=fold)
  est_sf_E_LDA <- est_sf_E_val$est_sf[1:grid_size]
  est_sf_E_feasible <- est_sf_E_val$est_sf[(grid_size+1):(2*grid_size)]

  # for the set of all FA-improvements C* (with -q)
  est_sf_Cstar <- matrix(NA, nrow=num_param, ncol=grid_size)
  # test statistic corresponding to the constraint that e is feasible
  arg_max_q_feasible <- c()  # should have length=num_param
  for (param in 1:num_param){
    cur_e_r <- e1[param]
    cur_e_b <- e2[param]
    cur_e_diff <- cur_e_r-cur_e_b # (e_r^*-e_b^*)
    # support function of C* for the current e, h_{C^*}(-q)
    est_sf_Cstar[param, ] <- apply(-grid_q_LDA, 1,
                                   function(q){max(2*q[2]*cur_e_diff+(q[1]-q[2])*max(2*cur_e_diff,0), 0)+q[1]*cur_e_r+q[2]*cur_e_b-q[1]*max(2*cur_e_diff,0)}) # length=grid_size

    # check feasibility constraint for e
    cur_arg_max_q <- which.max((grid_q_feasible[,1]*cur_e_r+grid_q_feasible[,2]*cur_e_b)-est_sf_E_feasible)
    arg_max_q_feasible <- append(arg_max_q_feasible, cur_arg_max_q)
  }

  # compute the LDA test statistic, one for each element of (e1, e2)
  # min (est_sf_Cstar+est_sf_E)
  arg_min_q_LDA <- t(apply(est_sf_Cstar, 1, function(x){x+c(est_sf_E_LDA)})) %>% apply(1, FUN = which.min) # length=num_param

  arg_max_rad_feasible <- grid_rad_feasible[arg_max_q_feasible]
  arg_min_rad_LDA <- grid_rad_LDA[arg_min_q_LDA]

  return(list("arg_max_rad_feasible"=arg_max_rad_feasible,
              "arg_min_rad_LDA"=arg_min_rad_LDA))
}


### `alg_on_FAfrontier` is a function that splits the data into a training sample used to learn the nuisance parameters and a prediction sample used for evaluation; it returns the predictions of the trained nuisance and indices used for sample splitting.
alg_on_FAfrontier <- function(X, Y, G,
                              frac_train=0.5, # fraction of the sample used for learning the nuisance parameter
                              l="classification",
                              method="lasso-logit",
                              seed=NULL, num.trees=10000, # for the grf package
                              parallel=TRUE # for multithreading in glmnet
){
  Y <- data.matrix(Y)
  X <- data.matrix(X)
  G <- data.matrix(G)

  # sample size
  n <- length(Y)

  # set up the loss function
  if (is.character(l)){
    if (l=="classification"){
      l <- function(d,y) return(as.numeric(d != y))
    } else if (l=="square-loss") {
      l <- function(d,y) return((d-y)^2)
    }
    else{
      stop(paste0("Argument l is not in c('classification', 'square-loss') or an object of class 'function'."))
    }
  } else{
    if (!inherits(l,"function")){
      stop(paste0("Argument l is not in c('classification', 'square-loss') or an object of class 'function'."))
    }
    if (length(formals(l))!=2){
      stop(cat(paste0("The number of arguments in the loss function should be 2. For example, for classification loss the function l should be defined as: \n l <- function(d,y) return(as.numeric(d != y))\n Check your definition of l.")))
    }
  }

  ### split the data into ~frac_train used to learn the nuisance parameter used to build the new algorithm (uses G), and ~(1-frac_train) used for prediction and evaluation (does not use G);
  alg_split_ind <- sample(1:2, size=n, replace=TRUE,
                          prob=c(frac_train, 1-frac_train))

  # alg_split_ind==1 is used for training
  Y_train <- Y[alg_split_ind==1]
  X_train <- X[alg_split_ind==1, ]
  G_train <- G[alg_split_ind==1]

  # alg_split_ind==2 is used for evaluation (only X is used for prediction)
  X_eval <- X[alg_split_ind==2, ]

  # learn the nuisance parameter using the training data
  est_mu_1 <- mean(G_train) # estimated mu_1
  for (g in c(1, 0)){
    # create a data frame for regression
    label <- (l(1,Y_train)-l(0,Y_train))*as.numeric(G_train==g) # the effective training label
    datause_dg <- data.frame(X_train)
    datause_dg$label <- label

    group_name <- ifelse(g==1, "r", "b")

    # set up the machine learner
    if (is.character(method)){
      if (method=="grf"){
        if (is.null(seed)){
          seed <- 1
          warning(paste0("Argument `seed` is not provided for `regression_forest`; a random seed=", seed, " has been set for reproducibility on the same platform (grf version >= 2.4.0 required)."))
        }
        assign(paste0("est_diff_theta_", group_name),
               regression_forest(X_train, label,
                                 num.trees=num.trees,
                                 tune.parameters=c("sample.fraction",
                                                   "mtry",
                                                   "min.node.size",
                                                   "honesty.fraction",
                                                   "honesty.prune.leaves",
                                                   "alpha",
                                                   "imbalance.penalty"),
                                 seed=seed))
      } else if (method=="nnet"){
        assign(paste0("est_diff_theta_", group_name),
               nnet(label ~.,
                    data=datause_dg,
                    size=8,
                    maxit=1000,
                    decay=0.01,
                    MaxNWts=10000,
                    trace=FALSE))
      } else if (method=="lasso-logit"){
        assign(paste0("est_diff_theta_", group_name),
               cv.glmnet(x=X_train,
                         y=factor(label, levels = c("-1", "0", "1")),
                         family="multinomial", alpha=1,
                         intercept=TRUE, parallel=parallel))
      } else if (method=="lasso"){
        assign(paste0("est_diff_theta_", group_name),
               cv.glmnet(x=X_train,
                         y=label,
                         family="gaussian", alpha=1,
                         intercept=TRUE, parallel=parallel))
      } else{
        stop(paste0("Argument 'method' should be a string in c('grf', 'nnet', 'lasso', 'lasso-logit')."))
      }
    }
    else{
      stop(paste0("Argument 'method' should be a string in c('grf', 'nnet', 'lasso', 'lasso-logit')."))
    }
  }

  # evaluate using the evaluation sample
  if (method=="lasso") {
    pred_diff_theta_r <- predict(est_diff_theta_r, newx=X_eval, s="lambda.min")
    pred_diff_theta_b <- predict(est_diff_theta_b, newx=X_eval, s="lambda.min")

  } else if (method=="lasso-logit"){
    pred_probs_r <- predict(est_diff_theta_r, newx=X_eval,
                            s="lambda.min", type="response")[ , , 1]
    pred_diff_theta_r <- -1*pred_probs_r[, 1]+1*pred_probs_r[, 3]

    pred_probs_b <- predict(est_diff_theta_b, newx=X_eval,
                            s="lambda.min", type="response")[ , , 1]
    pred_diff_theta_b <- -1*pred_probs_b[, 1]+1*pred_probs_b[, 3]
  } else{
    pred_diff_theta_r <- as.numeric(unlist(predict(est_diff_theta_r, X_eval)))
    pred_diff_theta_b <- as.numeric(unlist(predict(est_diff_theta_b, X_eval)))
  }


  return(list("pred_diff_theta_r"=pred_diff_theta_r,
              "pred_diff_theta_b"=pred_diff_theta_b,
              "est_mu_1"=est_mu_1,
              "alg_split_ind"=alg_split_ind))

}


## Auxiliary function that finds the intersection between a convex polygon and the 45-degree line crossing the origin
find_intersections <- function(conv_hull) {
  n_row <- nrow(conv_hull)

  # consecutive pairs of (x1, y1) and (x2, y2)
  x1 <- conv_hull$e_r[1:(n_row-1)]
  y1 <- conv_hull$e_b[1:(n_row-1)]
  x2 <- conv_hull$e_r[2:n_row]
  y2 <- conv_hull$e_b[2:n_row]

  a <- (y2-y1)/(x2-x1) # slope a
  b <- y1-a*x1 # intercept b

  # solve for the intersection with y = x
  cross <- abs(1-a) > 1e-10 # ignore near parallel cases where slope is close to 1
  x_intersect <- b[cross]/(1-a[cross]) # y = ax + b & y = x
  y_intersect <- x_intersect

  # check if intersection points are within segment bounds
  within_bounds <- (x_intersect>=pmin(x1[cross], x2[cross])) &
    (x_intersect<=pmax(x1[cross], x2[cross])) &
    (y_intersect>=pmin(y1[cross], y2[cross])) &
    (y_intersect<=pmax(y1[cross], y2[cross]))

  return(data.frame(e_r=x_intersect[within_bounds],
                    e_b=y_intersect[within_bounds]))
}
