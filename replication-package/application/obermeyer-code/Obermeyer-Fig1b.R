### This file adapts the code in figure1b.R (https://gitlab.com/labsysmed/dissecting-bias/-/blob/master/code/figure1) to avoid installing `plot0_0.1.tar.gz` and to replicate Figure 1b of Obermeyer et al. (2019)
library(data.table)

#### MODIFYING figure1b.R -------------
setup <- function(default_in_percentile = c(95, 97)) {
  # load
  # filepath <- paste0(getwd(), '/data')
  # filename <- 'data_new.csv'
  # cohort <- MyFread(filename, filepath)
  cohort <- as.data.table(fread("../../obermeyer-code/dissecting-bias-modified/data/data_new.csv")) # MODIFIED BY Y. LIU: CALLING `as.data.table` DIRECTLT TO AVOID CALLING `MyFread`

  cohort <- cohort[, c('race', 'risk_score_t', 'gagne_sum_t')]
  dt <- cohort
  dt[, risk_pctile := cut(risk_score_t, unique(quantile(risk_score_t, probs=0:100/100)), include.lowest=TRUE, labels=FALSE), ]

  # enrollment stats: black and white enrollment and their ratio
  enroll_stats <- matrix(nrow = length(default_in_percentile), ncol = 3)
  rownames(enroll_stats) <- default_in_percentile
  colnames(enroll_stats) <- c('black_before', 'black_after', 'ratio')

  return(list(dt = dt,
              enroll_stats = enroll_stats))
}

### Y. LIU: THIS FUNCTION IS UNCHANGED
exercise <- function(default_in_percentile){
  dt <- setup(default_in_percentile)$dt
  enroll_stats <- setup(default_in_percentile)$enroll_stats

  for(j in seq_along(default_in_percentile)){
    # enrolled
    prior_enrolled <- dt[risk_pctile >= default_in_percentile[j]]

    prior_w <- prior_enrolled[race == 'white']
    prior_b <- prior_enrolled[race == 'black']
    # prep
    upperb <- dt[risk_pctile >= default_in_percentile[j] & race == 'black']
    upperw <- dt[risk_pctile >= default_in_percentile[j] & race == 'white']
    lowerb <- dt[risk_pctile < default_in_percentile[j] & race == 'black']

    # rank
    upperw <- upperw[order(gagne_sum_t), ]
    lowerb <- lowerb[order(-risk_score_t, -gagne_sum_t), ]

    # tracking comparisons
    sw <- 1
    sb <- 1
    switched_count <- 0
    switched_w <- NULL
    switched_b <- NULL
    while( sw < nrow(upperw)  & sb < nrow(lowerb)){
      if(upperw[sw, gagne_sum_t] < lowerb[sb, gagne_sum_t]){
        # keep track of marginal switched
        switched_w <- rbind(switched_w, upperw[sw,]) %>% as.data.table
        switched_b <- rbind(switched_b, lowerb[sb,]) %>% as.data.table
        # update enrolled blacks
        upperb <- rbind(upperb, lowerb[sb]) %>% as.data.table
        # update enrolled whites
        upperw <- upperw[-sw,]
        upperw <- upperw[order(gagne_sum_t),]
        ##### 9/1/2022: commenting out this step fixes the error - swapped White patient is removed from upperw in line 67...
        # sw = sw + 1
        ##### ...so there is no need to increment sw counter (ever)
        sb = sb + 1
        switched_count = switched_count + 1
      }else{
        sb = sb + 1
        sw = sw
        switched_count = switched_count
      }
    }
    # calculate means
    sampw <- prior_w
    sampb <- prior_b

    black_before <- nrow(prior_b)/(nrow(prior_w) + nrow(prior_b))
    black_after <- (nrow(prior_b) + switched_count)/(nrow(prior_w) + nrow(prior_b))

    ratio <- black_after / black_before
    enroll_stats[j,] <- c(black_before, black_after, ratio)
  }
  return(enroll_stats = enroll_stats)
}
