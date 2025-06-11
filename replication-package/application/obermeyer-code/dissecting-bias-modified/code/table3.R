################################################################################
# prep
################################################################################
rm(list=ls())
# load packages
library(plot0)
# set output path
res_dir <- paste0(getwd(), '/results/')

#################################################################################
# Table 3
#################################################################################

# load -------------------------------------------------------------------------
filepath <- paste0(getwd(), '/data')
filename <- 'data_new.csv'
dt <- MyFread(filename, filepath)

# enrollment rate by cost bin
MyComputePercentile(dt, 'risk_score_t', NULL)
df_enrollment_rate <- dt[, list(enrollment_rate = mean(program_enrolled_t)), by = percentile]

# load prediction file ----------------------------------------------------------
filepath <- res_dir
filename <- 'model_lasso_predictors.csv'
pred_df <- MyFread(filename, filepath) %>% as.data.table
pred_df[, race := ifelse(dem_race_black, 'black', 'white')]

# enrollment proprtions
k <- prop.table(table(pred_df$program_enrolled_t))['1']

# Compute for Table 3 -----------------------------------------------------------
report_stats <- function(){

    calculations<- function(df, gnum){

        #1. fraction black
        k1 <- prop.table(table(df$race))['black']
        #print(sprintf('1. fraction black = %s', round(k1, 4)))
        k1se <- sqrt( k1*(1 - k1) / nrow(pred_df))
        #print(sprintf('SE = %s', round(k1se, 4) ))

        #2. fraction cost
        k2 <- sum(df[,cost_t])/sum(pred_df[,cost_t])
        #print(sprintf('2. fraction cost = %s', round(k2, 4)))
        k2se <- sqrt( k2*(1 - k2)/nrow(pred_df))
        #print(sprintf('SE = %s', round(k2se, 4) ))

        #3 fraction gagne
        k3 <- sum(df[,gagne_sum_t])/sum(pred_df[,gagne_sum_t])
        #print(sprintf('3. fraction health = %s', round(k3, 4)))
        k3se <- sqrt( k3*(1 - k3) / nrow(pred_df))
        #print(sprintf('SE = %s', round(k3se, 4) ))

        group_stats <- data.frame(population = gnum,
                          frac_black = round(k1, 3),
                          frac_black_se = round(k1se, 3),
                          frac_cost = round(k2, 3),
                          frac_cost_se = round(k2se, 3),
                          frac_health = round(k3, 3),
                          frac_health_se = round(k3se, 3))
        return(group_stats)
    }

    # group 0: observed program enrollment
    df <- pred_df[program_enrolled_t == 1]
    print("Group 0: Observed program enrollment")
    g0 <- calculations(df = df, 'Observed program enrollment')

    # group 1: Random, in predicted cost bin
    print("Group 1: Random, in predicted cost bin")
    df_samp <- NULL
    set.seed(12345)
    for(i in 1:100){
        df_bin <- pred_df[risk_score_t_percentile==i]
        enrollment_rate <- df_enrollment_rate[percentile == i, enrollment_rate]
        samp_size <- floor(enrollment_rate * nrow(df_bin))
        samp_ind <- sample(seq_len(nrow(df_bin)), size = samp_size)
        df_samp <- rbind(df_samp, df_bin[samp_ind])
    }
    g1 <- calculations(df = df_samp, 'Random, in predicted cost bin')

    # group 2: Predicted health, in predicted cost bin
    print("Group 2: Predicted health, in predicted cost bin")
    df_samp <- NULL
    for(i in 1:100){
        df_bin <- pred_df[risk_score_t_percentile==i]
        enrollment_rate <- df_enrollment_rate[percentile == i, enrollment_rate]
        samp_size <- floor(enrollment_rate * nrow(df_bin))
        if(samp_size == 0){
            res <- NULL
        }else{
            res <- df_bin[order(-gagne_sum_t_hat)] %>%
                .[1:samp_size]
        }
        df_samp <- rbind(df_samp, res)
    }
    g2 <- calculations(df = df_samp, 'Predicted health, in predicted-cost bin')

    # group 3: Highest predicted cost
    print("Group 3: Highest predicted cost")
    samp_size <- round(k * nrow(pred_df))
    df <- pred_df[order(-log_cost_t_hat)] %>%
                .[1:samp_size]
    g3 <- calculations(df = df, 'Highest predicted cost')

    # group 4: Worst predicted health
    print("Group 4: Worst predicted health")
    samp_size <- round(k * nrow(pred_df))
    df <- pred_df[order(-gagne_sum_t_hat)] %>%
                .[1:samp_size]
    g4 <- calculations(df = df, 'Worst predicted health')

    tab3 <- rbind(g0, g1, g2, g3, g4)

    return(tab3)
}

tab3 <- report_stats()
write_csv(tab3, paste0(res_dir, 'table3.csv'))
print(paste('saved table3.csv at', res_dir))
