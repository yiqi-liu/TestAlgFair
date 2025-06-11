#' Compute percentiles
#'
#' @param DF data.table of interest
#' @param col.to.cut the column with values to be ranked by percentiles
#' @param col.to.groupby columns to group by
#'
#' @return a column of percentile values
#' @export
#'
#' @import data.table
#'
#' @examples
#' MyComputePercentile(dt, 'risk_score', NULL)
#'
MyComputePercentile <- function(DF, col.to.cut, col.to.groupby){
    DF[, percentile := cut(get(col.to.cut), 
                           unique(quantile(get(col.to.cut), probs=0:100/100)),
                           include.lowest=TRUE, labels=FALSE), by = eval(get('col.to.groupby'))]
    return(DF)
}

#' Compute quantiles
#'
#' @param DF data.table of interest
#' @param nquantiles number of quantiles to be computed 
#' @param col.to.cut the column with values to be ranked by percentiles
#' @param col.to.groupby columns to group by
#'
#' @return a column of quantile values
#' @export
#'
#' @import data.table
#'
#' @examples
#' MyComputeQuantile(dt, 10,'risk_score', NULL)
#'
MyComputeQuantile<- function(DF, nquantiles, col.to.cut, col.to.groupby){

    DF[, quantile := as.numeric(cut(get(col.to.cut), unique(quantile(get(col.to.cut), probs=0:nquantiles/nquantiles)), include.lowest=TRUE, labels=FALSE)), by = eval(get('col.to.groupby'))] 
    # scale quantile values for plotting purposes
    DF[, quantile := quantile*100/nquantiles, by = eval(get('col.to.groupby'))]
    
   # manually move quantiles markes to middle of the quantiles for proper x axis placement
    #interv <- 100 / (nquantiles * 2)
    #DF[, quantile := quantile - interv, by = eval(get('col.to.groupby'))]

    return(DF)
}


#' Summarize col.to.y by mean 
#'
#' @param DF data.table of interest
#' @param col.to.y the column with values to be summarized
#' @param col.to.groupby columns to group by
#'
#' @return a column where values are summarized by groups
#' @export
#'
#' @import data.table
#'
#' @examples
#' MyComputeMean(dt, 'risk_score', 'race')
#'
MyComputeMean<- function(DF, col.to.y, col.to.groupby){

    # compute mean
    col.to.mean <- paste0('col_to_mean_by_', str_c(get('col.to.groupby'), collapse = '_by_'))

    DF[, (col.to.mean) := mean(get(col.to.y), na.rm = T), by = eval(get('col.to.groupby'))]

    return(DF)
}



#' Compute SE
#'
#' @param DF data.table of interest
#' @param col.to.y the column with values to be summarized
#' @param col.to.groupby columns to group by
#'
#' @return a column where values are summarized by groups
#' @export
#'
#' @import data.table
#'
#' @examples
#' MyComputeSE(dt, 'risk_score', 'race')
#'
MyComputeSE<- function(DF, col.to.y, col.to.groupby){

    # Compute SE
    DF[, ci_se := sd(get(col.to.y), na.rm = T)/sqrt(.N), by= eval(get('col.to.groupby'))]
    
    return(DF)
}


#' Compute PlotDF to be plot mean Y by X percentile-quantile by Z groups
#'
#' @param DF data.table of interest
#' @param col.to.y column name of y
#' @param col.to.cut column name of x to percentiles and quantiles
#' @param col.to.groupby  colnumn name to group by
#' @param nquantiles the number of quantiles 
#' @param ci.level the level of confidence (provided as a decimal < 1)
#'
#' @return a data.table with all the necessary ingredients for plotting Y by X percentile-quantile by Z groups
#' @export
#'
#' @import data.table
#'
#' @examples
#' x <- MyComputePlotDF(dt, col.to.y = 'cost', col.to.cut = 'risk_score', col.to.groupby = 'race', nquantiles = 10, ci.level = 0.95)
MyComputePlotDF <- function(DF,
                            col.to.y = 'cost',
                            col.to.cut = 'risk_score',
                            col.to.groupby = 'race', 
                            nquantiles = 10,
                            ci.level = 0.95, ...){
    DFRaw <- DF
    subset.cols <- c(col.to.y, col.to.cut, col.to.groupby)
    DF <-  copy(DFRaw[, ..subset.cols])
    DF[, (col.to.groupby) := as.factor(get(col.to.groupby))]
    if(col.to.cut == 'gagne_sum'){
        # very specific to racial bias
        DF[, gagne_sum_bin := gagne_sum]
        DF[gagne_sum >= 9, gagne_sum_bin := 9]
        # Compute mean values of y by x by z
        MyComputeMean(DF, get('col.to.y'), 
                         col.to.groupby = c('gagne_sum_bin', get('col.to.groupby')))
        MyComputeMean(DF, get('col.to.y'), 
                         col.to.groupby = c('gagne_sum_bin', get('col.to.groupby')))
        ci.to.groupby <- 'gagne_sum_bin'
    }else{

        # Compute percentiles
        MyComputePercentile(DF, 
                            col.to.cut = get('col.to.cut'), 
                            col.to.groupby = NULL)
        # Compute n quantiles 
        MyComputeQuantile(DF, nquantiles = get('nquantiles'), 
                          col.to.cut = get('col.to.cut'), 
                          col.to.groupby = NULL)
        # Compute mean values of y by x by z
        MyComputeMean(DF, get('col.to.y'), col.to.groupby = c('percentile', get('col.to.groupby')))
        MyComputeMean(DF, get('col.to.y'), c('quantile', get('col.to.groupby')))

        ci.to.groupby <- 'quantile'
    }

    # calculate se
    MyComputeSE(DF, get('col.to.y'), c(ci.to.groupby, get('col.to.groupby')))

    return(DF)
}
