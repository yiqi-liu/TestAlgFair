#' Install packages if not present
#'
#' @param package_names a list of package to be installed
#'
#' @return install the list of packages provided.
#' @export
#'
#' @examples
#' data(max_package_list)
#' MyInstall(max_package_list)
#'
MyInstall <- function(package_names) {
  sapply(package_names, function(x) if(!x %in% installed.packages()) 
    suppressMessages(install.packages(x,repos="http://cran.cnr.berkeley.edu/")))
} 


#' Load necessary packages
#'
#' @param package_names a list of packages to be loaded
#'
#' @return load the list of packages provided.
#' @export
#'
#' @examples
#' data(min_package_list)
#' MyLoad(min_package_list)
#'
MyLoad <- function(package_names) {  
    packages_loaded <- lapply(package_names, function(x) suppressMessages(library(x,character.only=TRUE, logical.return = FALSE)))
  packages_loaded[[length(package_names)]]
} 


#' Check basic summary statistics of a data frame
#'
#' @param DF a data frame of interest
#'
#' @return a list that contains the number of rows, the number of columns, a data.table summarizing the number of unique values, the fraction of NAs, the class of each column, a data.table summarizing the mean, sd, min, max, fraction of NAs, number of unique values of each numeric columns.
#' @export
#'
#' @import data.table
#' @import dplyr
#'
#' @examples
#' MyCheckSummary(dt)
#'
MyCheckSummary <- function(DF){
    
    # all column summary -- unc (Unique, NA, Class)
    n.unique = sapply(DF, function(e) length(unique(e)))
    frac.NA = sapply(DF, function(e) sum(is.na(e))/length(e))
    class.DF = sapply(DF, class)
    unc.summ = data.table::data.table(data.frame(n.unique, frac.NA, class.DF), 
                          keep.rownames = TRUE)
    
    # numeric column summary
    DF.num <- dplyr::select_if(DF, is.numeric)
    num.mean <- sapply(DF.num,  mean, na.rm = TRUE)
    num.sd <- sapply(DF.num, sd, na.rm = TRUE)
    num.min <- sapply(DF.num, min, na.rm = TRUE)
    num.max <- sapply(DF.num, max, na.rm = TRUE)
    num.NA <- sapply(DF.num, function(e) sum(is.na(e))/length(e))
    num.unique <- sapply(DF.num, function(e) length(unique(e)))
    num.summ <- data.table::data.table(data.frame(mean = num.mean, 
                                                  sd = num.sd, 
                                                  min = num.min, 
                                                  max = num.max, 
                                                  n.unique = num.unique,
                                                  frac.NA = num.NA), keep.rownames = TRUE)
    return(list(n.row = nrow(DF), 
                n.col = ncol(DF),
                unc.summary = unc.summ,
                numeric.summary = num.summ))
}


#' Print and sink output result to a text file
#'
#' @param output_res an output to be printed and sinked
#' @param path path to file
#' @param filename name of the file
#' @param width width of the text file to be sinked
#'
#' @return print output result to a text file and export it to the path provided
#' @export
#'
#' @examples
#' MySink(dt, "path/to/file", "filename", width = "200")
#'
MySink <- function(output_res, path, filename, width = '200'){
    # set print column width
    options(width=width)
    sink(paste0(path, filename))
    print(output_res)
    sink()
    # reset back to default width
    options(width = '88')
}

#' Generalize reading data and sourcing scripts
#'
#' @param FUN a function to read data 
#' @param to.data.table whether to convert output result to a data.table
#' @param path path to the data 
#'
#' @return a function with path preprogrammed in
#' @export
#'
#' @examples
#' pathit(dt, "path/to/file", "filename", width = "200")
#'
pathit <- function(FUN, to.data.table = FALSE, path){

    if(to.data.table){
        function(file, path, ...){
            as.data.table(FUN(file.path(path, file), ...))
        }
    }else{
        function(file, path, ...){
            FUN(file.path(path, file), ...)
        }
    }
}

#' Read RDS files with pre-specified path
#'
#' @param readRDS the function that reads RDS files
#' @param to.data.table TRUE
#' @param path path to the data 
#'
#' @return a data.table
#' @export
#'
#' @examples
#' external_path <- '/external/SFA/PHMRES/research-fellow/Powers,\ Brian/Jan\ 2017\ Merge/racial_bias_final/data/cohort/new/'
#' MyReadRDS('cohort_claims_biomarkers_bw.Rds', path = external_path)
#'
MyReadRDS <- pathit(readRDS, to.data.table = TRUE, path = path)


#' Source files with pre-specified path
#'
#' @param source the function that source .R files
#' @param to.data.table FALSE
#' @param path path to the data 
#'
#' @return objects from the sourced script
#' @export
#'
#' @examples
#' support_path <- '/data/zolab/racial_bias_final/code/support_code/'
#' MySource('exhibit_helper.R', path = support_path)
#'
MySource <- pathit(source, to.data.table = FALSE, path = path)


#' Read feather files with pre-specified path
#'
#' @param read_feather the function that reads feather files
#' @param to.data.table TRUE
#' @param path path to the data 
#'
#' @import feather
#'
#' @return a data.table 
#' @export
#'
#' @examples
#' feather_path <- '/data/zolab/racial_bias_final/data/master/new/'
#' enc_dt <- MyReadFeather('edw_enc_master_new_cohort.feather', path = feather_path)
#' 
MyReadFeather <- pathit(feather::read_feather, to.data.table = TRUE, path = path)


#' Read text files with pre-specified path
#'
#' @param fread the function that reads text files
#' @param to.data.table TRUE
#' @param path path to the data 
#' 
#' @import data.table
#'
#' @return a data.table
#' @export
#'
MyFread <- pathit(data.table::fread, to.data.table = TRUE, path = path)


#' Read csv text files with pre-specified path
#'
#' @param read.csv the function that reads csv text files
#' @param to.data.table TRUE
#' @param path path to the data 
#' 
#' @import data.table
#'
#' @return a data.table 
#' @export
#'
MyReadCSV <- pathit(read.csv, to.data.table = TRUE, path = path)


#' Create a directory if not already exists
#'
#' @param path a string of the path to be created
#' 
#' @return create a directory at the path provided
#' @export
#'
MyCreateDir <- function(path='') {
  if(!dir.exists(path)) { dir.create(path) }
}


#' Get variable name as a string 
#'
#' @param var a variable 
#' 
#' @return name of the variable as a string
#' @export
#'
MyGetVarName <- function(var){
    return(deparse(substitute(var)))
}
