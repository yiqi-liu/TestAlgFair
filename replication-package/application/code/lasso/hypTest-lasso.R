############################################
# Knit .Rmd file to .html file in directory
# /application/results/hypTest-results/lasso
############################################

rmarkdown::render("../../results/hypTest-results/lasso/hypTest-lasso.Rmd", 
                  output_dir = "../../results/hypTest-results/lasso/")