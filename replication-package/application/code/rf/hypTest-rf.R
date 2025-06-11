############################################
# Knit .Rmd file to .html file in directory
# /application/results/hypTest-results/rf
############################################

rmarkdown::render("../../results/hypTest-results/rf/hypTest-rf.Rmd", 
                  output_dir = "../../results/hypTest-results/rf/")