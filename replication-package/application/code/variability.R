############################################
# Knit .Rmd file to .html file in directory
#    ../application/results/rep-results
############################################

rmarkdown::render("../results/rep-results/variability.Rmd", 
                  output_dir = "../results/rep-results/")
