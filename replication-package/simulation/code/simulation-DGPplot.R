##############################
# Knit .Rmd file to .html file
##############################

rmarkdown::render("../results/simulation-DGPplot.Rmd", 
                  output_dir = "../results/")