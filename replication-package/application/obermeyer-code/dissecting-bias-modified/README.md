# Public Release Synthetic Data and Code

*9/1/2022: Please note an error in the code for generating Figure 1b, kindly pointed out by Sam Corbett-Davies. More details are below and, the error is also flagged and corrected in the relevant file.  

## Project Repository Structure

This repository contains data and code needed to reproduce the main results for our [paper]() Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations.

1. *data*: A synthetic master dataset that closely mirrors the dataset used to produce our original results (which cannot be shared to protect patient privacy).
2. *code*: Code in R and Python that can be used to replicate the figures and tables from the main manuscript.
3. *results*: Our own replication of these results using the synthetic dataset.

## Synthetic Dataset Creation

Beginning with our final analytic files for the study, we used the [synthpop](https://cran.r-project.org/web/packages/synthpop/index.html) R package to create a synthetic version of the key variables needed to replicate all analyses in the paper.

The synthetic dataset contains the same number of observations (patient-years) as our original dataset. The package uses classification and regression trees to sequentially generate variables that captures the moments and covariances of the original dataset. More details can be found at https://cran.r-project.org/web/packages/synthpop/synthpop.pdf.

We include a [data dictionary](./data/data_dictionary.md) describing each of the individual variables.


## Preparation

- Set working directory to `dissecting-bias`
    - `cd dissecting-bias`
- R / python environment setup:
    1. Create our [conda](https://www.anaconda.com/distribution/) environment:
        - `conda env create -f bias.yml`
    1. Activate environment:
        - `conda activate bias`
    1. Install our utility package:
        - `R CMD INSTALL plot0_0.1.tar.gz`

## Execution
Remember to activate our conda environment prior to executing:
```
conda activate bias
```

- To replicate figures:
    - `Rscript code/figure1/figure1.R`
        - 'figure1.R' imports results from `figure1b.R`, but there's no need to
          run them separately to produce figure1.
        - If you would like to run `figure1b.R` separately and generated
        `figure1b.csv`, run `Rscript code/figure1/figure1b.R`.
    - `Rscript code/figure2.R`
    - `Rscript code/figure3.R`
- To train model and save predictions on the holdout:
    - `python code/model/main.py`
- To replicate tables:
    - `python code/table1.py`
    - `python code/table2.py`
        - `table3.R` imports results from `table2.py`, please make sure to run
          `table2.py` prior to running `table3.R`
    - `Rscript code/table3.R`

Let us know if there are any issues/questions! Zoey Li (@lizeyu), Katie Lin (@kl2532), Ziad Obermeyer (@ziadoo)

## Error in code that makes Figure 1b
The error happens in the procedure for swapping healthier White patients in the fast-track group, with sicker Black patients not in the fast-track group. The original code removes the healthiest fast-tracked White patient from the simulated counterfactual fast-track table, if she is healthier than a non-fast-tracked Black patients (old line 67, new line 68); then *also* increments the number indexing the next patient to be considered for the counterfactual fast-track (old line 69, new line 71). This skips the next-healthiest White patient in the (factual) fast-track, never comparing her to a Black patient, and thus ensuring she remains in the counterfactual fast-track, irrespective of her health. Due to the error, we underestimate the number of less-healthy Black patients who should be switched in: the fraction of Black patients in the unbiased counterfactual fast-track should be 59%, rather than 47% as originally published. So by this metric, the algorithm looks more biased than we thought. We are embarrassed by the error, and so grateful to Sam Corbett-Davies for pointing this out. 

It's worth also noting a broader point Sam makes about this `swapping' procedure. To quote his very thoughtful email: "white patients can only be removed from the enrollees and black patients can only be added, even though there are enrolled patients with very few (or zero) chronic medical conditions of both races. Thus, after the procedure ends you could flip the race labels, run the procedure again, and it would continue to find favorable matches." At the time of the original analysis, we were working closely with our health system partner, and focused on a pragmatic solution that would make the smallest change to the fast-track population. But we agree that Sam's thoughtful critique suggests that there is a lot of progress to be made on these swapping exercises. 

