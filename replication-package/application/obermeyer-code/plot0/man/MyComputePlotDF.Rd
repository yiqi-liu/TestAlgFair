% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot0_compute.R
\name{MyComputePlotDF}
\alias{MyComputePlotDF}
\title{Compute PlotDF to be plot mean Y by X percentile-quantile by Z groups}
\usage{
MyComputePlotDF(DF, col.to.y = "cost", col.to.cut = "risk_score",
  col.to.groupby = "race", nquantiles = 10, ci.level = 0.95, ...)
}
\arguments{
\item{DF}{data.table of interest}

\item{col.to.y}{column name of y}

\item{col.to.cut}{column name of x to percentiles and quantiles}

\item{col.to.groupby}{colnumn name to group by}

\item{nquantiles}{the number of quantiles}

\item{ci.level}{the level of confidence (provided as a decimal < 1)}
}
\value{
a data.table with all the necessary ingredients for plotting Y by X percentile-quantile by Z groups
}
\description{
Compute PlotDF to be plot mean Y by X percentile-quantile by Z groups
}
\examples{
x <- MyComputePlotDF(dt, col.to.y = 'cost', col.to.cut = 'risk_score', col.to.groupby = 'race', nquantiles = 10, ci.level = 0.95)
}
