% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/p_value.R
\name{ComputePValue}
\alias{ComputePValue}
\title{Compute p-values from information gains and return MDFS}
\usage{
ComputePValue(
  IG,
  dimensions,
  divisions,
  response.divisions = 1,
  df = NULL,
  contrast.mask = NULL,
  ig.in.bits = TRUE,
  ig.doubled = FALSE,
  one.dim.mode = "exp",
  irr.vars.num = NULL,
  ign.low.ig.vars.num = NULL,
  min.irr.vars.num = NULL,
  max.ign.low.ig.vars.num = NULL,
  search.points = 8,
  level = 0.05
)
}
\arguments{
\item{IG}{max conditional information gains}

\item{dimensions}{number of dimensions}

\item{divisions}{number of divisions}

\item{response.divisions}{number of response divisions (i.e. categories-1)}

\item{df}{vector of degrees of freedom for each variable (optional)}

\item{contrast.mask}{boolean mask on \code{IG} specifying which variables are contrast variables (or \code{NULL} if none, otherwise at least 3 variables must be marked)}

\item{ig.in.bits}{\code{TRUE} if input is in binary log (as opposed to natural log)}

\item{ig.doubled}{\code{TRUE} if input is doubled (to follow the chi-squared distribution)}

\item{one.dim.mode}{\code{'exp'} for exponential distribution, \code{'lin'} for linear function of chi-squared or \code{'raw'} for raw chi-squared}

\item{irr.vars.num}{if not NULL, number of irrelevant variables, specified by the user}

\item{ign.low.ig.vars.num}{if not NULL, number of ignored low IG variables, specified by the user}

\item{min.irr.vars.num}{minimum number of irrelevant variables (\code{NULL} selects probable optimal number)}

\item{max.ign.low.ig.vars.num}{maximum number of ignored low IG variables (\code{NULL} selects probable optimal number)}

\item{search.points}{number of points in search procedure for the optimal number of ignored variables}

\item{level}{acceptable error level of goodness-of-fit one-sample Kolmogorov-Smirnov test (used only for warning)}
}
\value{
A \code{\link{data.frame}} with class set to \code{MDFS}. Can be coerced back to \code{data.frame} using \code{\link{as.data.frame}}.

 The following columns are present:
 \itemize{
   \item \code{IG} -- information gains (input copy)
   \item \code{chi.squared.p.value} -- chi-squared p-values
   \item \code{p.value} -- theoretical p-values
 }

 Additionally the following \code{\link{attributes}} are set:
 \itemize{
  \item \code{run.params} -- run parameters
  \item \code{sq.dev} -- vector of square deviations used to estimate the number of irrelevant variables
  \item \code{dist.param} -- distribution parameter
  \item \code{err.param} -- squared error of the distribution parameter
  \item \code{fit.p.value} -- p-value of fit
 }
}
\description{
Compute p-values from information gains and return MDFS
}
\examples{
ComputePValue(madelon$IG.2D, dimensions = 2, divisions = 1)
}
