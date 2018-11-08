#' Run end-to-end MDFS
#'
#' @details
#' In case of FDR control it is recommended to use Benjamini-Hochberg-Yekutieli p-value adjustment
#' method (\code{"BY"} in \code{\link[stats]{p.adjust}}) due to unknown dependencies between tests.
#'
#' @param data input data where columns are variables and rows are observations (all numeric)
#' @param decision decision variable as a boolean vector of length equal to number of observations
#' @param n.contrast number of constrast variables (defaults to max of 1/10 of variables number and 30)
#' @param dimensions number of dimensions (a positive integer; on CUDA limited to 2--5 range)
#' @param divisions number of divisions (from 1 to 15; \code{NULL} selects probable optimal number)
#' @param discretizations number of discretizations
#' @param range discretization range (from 0.0 to 1.0; \code{NULL} selects probable optimal number)
#' @param pseudo.count pseudo count
#' @param p.adjust.method method as accepted by \code{\link[stats]{p.adjust}} (\code{"BY"} is recommended for FDR, see Details)
#' @param level statistical significance level
#' @param seed seed for PRNG used during discretizations (\code{NULL} for random)
#' @param use.CUDA whether to use CUDA acceleration (must be compiled with CUDA)
#' @return A \code{\link{list}} with the following fields:
#'  \itemize{
#'    \item \code{contrast.indices} -- indices of variables chosen to build contrast variables
#'    \item \code{contrast.variables} -- built contrast variables
#'    \item \code{MIG.Result} -- result of ComputeMaxInfoGains
#'    \item \code{MDFS} -- result of ComputePValue (the MDFS object)
#'    \item \code{statistic} -- vector of statistic's values (IGs) for corresponding variables
#'    \item \code{p.value} -- vector of p-values for corresponding variables
#'    \item \code{adjusted.p.value} -- vector of adjusted p-values for corresponding variables
#'    \item \code{relevant.variables} -- vector of relevant variables indices
#'  }
#' @examples
#' \donttest{
#' MDFS(madelon$data, madelon$decision, dimensions = 2, divisions = 1,
#'      range = 0, seed = 0)
#' }
#' @importFrom stats p.adjust
#' @export
MDFS <- function(
  data,
  decision,
  n.contrast = max(ncol(data)/10, 30),
  dimensions = 1,
  divisions = NULL,
  discretizations = 1,
  range = NULL,
  pseudo.count = 0.25,
  p.adjust.method = "holm",
  level = 0.05,
  seed = NULL,
  use.CUDA = FALSE
 ) {
 if(!is.null(seed)) {set.seed(seed)}
 if (n.contrast>0) {
  contrast<-AddContrastVariables(data, n.contrast)
  contrast.indices<-contrast$indices
  contrast.variables<-contrast$x[,contrast$mask]
  data.contrast<-contrast$x
  contrast.mask<-contrast$mask
 } else {
  contrast.mask<-contrast.indices<-contrast.variables<-NULL
  data.contrast<-data
 }

 MIG.Result <- ComputeMaxInfoGains(data.contrast, decision,
  dimensions = dimensions, divisions = divisions,
  discretizations = discretizations, range = range, pseudo.count = pseudo.count,
  seed = seed, return.tuples = !use.CUDA && dimensions > 1, use.CUDA = use.CUDA)

 divisions <- attr(MIG.Result, "run.params")$divisions

 fs <- ComputePValue(MIG.Result$IG,
  dimensions = dimensions, divisions = divisions,
  contrast.mask = contrast.mask,
  one.dim.mode = ifelse (discretizations==1, "raw", ifelse(divisions*discretizations<12, "lin", "exp")))

 statistic <- if(is.null(contrast.mask)) { MIG.Result$IG } else { MIG.Result$IG[!contrast.mask] }
 p.value <- if(is.null(contrast.mask)) { fs$p.value } else { fs$p.value[!contrast.mask] }
 adjusted.p.value <- p.adjust(p.value, method=p.adjust.method)
 relevant.variables <- which(adjusted.p.value<level)

 return(list(
  contrast.indices = contrast.indices,
  contrast.variables = contrast.variables,
  MIG.Result = MIG.Result,
  MDFS = fs,
  statistic = statistic,
  p.value = p.value,
  adjusted.p.value = adjusted.p.value,
  relevant.variables = relevant.variables
  )
 )
}
