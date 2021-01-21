#' Add contrast variables to data
#'
#' @param data data organized in matrix with separate variables in columns
#' @param n.contrast number of constrast variables (defaults to max of 1/10 of variables number and 30)
#' @return A list with the following key names:
#'  \itemize{
#'   \item \code{indices} -- vector of indices of input variables used to construct contrast variables
#'   \item \code{x} -- data with constrast variables appended to it
#'   \item \code{mask} -- vector of booleans making it easy to select just contrast variables
#'  }
#' @examples
#' AddContrastVariables(madelon$data)
#' @export
AddContrastVariables <- function(
    data,
    n.contrast = max(ncol(data)/10, 30)) {
  if (is.null(ncol(data))) {
    stop('Data has to have columns')
  }
  indices <- sample.int(ncol(data), n.contrast, replace = n.contrast>ncol(data))
  x.contrast <- apply(data[, indices], 2, sample)
  mask <- c(rep.int(F, ncol(data)), rep.int(T, ncol(x.contrast)))
  return(list(
      indices = indices,
      x = cbind(data, x.contrast),
      mask = mask))
}
