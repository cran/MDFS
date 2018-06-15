#' Max information gains
#'
#' @param data input data where columns are variables and rows are observations (all numeric)
#' @param decision decision variable as a binary sequence of length equal to number of observations
#' @param dimensions number of dimensions (a positive integer; on CUDA limited to 2--5 range)
#' @param divisions number of divisions (from 1 to 15; additionally limited by dimensions if using CUDA; \code{NULL} selects probable optimal number)
#' @param discretizations number of discretizations
#' @param seed seed for PRNG used during discretizations (\code{NULL} for random)
#' @param range discretization range (from 0.0 to 1.0; \code{NULL} selects probable optimal number)
#' @param pseudo.count pseudo count
#' @param return.tuples whether to return tuples where max IG was observed (one tuple per variable) - not supported with CUDA and in 1D
#' @param interesting.vars limit examined tuples to only those containing all specified variables (default: do not limit) - not supported with CUDA
#' @param use.CUDA whether to use CUDA acceleration (must be compiled with CUDA)
#' @return A \code{\link{data.frame}} with the following columns:
#'  \itemize{
#'    \item \code{IG} -- max information gain (of each variable)
#'    \item \code{Tuple.1, Tuple.2, ...} -- corresponding tuple (up to \code{dimensions} columns, available only when \code{return.tuples == T} and \code{dimensions > 1})
#'  }
#'
#'  Additionally attribute named \code{run.params} with run parameters is set on the result.
#' @examples
#' \donttest{
#' ComputeMaxInfoGains(madelon$data, madelon$decision, dimensions = 2, divisions = 1,
#'                     range = 0, seed = 0)
#' }
#' @importFrom stats runif
#' @export
#' @useDynLib MDFS r_compute_max_ig
ComputeMaxInfoGains <- function(
    data,
    decision,
    dimensions = 1,
    divisions = NULL,
    discretizations = 1,
    seed = NULL,
    range = NULL,
    pseudo.count = 0.25,
    return.tuples = FALSE,
    interesting.vars = c(),
    use.CUDA = FALSE) {
  data <- data.matrix(data)
  storage.mode(data) <- "double"
  decision <- as.vector(decision, mode="integer")

  if (length(decision) != nrow(data)) {
    stop('Length of decision is not equal to the number of rows in data.')
  }

  # try to rebase decision from 1 to 0 (as is the case with e.g. factors)
  if (!any(decision == 0)) {
    decision <- decision - 1
  }

  if (!all(decision == 0 | decision == 1)) {
    stop('Decision must be binary.')
  }

  if (all(decision == 0) || all(decision == 1)) {
    stop('Both classes have to be represented.')
  }

  if (as.integer(dimensions) != dimensions || dimensions < 1) {
    stop('Dimensions has to be a positive integer.')
  }

  if (is.null(divisions)) {
    if (dimensions == 1) {
      divisions <- as.integer(
        min(
          max(
            floor(min(sum(decision==0),sum(decision==1))^(1/2/dimensions)),
            1),
          15)
      )
    } else {
      divisions <- 1
    }
  } else if (as.integer(divisions) != divisions || divisions < 1 || divisions > 15) {
    stop('Divisions has to be an integer between 1 and 15 (inclusive).')
  }

  if (pseudo.count <= 0) {
    stop('Pseudo count has to be strictly greater than 0.')
  }

  if (as.integer(discretizations) != discretizations || discretizations < 1) {
    stop('Discretizations has to be a positive integer.')
  }

  if (is.null(range)) {
    range <- max(min(2*(sqrt(((min(sum(decision==0),sum(decision==1))/5)^(2/dimensions)/(1+divisions)^2-1)^2+((min(sum(decision==0),sum(decision==1))/5)^(2/dimensions)/(1+divisions)^2-1))-((min(sum(decision==0),sum(decision==1))/5)^(2/dimensions)/(1+divisions)^2-1)),
                     1),
                 0)
  } else if (as.double(range) != range || range < 0 || range > 1) {
    stop('Range has to be a number between 0.0 and 1.0')
  }

  if (range == 0 && discretizations > 1) {
    stop('Zero range does not make sense with more than one discretization. All will always be equal.')
  }

  if (is.null(seed)) {
    seed <- round(runif(1, 0, 2^31-1)) # unsigned passed as signed, the highest bit remains unused for best compatibility
  } else if (as.integer(seed) != seed || seed < 0 || seed > 2^31-1) {
    warning('Only integer seeds from 0 to 2^31-1 are portable. Using non-portable seed may make result harder to reproduce.')
  }

  if (dimensions == 1 && return.tuples) {
    stop('return.tuples does not make sense in 1D')
  }

  if (use.CUDA) {
    if (dimensions == 1) {
      stop('CUDA acceleration does not support 1 dimension')
    }

    if (dimensions > 5) {
      stop('CUDA acceleration does not support more than 5 dimensions')
    }

    if ((divisions+1)^dimensions > 256) {
      stop('CUDA acceleration does not support more than 256 cubes = (divisions+1)^dimensions')
    }

    if (return.tuples) {
      stop('CUDA acceleration does not support return.tuples parameter (for now)')
    }

    if (length(interesting.vars) > 0) {
      stop('CUDA acceleration does not support interesting.vars parameter (for now)')
    }
  }

  out <- .Call(
      r_compute_max_ig,
      data,
      decision,
      as.integer(dimensions),
      as.integer(divisions),
      as.integer(discretizations),
      as.integer(seed),
      as.double(range),
      as.double(pseudo.count),
      as.integer(interesting.vars - 1),  # send C-compatible 0-based indices
      as.logical(return.tuples),
      as.logical(use.CUDA))

  if (return.tuples) {
    names(out) <- c("IG", "Tuple")
    out$Tuple = t(out$Tuple + 1) # restore R-compatible 1-based indices, transpose to remain compatible with ComputeInterestingTuples
  } else {
    names(out) <- c("IG")
  }

  out <- as.data.frame(out)

  attr(out, 'run.params') <- list(
    dimensions      = dimensions,
    divisions       = divisions,
    discretizations = discretizations,
    seed            = seed,
    range           = range,
    pseudo.count    = pseudo.count)

  return(out)
}

#' Interesting tuples
#'
#' @param data input data where columns are variables and rows are observations (all numeric)
#' @param decision decision variable as a binary sequence of length equal to number of observations
#' @param dimensions number of dimensions (an integer greater than or equal to 2)
#' @param divisions number of divisions (from 1 to 15; \code{NULL} selects probable optimal number)
#' @param discretizations number of discretizations
#' @param seed seed for PRNG used during discretizations (\code{NULL} for random)
#' @param range discretization range (from 0.0 to 1.0; \code{NULL} selects probable optimal number)
#' @param pseudo.count pseudo count
#' @param ig.thr IG threshold above which the tuple is interesting
#' @param interesting.vars variables for which to check the IGs (none = all)
#' @return A \code{\link{data.frame}} or \code{\link{NULL}} (following a warning) if no tuples are found.
#'
#'  The following columns are present in the \code{\link{data.frame}}:
#'  \itemize{
#'    \item \code{Var} -- interesting variable index
#'    \item \code{Tuple.1, Tuple.2, ...} -- corresponding tuple (up to \code{dimensions} columns)
#'    \item \code{IG} -- information gain achieved by \code{var} in \code{Tuple.*}
#'  }
#'
#'  Additionally attribute named \code{run.params} with run parameters is set on the result.
#' @examples
#' \donttest{
#' ComputeInterestingTuples(madelon$data, madelon$decision, dimensions = 2, divisions = 1,
#'                          range = 0, seed = 0, ig.thr = 100)
#' }
#' @export
#' @useDynLib MDFS r_compute_all_matching_tuples
ComputeInterestingTuples <- function(
    data,
    decision,
    dimensions = 2,
    divisions = NULL,
    discretizations = 1,
    seed = NULL,
    range = NULL,
    pseudo.count = 0.25,
    ig.thr,
    interesting.vars = c()) {
  data <- data.matrix(data)
  storage.mode(data) <- "double"
  decision <- as.vector(decision, mode="integer")

  if (length(decision) != nrow(data)) {
    stop('Length of decision is not equal to the number of rows in data.')
  }

  # try to rebase decision from 1 to 0 (as is the case with e.g. factors)
  if (!any(decision == 0)) {
    decision <- decision - 1
  }

  if (!all(decision == 0 | decision == 1)) {
    stop('Decision must be binary.')
  }

  if (all(decision == 0) || all(decision == 1)) {
    stop('Both classes have to be represented.')
  }

  if (as.integer(dimensions) != dimensions || dimensions < 1) {
    stop('Dimensions has to be a positive integer.')
  }

  if (is.null(divisions)) {
    if (dimensions == 1) {
      divisions <- as.integer(
        min(
          max(
            floor(min(sum(decision==0),sum(decision==1))^(1/2/dimensions)),
            1),
          15)
      )
    } else {
      divisions <- 1
    }
  } else if (as.integer(divisions) != divisions || divisions < 1 || divisions > 15) {
    stop('Divisions has to be an integer between 1 and 15 (inclusive).')
  }

  if (pseudo.count <= 0) {
    stop('Pseudo count has to be strictly greater than 0.')
  }

  if (as.integer(discretizations) != discretizations || discretizations < 1) {
    stop('Discretizations has to be a positive integer.')
  }

  if (is.null(range)) {
    range <- max(min(2*(sqrt(((min(sum(decision==0),sum(decision==1))/5)^(2/dimensions)/(1+divisions)^2-1)^2+((min(sum(decision==0),sum(decision==1))/5)^(2/dimensions)/(1+divisions)^2-1))-((min(sum(decision==0),sum(decision==1))/5)^(2/dimensions)/(1+divisions)^2-1)),
                     1),
                 0)
  } else if (as.double(range) != range || range < 0 || range > 1) {
    stop('Range has to be a number between 0.0 and 1.0')
  }

  if (range == 0 && discretizations > 1) {
    stop('Zero range does not make sense with more than one discretization. All will always be equal.')
  }

  if (is.null(seed)) {
    seed <- round(runif(1, 0, 2^31-1)) # unsigned passed as signed, the highest bit remains unused for best compatibility
  } else if (as.integer(seed) != seed || seed < 0 || seed > 2^31-1) {
    warning('Only integer seeds from 0 to 2^31-1 are portable. Using non-portable seed may make result harder to reproduce.')
  }

  if (dimensions < 2) {
    stop('Dimensions has to be at least 2 for this function to make any sense.')
  }

  result <- .Call(
      r_compute_all_matching_tuples,
      data,
      decision,
      as.integer(dimensions),
      as.integer(divisions),
      as.integer(discretizations),
      as.integer(seed),
      as.double(range),
      as.double(pseudo.count),
      as.integer(interesting.vars - 1),  # send C-compatible 0-based indices
      as.double(ig.thr))

  if (length(result[[1]]) == 0) {
    warning("No tuples were returned.")
    return(NULL)
  }

  names(result) <- c("Var", "Tuple", "IG")

  result$Var = result$Var + 1 # restore R-compatible 1-based indices
  result$Tuple = result$Tuple + 1 # restore R-compatible 1-based indices

  result <- as.data.frame(result)

  attr(result, 'run.params') <- list(
    dimensions      = dimensions,
    divisions       = divisions,
    discretizations = discretizations,
    seed            = seed,
    range           = range,
    pseudo.count    = pseudo.count)

  return(result)
}
