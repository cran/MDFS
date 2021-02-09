GetRange <- function (k = 5, n, dimensions, divisions) {
  ksi <- (k/n)^(1/dimensions)
  suggested.range <- (1 - ksi*(1+divisions)) / (1 - ksi*(1-divisions))
  range <- max(0, min(suggested.range, 1))
  reasonable.range <- 0.25
  if (range == 0) {
    warning('Too small sample for the test')
  } else if (range < reasonable.range) {
    warning('Too small sample for multiple discretizations')
  }
  range
}

#' Max information gains
#'
#' @param data input data where columns are variables and rows are observations (all numeric)
#' @param decision decision variable as a binary sequence of length equal to number of observations
#' @param dimensions number of dimensions (a positive integer; 5 max)
#' @param divisions number of divisions (from 1 to 15; additionally limited by dimensions if using CUDA; \code{NULL} selects probable optimal number)
#' @param discretizations number of discretizations
#' @param seed seed for PRNG used during discretizations (\code{NULL} for random)
#' @param range discretization range (from 0.0 to 1.0; \code{NULL} selects probable optimal number)
#' @param pc.xi parameter xi used to compute pseudocounts (the default is recommended not to be changed)
#' @param return.tuples whether to return tuples (and relevant discretization number) where max IG was observed (one tuple and relevant discretization number per variable) - not supported with CUDA nor in 1D
#' @param return.min whether to return min instead of max (per tuple, always max per discretization) - not supported with CUDA
#' @param interesting.vars variables for which to check the IGs (none = all) - not supported with CUDA
#' @param require.all.vars boolean whether to require tuple to consist of only interesting.vars
#' @param use.CUDA whether to use CUDA acceleration (must be compiled with CUDA)
#' @return A \code{\link{data.frame}} with the following columns:
#'  \itemize{
#'    \item \code{IG} -- max information gain (of each variable)
#'    \item \code{Tuple.1, Tuple.2, ...} -- corresponding tuple (up to \code{dimensions} columns, available only when \code{return.tuples == T})
#'    \item \code{Discretization.nr} -- corresponding discretization number (available only when \code{return.tuples == T})
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
    pc.xi = 0.25,
    return.tuples = FALSE,
    return.min = FALSE,
    interesting.vars = vector(mode = "integer"),
    require.all.vars = FALSE,
    use.CUDA = FALSE) {
  data <- data.matrix(data)
  storage.mode(data) <- "double"
  decision <- as.vector(decision, mode="integer")

  if (length(decision) != nrow(data)) {
    stop('Length of decision is not equal to the number of rows in data.')
  }

  # try to rebase decision from 1 to 0 (as is the case with e.g. factors)
  if (!any(decision == 0)) {
    decision <- decision - as.integer(1) # as.integer is crucial in keeping this an integer vector
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

  if (dimensions > 5) {
    stop('Dimensions cannot exceed 5')
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

  if (as.integer(discretizations) != discretizations || discretizations < 1) {
    stop('Discretizations has to be a positive integer.')
  }

  if (is.null(pc.xi)) {
    if (dimensions == 1 && discretizations <= 5) {
      pc.xi <- 0.25
    } else {
      pc.xi <- 2
    }
  } else if (!is.numeric(pc.xi) || pc.xi <= 0) {
    stop('pc.xi has to be a real number strictly greater than 0.')
  }

  if (is.null(range)) {
    min.obj <- min(sum(decision == 0), sum(decision == 1))
    range <- GetRange(n = min.obj, dimensions = dimensions, divisions = divisions)
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

    if ((divisions+1)^dimensions > 256) {
      stop('CUDA acceleration does not support more than 256 cubes = (divisions+1)^dimensions')
    }

    if (return.tuples) {
      stop('CUDA acceleration does not support return.tuples parameter (for now)')
    }

    if (return.min) {
      stop('CUDA acceleration does not support return.min parameter (for now)')
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
      as.double(pc.xi),
      as.integer(interesting.vars[order(interesting.vars)] - 1),  # send C-compatible 0-based indices
      as.logical(require.all.vars),
      as.logical(return.tuples),
      as.logical(return.min),
      as.logical(use.CUDA))

  if (return.tuples) {
    names(out) <- c("IG", "Tuple", "Discretization.nr")
    out$Tuple = t(out$Tuple + 1) # restore R-compatible 1-based indices, transpose to remain compatible with ComputeInterestingTuples
    out$Discretization.nr = out$Discretization.nr + 1 # restore R-compatible 1-based indices
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
    pc.xi           = pc.xi)

  return(out)
}

#' Interesting tuples
#'
#' @details
#' If no filtering is applied, this function is able to run in an
#' optimised fashion. It is recommended to avoid filtering if only it is
#' feasible.
#'
#' @param data input data where columns are variables and rows are observations (all numeric)
#' @param decision decision variable as a binary sequence of length equal to number of observations
#' @param dimensions number of dimensions (a positive integer; 5 max) - FIXME: only 2D supported for now!
#' @param divisions number of divisions (from 1 to 15; \code{NULL} selects probable optimal number)
#' @param discretizations number of discretizations
#' @param seed seed for PRNG used during discretizations (\code{NULL} for random)
#' @param range discretization range (from 0.0 to 1.0; \code{NULL} selects probable optimal number)
#' @param pc.xi parameter xi used to compute pseudocounts (the default is recommended not to be changed)
#' @param ig.thr IG threshold above which the tuple is interesting (0 and negative mean no filtering)
#' @param I.lower IG values computed for lower dimension (1D for 2D, etc.)
#' @param interesting.vars variables for which to check the IGs (none = all)
#' @param require.all.vars boolean whether to require tuple to consist of only interesting.vars
#' @param return.matrix boolean whether to return a matrix instead of a list (ignored if not using the optimised method variant)
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
#' ig.1d <- ComputeMaxInfoGains(madelon$data, madelon$decision, dimensions = 1, divisions = 1,
#'                              range = 0, seed = 0)
#' ComputeInterestingTuples(madelon$data, madelon$decision, dimensions = 2, divisions = 1,
#'                          range = 0, seed = 0, ig.thr = 100, I.lower = ig.1d$IG)
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
    pc.xi = 0.25,
    ig.thr = 0,
    I.lower,
    interesting.vars = vector(mode = "integer"),
    require.all.vars = FALSE,
    return.matrix = FALSE) {
  data <- data.matrix(data)
  storage.mode(data) <- "double"
  decision <- as.vector(decision, mode="integer")

  if (length(decision) != nrow(data)) {
    stop('Length of decision is not equal to the number of rows in data.')
  }

  if (length(I.lower) != ncol(data)) {
    stop('Length of I.lower is not equal to the number of columns in data.')
  }

  # try to rebase decision from 1 to 0 (as is the case with e.g. factors)
  if (!any(decision == 0)) {
    decision <- decision - as.integer(1) # as.integer is crucial in keeping this an integer vector
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

  if (dimensions > 5) {
    stop('Dimensions cannot exceed 5')
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

  if (as.integer(discretizations) != discretizations || discretizations < 1) {
    stop('Discretizations has to be a positive integer.')
  }

  if (is.null(pc.xi)) {
    if (dimensions == 1 && discretizations <= 5) {
      pc.xi <- 0.25
    } else {
      pc.xi <- 2
    }
  } else if (!is.numeric(pc.xi) || pc.xi <= 0) {
    stop('pc.xi has to be a real number strictly greater than 0.')
  }

  if (is.null(range)) {
    min.obj <- min(sum(decision == 0), sum(decision == 1))
    range <- GetRange(n = min.obj, dimensions = dimensions, divisions = divisions)
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

  if (dimensions != 2) {
    # FIXME:
    stop('More dimensions than 2 not supported for now')
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
      as.double(pc.xi),
      as.integer(interesting.vars[order(interesting.vars)] - 1),  # send C-compatible 0-based indices
      as.logical(require.all.vars),
      as.double(ig.thr),
      as.double(I.lower),
      as.logical(return.matrix))

  if (length(interesting.vars) == 0 && ig.thr <= 0 && return.matrix) {
    # do nothing, we have a matrix for you
  } else {
    if (length(result[[1]]) == 0) {
      warning("No tuples were returned.")
      return(NULL)
    }

    names(result) <- c("Var", "Tuple", "IG")

    result$Var = result$Var + 1 # restore R-compatible 1-based indices
    result$Tuple = result$Tuple + 1 # restore R-compatible 1-based indices

    result <- as.data.frame(result)
  }

  attr(result, 'run.params') <- list(
    dimensions      = dimensions,
    divisions       = divisions,
    discretizations = discretizations,
    seed            = seed,
    range           = range,
    pc.xi           = pc.xi)

  return(result)
}

#' Discretize variable on demand
#'
#' @param data input data where columns are variables and rows are observations (all numeric)
#' @param variable.idx variable index (as it appears in \code{data})
#' @param divisions number of divisions
#' @param discretization.nr discretization number (positive integer)
#' @param seed seed for PRNG
#' @param range discretization range
#' @return Discretized variable.
#' @examples
#' Discretize(madelon$data, 3, 1, 1, 0, 0.5)
#' @export
#' @useDynLib MDFS r_discretize
Discretize <- function(
    data,
    variable.idx,
    divisions,
    discretization.nr,
    seed,
    range) {
  data <- data.matrix(data)
  storage.mode(data) <- "double"

  if (as.integer(divisions) != divisions || divisions < 1 || divisions > 15) {
    stop('Divisions has to be an integer between 1 and 15 (inclusive).')
  }

  if (as.integer(variable.idx) != variable.idx || variable.idx < 1) {
    stop('variable.idx has to be a positive integer.')
  }

  if (variable.idx > dim(data)[2]) {
    stop('variable.idx has to be in data bounds.')
  }

  if (as.integer(discretization.nr) != discretization.nr || discretization.nr < 1) {
    stop('discretization.nr has to be a positive integer.')
  }

  if (as.double(range) != range || range < 0 || range > 1) {
    stop('Range has to be a number between 0.0 and 1.0')
  }

  if (as.integer(seed) != seed || seed < 0 || seed > 2^31-1) {
    warning('Only integer seeds from 0 to 2^31-1 are portable. Using non-portable seed may make result harder to reproduce.')
  }

  variable <- data[, variable.idx]

  result <- .Call(
      r_discretize,
      variable,
      as.integer(variable.idx - 1), # convert to C 0-based
      as.integer(divisions),
      as.integer(discretization.nr - 1), # convert to C 0-based
      as.integer(seed),
      as.double(range))

  attr(result, 'run.params') <- list(
    variable.idx      = variable.idx,
    divisions         = divisions,
    discretization.nr = discretization.nr,
    seed              = seed,
    range             = range)

  return(result)
}
