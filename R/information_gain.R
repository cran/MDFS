#' Max information gains
#'
#' @details
#' If \code{decision} is omitted, this function calculates either the variable entropy
#' (in 1D) or mutual information (in higher dimensions).
#' Translate "IG" respectively to entropy or mutual information in the
#' rest of this function's description.
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
    decision = NULL,
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

  decision <- prepare_decision(decision)

  if (!is.null(decision)) {
    if (length(decision) != nrow(data)) {
      stop("Length of decision is not equal to the number of rows in data.")
    }

    count_0 <- sum(decision == 0)
    min.obj <- min(count_0, length(decision) - count_0)
  } else {
    min.obj <- nrow(data)
  }

  dimensions <- prepare_integer_in_bounds(dimensions, "Dimensions", as.integer(1), as.integer(5))

  if (is.null(divisions)) {
    if (dimensions == 1) {
      divisions <- min(floor(sqrt(min.obj)), 15)
    } else {
      divisions <- 1
    }
  }

  divisions <- prepare_integer_in_bounds(divisions, "Divisions", as.integer(1), as.integer(15))

  discretizations <- prepare_integer_in_bounds(discretizations, "Discretizations", as.integer(1))

  pc.xi <- prepare_double_in_bounds(pc.xi, "pc.xi", .Machine$double.xmin)

  if (is.null(range)) {
    range <- GetRange(n = min.obj, dimensions = dimensions, divisions = divisions)
  }

  range <- prepare_double_in_bounds(range, "Range", 0.0, 1.0)

  if (range == 0 && discretizations > 1) {
    stop("Zero range does not make sense with more than one discretization. All will always be equal.")
  }

  if (is.null(seed)) {
    seed <- round(runif(1, 0, 2^31 - 1)) # unsigned passed as signed, the highest bit remains unused for best compatibility
  }

  seed <- prepare_integer_in_bounds(seed, "Seed", as.integer(0))

  if (dimensions == 1 && return.tuples) {
    stop("return.tuples does not make sense in 1D")
  }

  if (use.CUDA) {
    if (dimensions == 1) {
      stop("CUDA acceleration does not support 1 dimension")
    }

    if ((divisions + 1)^dimensions > 256) {
      stop("CUDA acceleration does not support more than 256 cubes = (divisions+1)^dimensions")
    }

    if (return.tuples) {
      stop("CUDA acceleration does not support return.tuples parameter (for now)")
    }

    if (return.min) {
      stop("CUDA acceleration does not support return.min parameter (for now)")
    }

    if (length(interesting.vars) > 0) {
      stop("CUDA acceleration does not support interesting.vars parameter (for now)")
    }
  }

  out <- .Call(
      r_compute_max_ig,
      data,
      decision,
      dimensions,
      divisions,
      discretizations,
      seed,
      range,
      pc.xi,
      as.integer(interesting.vars[order(interesting.vars)] - 1),  # send C-compatible 0-based indices
      as.logical(require.all.vars),
      as.logical(return.tuples),
      as.logical(return.min),
      as.logical(use.CUDA))

  if (return.tuples) {
    names(out) <- c("IG", "Tuple", "Discretization.nr")
    out$Tuple <- t(out$Tuple + 1) # restore R-compatible 1-based indices, transpose to remain compatible with ComputeInterestingTuples
    out$Discretization.nr <- out$Discretization.nr + 1 # restore R-compatible 1-based indices
  } else {
    names(out) <- c("IG")
  }

  out <- as.data.frame(out)

  attr(out, "run.params") <- list(
    dimensions      = dimensions,
    divisions       = divisions,
    discretizations = discretizations,
    seed            = seed,
    range           = range,
    pc.xi           = pc.xi)

  return(out)
}

#' Max information gains (discrete)
#'
#' @details
#' If \code{decision} is omitted, this function calculates either the variable entropy
#' (in 1D) or mutual information (in higher dimensions).
#' Translate "IG" respectively to entropy or mutual information in the
#' rest of this function's description.
#'
#' @param data input data where columns are variables and rows are observations (all discrete with the same number of categories)
#' @param decision decision variable as a binary sequence of length equal to number of observations
#' @param dimensions number of dimensions (a positive integer; 5 max)
#' @param pc.xi parameter xi used to compute pseudocounts (the default is recommended not to be changed)
#' @param return.tuples whether to return tuples where max IG was observed (one tuple per variable) - not supported with CUDA nor in 1D
#' @param return.min whether to return min instead of max (per tuple) - not supported with CUDA
#' @param interesting.vars variables for which to check the IGs (none = all) - not supported with CUDA
#' @param require.all.vars boolean whether to require tuple to consist of only interesting.vars
#' @return A \code{\link{data.frame}} with the following columns:
#'  \itemize{
#'    \item \code{IG} -- max information gain (of each variable)
#'    \item \code{Tuple.1, Tuple.2, ...} -- corresponding tuple (up to \code{dimensions} columns, available only when \code{return.tuples == T})
#'    \item \code{Discretization.nr} -- always 1 (for compatibility with the non-discrete function; available only when \code{return.tuples == T})
#'  }
#'
#'  Additionally attribute named \code{run.params} with run parameters is set on the result.
#' @examples
#' \donttest{
#' ComputeMaxInfoGainsDiscrete(madelon$data > 500, madelon$decision, dimensions = 2)
#' }
#' @importFrom stats runif
#' @export
#' @useDynLib MDFS r_compute_max_ig_discrete
ComputeMaxInfoGainsDiscrete <- function(
    data,
    decision = NULL,
    dimensions = 1,
    pc.xi = 0.25,
    return.tuples = FALSE,
    return.min = FALSE,
    interesting.vars = vector(mode = "integer"),
    require.all.vars = FALSE) {
  data <- data.matrix(data)
  storage.mode(data) <- "integer"

  decision <- prepare_decision(decision)

  if (!is.null(decision)) {
    if (length(decision) != nrow(data)) {
      stop("Length of decision is not equal to the number of rows in data.")
    }
  }

  dimensions <- prepare_integer_in_bounds(dimensions, "Dimensions", as.integer(1), as.integer(5))

  divisions <- length(unique(c(data))) - 1

  divisions <- prepare_integer_in_bounds(divisions, "Divisions", as.integer(1), as.integer(15))

  pc.xi <- prepare_double_in_bounds(pc.xi, "pc.xi", .Machine$double.xmin)

  if (dimensions == 1 && return.tuples) {
    stop("return.tuples does not make sense in 1D")
  }

  out <- .Call(
      r_compute_max_ig_discrete,
      data,
      decision,
      dimensions,
      divisions,
      pc.xi,
      as.integer(interesting.vars[order(interesting.vars)] - 1),  # send C-compatible 0-based indices
      as.logical(require.all.vars),
      as.logical(return.tuples),
      as.logical(return.min),
      FALSE)  # CUDA variant is not implemented here

  if (return.tuples) {
    names(out) <- c("IG", "Tuple", "Discretization.nr")
    out$Tuple <- t(out$Tuple + 1) # restore R-compatible 1-based indices, transpose to remain compatible with ComputeInterestingTuples
    out$Discretization.nr <- out$Discretization.nr + 1 # restore R-compatible 1-based indices
  } else {
    names(out) <- c("IG")
  }

  out <- as.data.frame(out)

  attr(out, "run.params") <- list(
    dimensions      = dimensions,
    pc.xi           = pc.xi)

  return(out)
}

#' Interesting tuples
#'
#' @details
#' If running in 2D and no filtering is applied, this function is able to run in an
#' optimised fashion. It is recommended to avoid filtering in 2D if only it is
#' feasible.
#'
#' @details
#' If \code{decision} is omitted, this function calculates mutual information.
#' Translate "IG" to mutual information in the rest of this function's
#' description, except for \code{I.lower} where it means entropy.
#'
#' @param data input data where columns are variables and rows are observations (all numeric)
#' @param decision decision variable as a binary sequence of length equal to number of observations
#' @param dimensions number of dimensions (a positive integer; 5 max)
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
    decision = NULL,
    dimensions = 2,
    divisions = NULL,
    discretizations = 1,
    seed = NULL,
    range = NULL,
    pc.xi = 0.25,
    ig.thr = 0,
    I.lower = NULL,
    interesting.vars = vector(mode = "integer"),
    require.all.vars = FALSE,
    return.matrix = FALSE) {
  data <- data.matrix(data)
  storage.mode(data) <- "double"

  if (!is.null(I.lower)) {
    if (length(I.lower) != ncol(data)) {
      stop("Length of I.lower is not equal to the number of columns in data.")
    }

    if (dimensions != 2) {
      # TODO:
      stop("More dimensions than 2 not supported with I.lower set.")
    }

    I.lower <- as.double(I.lower)
  }

  decision <- prepare_decision(decision)

  if (!is.null(decision)) {
    if (length(decision) != nrow(data)) {
      stop("Length of decision is not equal to the number of rows in data.")
    }

    count_0 <- sum(decision == 0)
    min.obj <- min(count_0, length(decision) - count_0)
  } else {
    min.obj <- nrow(data)
  }

  dimensions <- prepare_integer_in_bounds(dimensions, "Dimensions", as.integer(2), as.integer(5))

  if (is.null(divisions)) {
    if (dimensions == 1) {
      divisions <- min(floor(sqrt(min.obj)), 15)
    } else {
      divisions <- 1
    }
  }

  divisions <- prepare_integer_in_bounds(divisions, "Divisions", as.integer(1), as.integer(15))

  discretizations <- prepare_integer_in_bounds(discretizations, "Discretizations", as.integer(1))

  pc.xi <- prepare_double_in_bounds(pc.xi, "pc.xi", .Machine$double.xmin)

  if (is.null(range)) {
    range <- GetRange(n = min.obj, dimensions = dimensions, divisions = divisions)
  }

  range <- prepare_double_in_bounds(range, "Range", 0.0, 1.0)

  if (range == 0 && discretizations > 1) {
    stop("Zero range does not make sense with more than one discretization. All will always be equal.")
  }

  if (is.null(seed)) {
    seed <- round(runif(1, 0, 2^31 - 1)) # unsigned passed as signed, the highest bit remains unused for best compatibility
  }

  seed <- prepare_integer_in_bounds(seed, "Seed", as.integer(0))

  result <- .Call(
      r_compute_all_matching_tuples,
      data,
      decision,
      dimensions,
      divisions,
      discretizations,
      seed,
      range,
      pc.xi,
      as.integer(interesting.vars[order(interesting.vars)] - 1),  # send C-compatible 0-based indices
      as.logical(require.all.vars),
      as.double(ig.thr),
      I.lower,
      as.logical(return.matrix))

  if (dimensions == 2 && length(interesting.vars) == 0 && ig.thr <= 0 && return.matrix) {
    # do nothing, we have a matrix for you
  } else {
    if (length(result[[1]]) == 0) {
      warning("No tuples were returned.")
      return(NULL)
    }

    names(result) <- c("Var", "Tuple", "IG")

    result$Var <- result$Var + 1 # restore R-compatible 1-based indices
    result$Tuple <- result$Tuple + 1 # restore R-compatible 1-based indices

    result <- as.data.frame(result)
  }

  attr(result, "run.params") <- list(
    dimensions      = dimensions,
    divisions       = divisions,
    discretizations = discretizations,
    seed            = seed,
    range           = range,
    pc.xi           = pc.xi)

  return(result)
}

#' Interesting tuples (discrete)
#'
#' @details
#' If running in 2D and no filtering is applied, this function is able to run in an
#' optimised fashion. It is recommended to avoid filtering in 2D if only it is
#' feasible.
#'
#' @details
#' If \code{decision} is omitted, this function calculates mutual information.
#' Translate "IG" to mutual information in the rest of this function's
#' description, except for \code{I.lower} where it means entropy.
#'
#' @param data input data where columns are variables and rows are observations (all discrete with the same number of categories)
#' @param decision decision variable as a binary sequence of length equal to number of observations
#' @param dimensions number of dimensions (a positive integer; 5 max)
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
#' ig.1d <- ComputeMaxInfoGainsDiscrete(madelon$data > 500, madelon$decision, dimensions = 1)
#' ComputeInterestingTuplesDiscrete(madelon$data > 500, madelon$decision, dimensions = 2,
#'                                  ig.thr = 100, I.lower = ig.1d$IG)
#' }
#' @export
#' @useDynLib MDFS r_compute_all_matching_tuples_discrete
ComputeInterestingTuplesDiscrete <- function(
    data,
    decision = NULL,
    dimensions = 2,
    pc.xi = 0.25,
    ig.thr = 0,
    I.lower = NULL,
    interesting.vars = vector(mode = "integer"),
    require.all.vars = FALSE,
    return.matrix = FALSE) {
  data <- data.matrix(data)
  storage.mode(data) <- "integer"

  if (!is.null(I.lower)) {
    if (length(I.lower) != ncol(data)) {
      stop("Length of I.lower is not equal to the number of columns in data.")
    }

    if (dimensions != 2) {
      # TODO:
      stop("More dimensions than 2 not supported with I.lower set.")
    }

    I.lower <- as.double(I.lower)
  }

  decision <- prepare_decision(decision)

  if (!is.null(decision)) {
    if (length(decision) != nrow(data)) {
      stop("Length of decision is not equal to the number of rows in data.")
    }
  }

  dimensions <- prepare_integer_in_bounds(dimensions, "Dimensions", as.integer(2), as.integer(5))

  divisions <- length(unique(c(data))) - 1

  divisions <- prepare_integer_in_bounds(divisions, "Divisions", as.integer(1), as.integer(15))

  pc.xi <- prepare_double_in_bounds(pc.xi, "pc.xi", .Machine$double.xmin)

  result <- .Call(
      r_compute_all_matching_tuples_discrete,
      data,
      decision,
      dimensions,
      divisions,
      pc.xi,
      as.integer(interesting.vars[order(interesting.vars)] - 1),  # send C-compatible 0-based indices
      as.logical(require.all.vars),
      as.double(ig.thr),
      I.lower,
      as.logical(return.matrix))

  if (dimensions == 2 && length(interesting.vars) == 0 && ig.thr <= 0 && return.matrix) {
    # do nothing, we have a matrix for you
  } else {
    if (length(result[[1]]) == 0) {
      warning("No tuples were returned.")
      return(NULL)
    }

    names(result) <- c("Var", "Tuple", "IG")

    result$Var <- result$Var + 1 # restore R-compatible 1-based indices
    result$Tuple <- result$Tuple + 1 # restore R-compatible 1-based indices

    result <- as.data.frame(result)
  }

  attr(result, "run.params") <- list(
    dimensions      = dimensions,
    divisions       = divisions,
    range           = range,
    pc.xi           = pc.xi)

  return(result)
}
