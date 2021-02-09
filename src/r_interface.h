#include <Rinternals.h>

extern "C"
SEXP r_compute_max_ig(
	SEXP Rin_data,
	SEXP Rin_decision,
	SEXP Rin_dimensions,
	SEXP Rin_divisions,
	SEXP Rin_discretizations,
	SEXP Rin_seed,
	SEXP Rin_range,
	SEXP Rin_pseudocount,
	SEXP Rin_interesting_vars,
	SEXP Rin_require_all_vars,
	SEXP Rin_return_tuples,
	SEXP Rin_return_min,
	SEXP Rin_use_cuda
);

extern "C"
	SEXP r_compute_all_matching_tuples(
	SEXP Rin_data,
	SEXP Rin_decision,
	SEXP Rin_dimensions,
	SEXP Rin_divisions,
	SEXP Rin_discretizations,
	SEXP Rin_seed,
	SEXP Rin_range,
	SEXP Rin_pseudocount,
	SEXP Rin_interesting_vars,
	SEXP Rin_require_all_vars,
	SEXP Rin_ig_thr,
	SEXP Rin_I_lower,
	SEXP Rin_return_matrix
);

extern "C"
	SEXP r_discretize(
	SEXP Rin_variable,
	SEXP Rin_variable_nr,
	SEXP Rin_divisions,
	SEXP Rin_discretization_nr,
	SEXP Rin_seed,
	SEXP Rin_range
);
