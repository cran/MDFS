#include "r_interface.h"

#include "cpu/dataset.h"
#include "cpu/mdfs.h"

#ifdef WITH_CUDA
// required to report errors - currently only CUDA reports any errors
#include <R.h>
#include "gpu/cucubes.h"
#endif

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
        SEXP Rin_return_tuples,
        SEXP Rin_use_cuda)
{
    #ifndef WITH_CUDA
    if (asLogical(Rin_use_cuda)) {
        error("CUDA acceleration not compiled");
    }
    #endif

    const int* dataDims = INTEGER(getAttrib(Rin_data, R_DimSymbol));

    const int obj_count = dataDims[0];
    const int variable_count = dataDims[1];

    #ifdef WITH_CUDA
    if (asLogical(Rin_use_cuda)) {
        SEXP Rout_max_igs = PROTECT(allocVector(REALSXP, variable_count));

        try {
            run_cucubes(
                    obj_count,
                    variable_count,
                    asInteger(Rin_dimensions),
                    asInteger(Rin_divisions),
                    asInteger(Rin_discretizations),
                    asInteger(Rin_seed),
                    asReal(Rin_range),
                    asReal(Rin_pseudocount),
                    REAL(Rin_data),
                    INTEGER(Rin_decision),
                    REAL(Rout_max_igs));
        } catch (const cudaException& e) {
            // TODO: ensure cleanup inside library
            error("CUDA exception: %s (in %s:%d)", cudaGetErrorString(e.code), e.file, e.line);
        } catch (const NotImplementedException& e) {
            // TODO: is it possible to get this?
            error("Not-implemented exception: %s", e.msg.c_str());
        }

        const int result_members_count = 1;

        SEXP Rout_result = PROTECT(allocVector(VECSXP, result_members_count));
        SET_VECTOR_ELT(Rout_result, 0, Rout_max_igs);

        UNPROTECT(1 + result_members_count);

        return Rout_result;
    }
    #endif

    const int discretizations = asInteger(Rin_discretizations);
    const int divisions = asInteger(Rin_divisions);

    RawData rawdata(RawDataInfo(obj_count, variable_count), REAL(Rin_data), INTEGER(Rin_decision));

    DataSet dataset;

    dataset.loadData(
        &rawdata,
        DiscretizationInfo(
            asInteger(Rin_seed), 
            discretizations, 
            divisions, 
            asReal(Rin_range)
        )
    );

    MDFSInfo mdfs_info(
        asInteger(Rin_dimensions),
        divisions,
        discretizations,
        asReal(Rin_pseudocount),
        0.0f,
        INTEGER(Rin_interesting_vars),
        length(Rin_interesting_vars),
        true // true in max IGs, false in all matching tuple
    );

    SEXP Rout_max_igs = PROTECT(allocVector(REALSXP, variable_count));
    SEXP Rout_tuples = nullptr;

    const bool return_tuples = asLogical(Rin_return_tuples);
    MDFSOutput mdfs_output(MDFSOutputType::MaxIGs, variable_count);
    if (return_tuples) {
        Rout_tuples = PROTECT(allocMatrix(INTSXP, mdfs_info.dimensions, variable_count));
        mdfs_output.setMaxIGsTuples(INTEGER(Rout_tuples)); // tuples are set row-first during computation, we transpose the result in R to speed up C code
    }

    scalarMDFS(mdfs_info, &dataset, mdfs_output);

    mdfs_output.copyMaxIGsAsDouble(REAL(Rout_max_igs));

    int result_members_count = 1;

    if (return_tuples) {
        result_members_count += 1;
    }

    SEXP Rout_result = PROTECT(allocVector(VECSXP, result_members_count));
    SET_VECTOR_ELT(Rout_result, 0, Rout_max_igs);

    if (return_tuples) {
        SET_VECTOR_ELT(Rout_result, 1, Rout_tuples);
    }

    UNPROTECT(1 + result_members_count);

    return Rout_result;
}

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
        SEXP Rin_ig_thr)
{
    const int* dataDims = INTEGER(getAttrib(Rin_data, R_DimSymbol));

    const int obj_count = dataDims[0];
    const int variable_count = dataDims[1];

    const int discretizations = asInteger(Rin_discretizations);
    const int divisions = asInteger(Rin_divisions);

    RawData rawdata(RawDataInfo(obj_count, variable_count), REAL(Rin_data), INTEGER(Rin_decision));

    DataSet dataset;

    dataset.loadData(
        &rawdata,
        DiscretizationInfo(
            asInteger(Rin_seed), 
            discretizations, 
            divisions, 
            asReal(Rin_range)
        )
    );

    MDFSInfo mdfs_info(
        asInteger(Rin_dimensions),
        divisions,
        discretizations,
        asReal(Rin_pseudocount),
        asReal(Rin_ig_thr),
        INTEGER(Rin_interesting_vars),
        length(Rin_interesting_vars),
        false // true in max IGs, false in all matching tuple
    );

    MDFSOutput mdfs_output(MDFSOutputType::MatchingTuples, variable_count);

    scalarMDFS(mdfs_info, &dataset, mdfs_output);

    const int result_members_count = 3;
    const int tuples_count = mdfs_output.getMatchingTuplesCount();

    SEXP Rout_igs = PROTECT(allocVector(REALSXP, tuples_count));
    SEXP Rout_tuples = PROTECT(allocMatrix(INTSXP, tuples_count, mdfs_info.dimensions));
    SEXP Rout_vars = PROTECT(allocVector(INTSXP, tuples_count));

    mdfs_output.copyMatchingTuples(INTEGER(Rout_vars), REAL(Rout_igs), INTEGER(Rout_tuples));

    SEXP Rout_result = PROTECT(allocVector(VECSXP, result_members_count));
    SET_VECTOR_ELT(Rout_result, 0, Rout_vars);
    SET_VECTOR_ELT(Rout_result, 1, Rout_tuples);
    SET_VECTOR_ELT(Rout_result, 2, Rout_igs);

    UNPROTECT(1 + result_members_count);

    return Rout_result;
}
