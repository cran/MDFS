#ifndef MDFS
#define MDFS

#include "mdfs_cpu_kernel_incremental.h"
#include "mdfs_cpu_kernel.h"

#include "common.h"
#include "dataset.h"
#include "discretize.h"

#include <algorithm>
#include <limits>
#include <memory>


template <uint8_t n_decision_classes, uint8_t n_dimensions>
void scalarMDFS(
    const MDFSInfo& mdfs_info,
    RawData* raw_data,
    std::unique_ptr<const DiscretizationInfo> dfi,
    MDFSOutput& out
) {
    size_t c[n_decision_classes];
    for (uint8_t i = 0; i < n_decision_classes; i++) {
        c[i] = 0;
    }
    uint8_t* decision = nullptr;
    if (n_decision_classes > 1) {
        decision = new uint8_t[raw_data->info.object_count];
        for (size_t i = 0; i < raw_data->info.object_count; i++) {
            decision[i] = raw_data->decision[i];
            c[decision[i]]++;
        }
    } else {
        c[0] = raw_data->info.object_count;
    }
    const float cmin = *std::min_element(c, c+n_decision_classes);

    // this is to treat ig_thr=0 and below as unset (ignored) and allow for the
    // numeric errors to pass through the relevance filter (IGs end up being
    // negative in some rare cases due to logarithm rounding)
    const float ig_thr = mdfs_info.ig_thr > 0.0f ? mdfs_info.ig_thr : -std::numeric_limits<float>::infinity();

    float* I_lower = nullptr;
    if (mdfs_info.I_lower != nullptr) {
        I_lower = new float[raw_data->info.variable_count];
        for (size_t i = 0; i < raw_data->info.variable_count; i++) {
            I_lower[i] = mdfs_info.I_lower[i];
        }
    }

    float* H = nullptr;
    if (n_decision_classes == 1 && n_dimensions == 2) {
        H = new float[raw_data->info.variable_count];
    }

    float p[n_decision_classes];
    for (uint8_t i = 0; i < n_decision_classes; i++) {
        p[i] = c[i] / cmin * mdfs_info.pseudo;
    }

    const size_t n_classes = mdfs_info.divisions + 1;
    const size_t num_of_cubes = std::pow(n_classes, n_dimensions);
    const size_t num_of_cubes_reduced = std::pow(n_classes, n_dimensions - 1);

    const auto d2 = n_classes*n_classes;
    const auto d3 = d2*n_classes;
    const auto d4 = d3*n_classes;
    const size_t d[3] = {d2, d3, d4};

    float H_Y_counters[n_decision_classes];
    for (uint8_t i = 0; i < n_decision_classes; i++) {
        H_Y_counters[i] = c[i] + p[i] * num_of_cubes;
    }
    // H(Y) (plain) entropy of decision (computed simply as conditional given an empty set of vars)
    const float H_Y = conditional_entropy<n_decision_classes>(1, H_Y_counters);

    const size_t n_vars_to_discretize = mdfs_info.interesting_vars_count && mdfs_info.require_all_vars ?
                                        mdfs_info.interesting_vars_count :
                                        raw_data->info.variable_count;

    if (out.type == MDFSOutputType::MinIGs) {
        // we will be doing overall max on discretizations
        std::fill(out.max_igs->begin(), out.max_igs->end(), -std::numeric_limits<float>::infinity());
    }

    // total of all counters; used only in no decision mode
    const float total_counters = raw_data->info.object_count + p[0] * num_of_cubes;

    for (size_t discretization_id = 0; discretization_id < mdfs_info.discretizations; discretization_id++) {
        TupleGenerator* generator;
        if (mdfs_info.interesting_vars_count && mdfs_info.require_all_vars) {
            generator = new TupleGenerator(n_dimensions, std::vector<size_t>(mdfs_info.interesting_vars,
                                                                             mdfs_info.interesting_vars + mdfs_info.interesting_vars_count));
        } else {
            generator = new TupleGenerator(n_dimensions, raw_data->info.variable_count);
        }

        uint8_t* data = new uint8_t[raw_data->info.object_count * raw_data->info.variable_count];

        if (dfi) {
            for (size_t i = 0; i < n_vars_to_discretize; ++i) {
                const size_t v = mdfs_info.interesting_vars_count && mdfs_info.require_all_vars ?
                                 mdfs_info.interesting_vars[i] :
                                 i;
                const double* in_data = raw_data->getVariable(v);

                std::vector<double> sorted_in_data(in_data, in_data + raw_data->info.object_count);
                std::sort(sorted_in_data.begin(), sorted_in_data.end());

                discretize(
                    dfi->seed,
                    discretization_id,
                    v,
                    dfi->divisions,
                    raw_data->info.object_count,
                    in_data,
                    sorted_in_data,
                    data + v * raw_data->info.object_count,
                    dfi->range
                );
            }
        } else {
            // rewrite int to uint8_t
            for (size_t i = 0; i < n_vars_to_discretize; ++i) {
                const size_t v = mdfs_info.interesting_vars_count && mdfs_info.require_all_vars ?
                                 mdfs_info.interesting_vars[i] :
                                 i;
                const int* in_data = raw_data->getVariableI(v);
                uint8_t* data_current_var = data + v * raw_data->info.object_count;

                for (size_t i = 0; i < raw_data->info.object_count; i++) {
                    data_current_var[i] = in_data[i];
                }
            }
        }

        if (H != nullptr) {
            float* mini_counters = new float[n_classes];
            float p_local = p[0] * num_of_cubes_reduced;  // to match counting in higher dimensions
            for (size_t i = 0; i < raw_data->info.variable_count; i++) {
                count_counters<1, 1>(data, nullptr, raw_data->info.object_count, 0, &i, mini_counters, n_classes, &p_local, nullptr);
                // H(X_k) (plain) entropy of the current var
                H[i] = entropy(total_counters, n_classes, mini_counters);
            }
            delete[] mini_counters;
        }

        MDFSOutput* local_mdfs_output = nullptr;
        if (out.type == MDFSOutputType::MinIGs) {
            local_mdfs_output = new MDFSOutput(MDFSOutputType::MinIGs, n_dimensions, raw_data->info.variable_count);
            if (out.max_igs_tuples != nullptr) {
                local_mdfs_output->setMaxIGsTuples(new int[n_dimensions*raw_data->info.variable_count], new int[raw_data->info.variable_count]);
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            size_t* tuple = new size_t[n_dimensions];
            float* igs = new float[n_dimensions];
            float* counters = new float[n_decision_classes * num_of_cubes];
            float* reduced = new float[n_decision_classes * num_of_cubes_reduced];

            bool hasWorkToDo = true;

            do {
                #ifdef _OPENMP
                #pragma omp critical (GetTheNextTuple)
                #endif
                if (!generator->hasNext()) { // we have to synchronize because TupleGenerator is not omp-thread-safe and it has to be shared
                    hasWorkToDo = false;
                    // one does not simply leave the OpenMP's critical section
                    // no, seriously, it is programmed such that you cannot use break/continue here
                    // hence we break later ¯\_(ツ)_/¯
                } else {
                    generator->next(tuple);
                }

                if (!hasWorkToDo) {
                    break; // this is right where we break, OpenMP :-)
                }

                if (mdfs_info.interesting_vars_count && !mdfs_info.require_all_vars) {
                    std::list<int> current_interesting_vars;
                    std::set_intersection(
                        tuple, tuple+n_dimensions,
                        mdfs_info.interesting_vars, mdfs_info.interesting_vars + mdfs_info.interesting_vars_count,
                        std::back_inserter(current_interesting_vars));

                    if (current_interesting_vars.empty()) {
                        continue;
                    }
                }

                if (n_dimensions >= 2) {
                    if (I_lower == nullptr) {
                        process_tuple<n_decision_classes, n_dimensions>(
                            data,
                            decision,
                            raw_data->info.object_count,
                            n_classes,
                            tuple,
                            counters, reduced,
                            num_of_cubes, num_of_cubes_reduced,
                            p,
                            total_counters,
                            d,
                            H,
                            igs,
                            nullptr);
                    } else {
                        // TODO: so far only 2D supported
                        process_tuple_incremental<n_decision_classes, n_dimensions>(
                            data,
                            decision,
                            raw_data->info.object_count,
                            n_classes,
                            tuple,
                            counters,
                            num_of_cubes,
                            p,
                            total_counters,
                            d,
                            H_Y,
                            I_lower,
                            igs,
                            nullptr);
                    }
                } else {  // 1D
                    process_tuple_incremental<n_decision_classes, n_dimensions>(
                        data,
                        decision,
                        raw_data->info.object_count,
                        n_classes,
                        tuple,
                        counters,
                        num_of_cubes,
                        p,
                        total_counters,
                        d,
                        H_Y,
                        nullptr,
                        igs,
                        nullptr);
                }

                #ifdef _OPENMP
                #pragma omp critical (SetOutput)
                #endif
                switch (out.type) { // we have to synchronize because MDFSOutput is not omp-thread-safe and it has to be shared
                    case MDFSOutputType::MaxIGs:
                        out.updateMaxIG(tuple, igs, discretization_id);
                        break;

                    case MDFSOutputType::MinIGs:
                        local_mdfs_output->updateMinIG(tuple, igs, discretization_id);
                        break;

                    case MDFSOutputType::MatchingTuples:
                        for (size_t v = 0; v < n_dimensions; ++v) {
                            if (igs[v] > ig_thr) {
                                out.addTuple(tuple[v], igs[v], discretization_id, tuple);
                            }
                        }
                        break;

                    case MDFSOutputType::AllTuples:
                        out.updateAllTuplesIG(tuple, igs, discretization_id);
                        break;
                }
            } while (true);

            delete[] reduced;
            delete[] counters;
            delete[] igs;
            delete[] tuple;
        }

        if (local_mdfs_output != nullptr) {
            // need max over discretizations so copy to out whichever max
            for (size_t i = 0; i < raw_data->info.variable_count; i++) {
                if ((*local_mdfs_output->max_igs)[i] > (*out.max_igs)[i]) {
                    (*out.max_igs)[i] = (*local_mdfs_output->max_igs)[i];
                    if (out.max_igs_tuples != nullptr) {
                        std::copy(local_mdfs_output->max_igs_tuples + n_dimensions * i,
                                  local_mdfs_output->max_igs_tuples + n_dimensions * (i+1),
                                  out.max_igs_tuples + n_dimensions * i);
                        out.dids[i] = local_mdfs_output->dids[i];
                    }
                }
            }
            if (out.max_igs_tuples != nullptr) {
                delete[] local_mdfs_output->max_igs_tuples;
                delete[] local_mdfs_output->dids;
            }
            delete local_mdfs_output;
        }

        delete[] data;
        delete generator;
    }

    if (H != nullptr) {
        delete[] H;
    }
    if (I_lower != nullptr) {
        delete[] I_lower;
    }
    delete[] decision;
}

typedef void (*MdfsImpl) (
    const MDFSInfo& mdfs_info,
    RawData* raw_data,
    std::unique_ptr<const DiscretizationInfo> dfi,
    MDFSOutput& out
);

const MdfsImpl mdfs[] = {
    scalarMDFS<2, 1>,
    scalarMDFS<2, 2>,
    scalarMDFS<2, 3>,
    scalarMDFS<2, 4>,
    scalarMDFS<2, 5>
};

const MdfsImpl mdfsNoDecision[] = {
    scalarMDFS<1, 1>,
    scalarMDFS<1, 2>,
    scalarMDFS<1, 3>,
    scalarMDFS<1, 4>,
    scalarMDFS<1, 5>
};

#endif
