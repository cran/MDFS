#ifndef MDFS
#define MDFS

#include "dataset.h"
#include "common.h"

#include <algorithm>
#include <cstring>

#include "mdfs.h"
#include "stats.h"

#define CONTAINS(x, y) (std::find((x).begin(), (x).end(), (y)) != (x).end())

template <uint8_t n_dimensions>
void scalarMDFS(
    const MDFSInfo& mdfs_info,
    DataSet* dataset,
    MDFSOutput& out
) {
    const size_t n_classes = mdfs_info.divisions + 1;
    const size_t num_of_cubes = std::pow(n_classes, n_dimensions);
    const size_t num_of_cubes_reduced = std::pow(n_classes, n_dimensions - 1);

    const auto c0 = dataset->info->object_count_per_class[0];
    const auto c1 = dataset->info->object_count_per_class[1];
    const float cmin = std::min(c0, c1);

    const auto d1 = n_classes;
    const auto d2 = d1*n_classes;
    const auto d3 = d2*n_classes;
    const auto d4 = d3*n_classes;

    const float p0 = c0 / cmin * mdfs_info.pseudo;
    const float p1 = c1 / cmin * mdfs_info.pseudo;

    TupleGenerator generator(n_dimensions, dataset->info->variable_count);

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        float* ig = new float[n_dimensions * mdfs_info.discretizations];
        float* dig = new float[n_dimensions];
        int* did = new int[n_dimensions];
        float* counters = new float[2 * num_of_cubes];
        float* reduced = new float[2 * num_of_cubes_reduced];

        std::unique_ptr<Tuple> variable_set;

        bool hasWorkToDo = true;

        do {
            #ifdef _OPENMP
            #pragma omp critical (GetTheNextTuple)
            #endif

            if (!generator.hasNext()) { // we have to synchronize because TupleGenerator is not omp-thread-safe and it has to be shared
                hasWorkToDo = false;
                // one does not simply leave the OpenMP's critical section
                // no, seriously, it is programmed such that you cannot use break/continue here
                // hence we break later ¯\_(ツ)_/¯
            } else {
                variable_set = generator.next();
            }

            if (!hasWorkToDo) {
                break; // this is right where we break, OpenMP :-)
            }

            std::list<int> current_interesting_vars;
            std::set_intersection(
                variable_set->begin(), variable_set->end(),
                mdfs_info.interesting_vars, mdfs_info.interesting_vars + mdfs_info.interesting_vars_count,
                std::back_inserter(current_interesting_vars));

            if (mdfs_info.interesting_vars_count) {
                if (mdfs_info.require_all_vars) {
                    if (mdfs_info.dimensions != current_interesting_vars.size()) {
                        continue;
                    }
                } else {
                    if (current_interesting_vars.empty()) {
                        continue;
                    }
                }
            }

            // d - iterates over discretizations (repeating the run)
            for (size_t d = 0; d < dataset->info->discretizations; d++) {
                std::memset(counters, 0, sizeof(float) * num_of_cubes * 2);

                // o - iterates over objects (from dataset)
                for (size_t o = 0; o < dataset->info->object_count; ++o) {
                    size_t bucket = dataset->getDiscretizationData(variable_set->get(0), d)[o];
                    if (n_dimensions >= 2) {
                        bucket += d1 * dataset->getDiscretizationData(variable_set->get(1), d)[o];
                    }
                    if (n_dimensions >= 3) {
                        bucket += d2 * dataset->getDiscretizationData(variable_set->get(2), d)[o];
                    }
                    if (n_dimensions >= 4) {
                        bucket += d3 * dataset->getDiscretizationData(variable_set->get(3), d)[o];
                    }
                    if (n_dimensions >= 5) {
                        bucket += d4 * dataset->getDiscretizationData(variable_set->get(4), d)[o];
                    }

                    size_t dec = dataset->decision[o];
                    counters[dec * num_of_cubes + bucket] += 1.0f;
                }

                // c - iterates over counters (for each cube)
                for (size_t c = 0; c < num_of_cubes; ++c) {
                    counters[0 * num_of_cubes + c] += p0;
                    counters[1 * num_of_cubes + c] += p1;
                }

                const float ign = informationGain(num_of_cubes, counters, counters + num_of_cubes);

                // v - iterates over variables (from variable_set)
                for (size_t v = 0, stride = 1; v < n_dimensions; ++v, stride *= n_classes) {
                    std::memset(reduced, 0, sizeof(float) * num_of_cubes_reduced * 2);
                    reduceCounter(n_classes, num_of_cubes, counters, reduced, stride);
                    reduceCounter(n_classes, num_of_cubes, counters + num_of_cubes, reduced + num_of_cubes_reduced, stride);
                    float igg = informationGain(num_of_cubes_reduced, reduced, reduced + num_of_cubes_reduced);
                    ig[v * mdfs_info.discretizations + d] = ign - igg;
                }
            }

            // reduce for each variable to max across discretizations
            for (size_t v = 0; v < n_dimensions; ++v) {
                auto begin = ig + v * mdfs_info.discretizations;
                auto maxIt = std::max_element(begin, begin + mdfs_info.discretizations);
                dig[v] = *maxIt;
                did[v] = std::distance(begin, maxIt);
            }

            #ifdef _OPENMP
            #pragma omp critical (SetOutput)
            #endif
            switch (out.type) { // we have to synchronize because MDFSOutput is not omp-thread-safe and it has to be shared
                case MDFSOutputType::MaxIGs:
                    out.updateMaxIG(*variable_set, dig, did);
                    break;

                case MDFSOutputType::MatchingTuples:

                    // v - iterates over variables (from variable set)
                    for (size_t v = 0; v < n_dimensions; ++v) {
                        if (dig[v] > mdfs_info.ig_thr && (current_interesting_vars.empty() || CONTAINS(current_interesting_vars, variable_set->get(v)))) {
                            out.addTuple(variable_set->get(v), dig[v], *variable_set);
                        }
                    }
                    break;
            }
        } while (true);

        delete[] reduced;
        delete[] counters;
        delete[] did;
        delete[] dig;
        delete[] ig;
    }
}

typedef void (*MdfsImpl) (
    const MDFSInfo& mdfs_info,
    DataSet* dataset,
    MDFSOutput& out
);

const MdfsImpl mdfs[] = {
    scalarMDFS<1>,
    scalarMDFS<2>,
    scalarMDFS<3>,
    scalarMDFS<4>,
    scalarMDFS<5>
};

#endif
