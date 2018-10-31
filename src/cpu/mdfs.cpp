#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "common.h"
#include "mdfs.h"
#include "stats.h"

#define CONTAINS(x, y) (std::find((x).begin(), (x).end(), (y)) != (x).end())

void scalarMDFS(
    const MDFSInfo& mdfs_info,
    DataSet* dataset,
    MDFSOutput& out
) {
    const int num_of_cubes = std::pow(mdfs_info.divisions + 1, mdfs_info.dimensions);
    const int num_of_cubes_reduced = std::pow(mdfs_info.divisions + 1, mdfs_info.dimensions - 1);

    const float cmin = std::min(dataset->c0(), dataset->c1());

    const float p0 = ((float)dataset->c0() / cmin) * mdfs_info.pseudo;
    const float p1 = ((float)dataset->c1() / cmin) * mdfs_info.pseudo;

    TupleGenerator generator(mdfs_info.dimensions, dataset->info->variable_count);

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        float* ig = new float[mdfs_info.dimensions * mdfs_info.discretizations];
        float* dig = new float[mdfs_info.dimensions];
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

            if (mdfs_info.require_all_vars) {
                if (mdfs_info.interesting_vars_count != current_interesting_vars.size()) {
                    continue;
                }
            } else {
                if ((mdfs_info.interesting_vars_count > 0) && current_interesting_vars.empty()) {
                    continue;
                }
            }

            // d - iterates over discretizations (repeating the run)
            for (int d = 0; d < dataset->info->discretizations; d++) {
                std::memset(counters, 0, sizeof(float) * num_of_cubes * 2);

                // o - iterates over objects (from dataset)
                for (int o = 0; o < dataset->info->object_count; ++o) {
                    int bucket = 0;

                    // v - iterates over variables (from variable_set)
                    for (int v = mdfs_info.dimensions - 1; v >= 0; --v) {
                        bucket *= mdfs_info.divisions + 1;
                        bucket += dataset->getDiscretizationData(variable_set->get(v), d)[o];
                    }

                    int dec = dataset->decision[o];
                    counters[dec * num_of_cubes + bucket] += 1.0f;
                }

                // c - iterates over counters (for each cube)
                for (int c = 0; c < num_of_cubes; ++c) {
                    counters[0 * num_of_cubes + c] += p0;
                    counters[1 * num_of_cubes + c] += p1;
                }

                const float ign = informationGain(num_of_cubes, counters, counters + num_of_cubes);

                // v - iterates over variables (from variable_set)
                for (int v = 0; v < mdfs_info.dimensions; ++v) {
                    reduceCounter(mdfs_info.divisions, counters, mdfs_info.dimensions, reduced, v + 1);
                    reduceCounter(mdfs_info.divisions, counters + num_of_cubes, mdfs_info.dimensions, reduced + num_of_cubes_reduced, v + 1);
                    float igg = informationGain(num_of_cubes_reduced, reduced, reduced + num_of_cubes_reduced);
                    ig[v * mdfs_info.discretizations + d] = ign - igg;
                }
            }

            // reduce for each variable to max across discretizations
            for (int v = 0; v < mdfs_info.dimensions; ++v) {
                dig[v] = *std::max_element(ig + v * mdfs_info.discretizations, ig + v * mdfs_info.discretizations + mdfs_info.discretizations);
            }

            #ifdef _OPENMP
            #pragma omp critical (SetOutput)
            #endif
            switch (out.type) { // we have to synchronize because MDFSOutput is not omp-thread-safe and it has to be shared
                case MDFSOutputType::MaxIGs:
                    out.updateMaxIG(*variable_set, dig);
                    break;

                case MDFSOutputType::MatchingTuples:

                    // v - iterates over variables (from variable set)
                    for (int v = 0; v < mdfs_info.dimensions; ++v) {
                        if (dig[v] > mdfs_info.ig_thr && (current_interesting_vars.empty() || CONTAINS(current_interesting_vars, variable_set->get(v)))) {
                            out.addTuple(variable_set->get(v), dig[v], *variable_set);
                        }
                    }
                    break;
            }
        } while (true);

        delete[] reduced;
        delete[] counters;
        delete[] dig;
        delete[] ig;
    }
}
