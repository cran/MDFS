#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "mdfs_scalar.h"
#include "stats.h"

#define CONTAINS(x, y) (std::find((x).begin(), (x).end(), (y)) != (x).end())

void scalarMDFS(const AlgInfo& algorithm_info,
                DiscretizedFile* in,
                MDFSOutput& out) {
    const int cubes = std::pow(algorithm_info.divisions + 1, algorithm_info.dimensions);
    const int cd = cubes / (algorithm_info.divisions + 1);

    const int c0 = in->c0();
    const int c1 = in->c1();

    const float cmin = std::min(c0, c1);

    const float p0 = ((float)c0 / cmin) * algorithm_info.pseudo;
    const float p1 = ((float)c1 / cmin) * algorithm_info.pseudo;

    TupleGenerator tg(algorithm_info.dimensions, in->info.variable_count);

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        float* ig = new float[algorithm_info.dimensions * algorithm_info.discretizations];
        float* dig = new float[algorithm_info.dimensions];
        float* counters = new float[2 * cubes];
        float* reduced = new float[2 * cd];

        std::unique_ptr<Tuple> v;

        bool hasWorkToDo = true;

        do {
            #ifdef _OPENMP
            #pragma omp critical (GetTheNextTuple)
            #endif
            if (!tg.hasNext()) { // we have to synchronize because TupleGenerator is not omp-thread-safe and it has to be shared
                hasWorkToDo = false;
                // one does not simply leave the OpenMP's critical section
                // no, seriously, it is programmed such that you cannot use break/continue here
                // hence we break later ¯\_(ツ)_/¯
            } else {
                v = tg.next();
            }

            if (!hasWorkToDo) {
                break; // this is right where we break, OpenMP :-)
            }

            std::list<int> current_interesting_vars;
            std::set_intersection(
                v->begin(), v->end(),
                algorithm_info.interesting_vars, algorithm_info.interesting_vars + algorithm_info.interesting_vars_count,
                std::back_inserter(current_interesting_vars));

            if (algorithm_info.require_all_vars) {
                if (algorithm_info.interesting_vars_count != current_interesting_vars.size()) {
                    continue;
                }
            } else {
                if ((algorithm_info.interesting_vars_count > 0) && current_interesting_vars.empty()) {
                    continue;
                }
            }

            for (int d = 0; d < in->info.discretizations; d++) {
                std::memset(counters, 0, sizeof(float) * cubes * 2);
                for (int o = 0; o < in->info.object_count; ++o) {
                    int bucket = 0;
                    for (int vv = algorithm_info.dimensions - 1; vv >= 0; --vv) {
                        bucket *= algorithm_info.divisions + 1;
                        bucket += in->getVD(v->get(vv), d)[o];
                    }

                    int dec = in->decision[o];
                    counters[dec * cubes + bucket] += 1.0f;
                }
                for (int b = 0; b < cubes; ++b) {
                    counters[0 * cubes + b] += p0;
                    counters[1 * cubes + b] += p1;
                }

                const float ign = informationGain(cubes, counters, counters + cubes);

                for (int vv = 0; vv < algorithm_info.dimensions; ++vv) {
                    reduceCounter(algorithm_info.divisions, counters, algorithm_info.dimensions, reduced, vv + 1);
                    reduceCounter(algorithm_info.divisions, counters + cubes, algorithm_info.dimensions, reduced + cd, vv + 1);
                    float igg = informationGain(cd, reduced, reduced + cd);
                    ig[vv * algorithm_info.discretizations + d] = ign - igg;
                }
            }

            // reduce for each variable to max across discretizations
            for (int vv = 0; vv < algorithm_info.dimensions; ++vv) {
                dig[vv] = *std::max_element(ig + vv * algorithm_info.discretizations, ig + vv * algorithm_info.discretizations + algorithm_info.discretizations);
            }

            #ifdef _OPENMP
            #pragma omp critical (SetOutput)
            #endif
            switch (out.type) { // we have to synchronize because MDFSOutput is not omp-thread-safe and it has to be shared
                case MDFSOutputType::MaxIGs:
                    out.updateMaxIG(*v, dig);
                    break;
                case MDFSOutputType::MatchingTuples:
                    for (int vv = 0; vv < algorithm_info.dimensions; ++vv) {
                        if (dig[vv] > algorithm_info.ig_thr && (current_interesting_vars.empty() || CONTAINS(current_interesting_vars, v->get(vv)))) {
                            out.addTuple(v->get(vv), dig[vv], *v);
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
