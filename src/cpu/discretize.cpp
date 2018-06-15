#include "discretize.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

DiscretizationInfo::DiscretizationInfo(uint32_t seed, int disc, int div, double range)
        : seed(seed), discretizations(disc), divisions(div), range(range) {}

void discretize(uint32_t seed,
                uint32_t discretization_id,
                uint32_t feature_id,
                std::size_t divisions,
                std::size_t length,
                const double* in_data,
                const std::vector<double>& sorted_in_data,
                uint8_t* out_data,
                double range) {
    double* thresholds = new double[divisions];
    {
        double sum = 0.0f;
        {
            std::mt19937 seed_random_generator0(seed);
            std::mt19937 seed_random_generator1(seed_random_generator0() ^ discretization_id);
            std::mt19937 random_generator(seed_random_generator1() ^ feature_id);
            // E(X) = (a + b) / 2 = (1 - range + 1 + range) / 2 = 1
            std::uniform_real_distribution<double> uniform_range(1.0f - range, 1.0f + range);

            for (std::size_t d = 0; d < divisions; ++d) {
                thresholds[d] = uniform_range(random_generator);
                sum += thresholds[d];
            }

            sum += uniform_range(random_generator);
        }

        std::size_t done = 0;
        const double length_step = static_cast<double>(length) / sum;
        // thresholds are converted from arbitrary space
        // to real data space (from sorted_in_data)
        for (std::size_t d = 0; d < divisions; ++d) {
            done += std::lround(thresholds[d] * length_step);
            if (done >= length)
                done = length - 1;
            thresholds[d] = sorted_in_data[done];
        }
    }

    for (std::size_t i = 0; i < length; ++i) {
        out_data[i] = 0;
        // out_data[i] is incremented every time in_data[i] is above given threashold
        for (std::size_t d = 0; d < divisions; ++d) {
            out_data[i] += in_data[i] > thresholds[d];
        }
    }

    delete[] thresholds;
}

void discretizeVar(DataFile* in,
                   DiscretizedFile* out,
                   int var,
                   DiscretizationInfo info) {
    const double* in_data = in->getV(var);
    std::vector<double> sorted_in_data(in_data, in_data + in->info.object_count);
    std::sort(sorted_in_data.begin(), sorted_in_data.end());
    for (int d = 0; d < info.discretizations; ++d) {
        discretize(info.seed, d, var, info.divisions, in->info.object_count, in_data, sorted_in_data, out->getVD(var, d), info.range);
    }
}

void discretizeFile(DataFile* in,
                    DiscretizedFile* out,
                    DiscretizationInfo info) {
    out->decision = in->decision;
    for (int v = 0; v < in->info.variable_count; ++v) {
        discretizeVar(in, out, v, info);
    }
}
