#ifndef DISCRETIZE_H
#define DISCRETIZE_H

#include "datafile.h"
#include "discretizedfile.h"

#include <cstddef>
#include <cstdint>

class DiscretizationInfo {
public:
    DiscretizationInfo(uint32_t seed, int disc, int div, double range);

    uint32_t seed;
    int discretizations;
    int divisions;
    double range;
};

void discretizeFile(DataFile *in,
                    DiscretizedFile *out,
                    DiscretizationInfo info);

#endif
