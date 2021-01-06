#ifndef MDFS_CPU_KERNEL_INCREMENTAL_H
#define MDFS_CPU_KERNEL_INCREMENTAL_H

#include <cstddef>
#include <cstdint>

#include "entropy.h"
#include "mdfs_count_counters.h"

template <uint8_t n_dimensions>
inline void process_tuple_incremental(
    const uint8_t *data,
    const uint8_t *decision,
    const size_t n_objects,
    const size_t n_classes,

    const size_t* tuple,

    float* counters,
    const size_t n_cubes,

    const float p0,
    const float p1,
    const size_t* d,

    const float H_Y,
    const float* I_lower,

    float* igs,
    void*
) {
    count_counters<n_dimensions>(data, decision, n_objects, n_classes, tuple, counters, n_cubes, p0, p1, d);

    const float H_total = entropy(n_cubes, counters, counters + n_cubes);

    if (n_dimensions == 1) {
        igs[0] = H_Y - H_total;
    } else if (n_dimensions == 2) {
        igs[0] = H_Y - I_lower[tuple[1]] - H_total;
        igs[1] = H_Y - I_lower[tuple[0]] - H_total;
    }
    // FIXME: more dims
}

#endif
