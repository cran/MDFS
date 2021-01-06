#ifndef MDFS_CPU_KERNEL_H
#define MDFS_CPU_KERNEL_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "entropy.h"
#include "mdfs_count_counters.h"
#include "mdfs_reduce_counters.h"

template <uint8_t n_dimensions>
inline void process_tuple(
    const uint8_t *data,
    const uint8_t *decision,
    const size_t n_objects,
    const size_t n_classes,

    const size_t* tuple,

    float* counters,
    float* counters_reduced,
    const size_t n_cubes,
    const size_t n_cubes_reduced,

    const float p0,
    const float p1,
    const size_t* d,

    float* igs,
    void*
) {
    count_counters<n_dimensions>(data, decision, n_objects, n_classes, tuple, counters, n_cubes, p0, p1, d);

    const float H_total = entropy(n_cubes, counters, counters + n_cubes);

    for (size_t v = 0, stride = 1; v < n_dimensions; ++v, stride *= n_classes) {
        std::memset(counters_reduced, 0, sizeof(float) * n_cubes_reduced * 2);
        reduce_counters(n_classes, n_cubes, counters, counters_reduced, stride);
        reduce_counters(n_classes, n_cubes, counters + n_cubes, counters_reduced + n_cubes_reduced, stride);
        float H_reduced = entropy(n_cubes_reduced, counters_reduced, counters_reduced + n_cubes_reduced);
        igs[v] = H_reduced - H_total;
    }
}

#endif
