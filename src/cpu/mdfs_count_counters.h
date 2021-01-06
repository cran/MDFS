#ifndef MDFS_COUNT_COUNTERS_H
#define MDFS_COUNT_COUNTERS_H

#include <cstddef>
#include <cstdint>
#include <cstring>

template <uint8_t n_dimensions>
inline void count_counters(
    const uint8_t *data,
    const uint8_t *decision,
    const size_t n_objects,
    const size_t n_classes,

    const size_t* tuple,

    float* counters,
    const size_t n_cubes,

    const float p0,
    const float p1,
    const size_t* d
) {
    std::memset(counters, 0, sizeof(float) * n_cubes * 2);

    for (size_t o = 0; o < n_objects; ++o) {
        size_t bucket = data[tuple[0] * n_objects + o];
        if (n_dimensions >= 2) {
            bucket += n_classes * data[tuple[1] * n_objects + o];
        }
        if (n_dimensions >= 3) {
            bucket += d[0] * data[tuple[2] * n_objects + o];
        }
        if (n_dimensions >= 4) {
            bucket += d[1] * data[tuple[3] * n_objects + o];
        }
        if (n_dimensions >= 5) {
            bucket += d[2] * data[tuple[4] * n_objects + o];
        }

        size_t dec = decision[o];
        counters[dec * n_cubes + bucket] += 1.0f;
    }

    for (size_t c = 0; c < n_cubes; ++c) {
        counters[0 * n_cubes + c] += p0;
        counters[1 * n_cubes + c] += p1;
    }
}

#endif
