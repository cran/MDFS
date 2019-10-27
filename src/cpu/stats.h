#ifndef STATS_H
#define STATS_H

#include <cstddef>
#include <cmath>

inline void reduceCounter(size_t n_classes, size_t n_cubes, float *in, float *out, size_t rstride) {
    for (size_t c = 0, v = 0; c < n_cubes; c += rstride * n_classes) {
        for (size_t s = 0; s < rstride; ++s, ++v) {
            for (size_t d = 0; d < n_classes; ++d) {
                out[v] += in[c + s + (d * rstride)];
            }
        }
    }
}

inline float informationGain(size_t length, float *c0, float *c1) {
    float ig = 0.0f;

    for (size_t i = 0; i < length; ++i) {
        const float c = c0[i] + c1[i];
        ig += (c0[i]) * std::log2(c0[i]/c);
        ig += (c1[i]) * std::log2(c1[i]/c);
    }
    return ig;
}

#endif
