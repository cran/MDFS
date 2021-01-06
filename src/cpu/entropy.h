#ifndef ENTROPY_H
#define ENTROPY_H

#include <cstddef>
#include <cmath>

inline float entropy(size_t length, const float *c0, const float *c1) {
    float ig = 0.0f;

    for (size_t i = 0; i < length; ++i) {
        const float c = c0[i] + c1[i];
        ig -= (c0[i]) * std::log2(c0[i]/c);
        ig -= (c1[i]) * std::log2(c1[i]/c);
    }

    return ig;
}

#endif
