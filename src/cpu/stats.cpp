#include <cmath>
#include <cstring>

void reduceCounter(int divisions, float *in, int dimensions, float *out, int stride) {
    divisions += 1;
    const int cubes = std::pow(divisions, dimensions);
    const int rstride = std::pow(divisions, (stride - 1));
    int v = 0;

    std::memset(out, 0, sizeof(float) * std::pow(divisions, (dimensions - 1)));

    for (int c = 0; c < cubes; c += rstride * divisions) {
        for (int s = 0; s < rstride; ++s, ++v) {
            for (int d = 0; d < divisions; ++d) {
                out[v] += in[c + s + (d * rstride)];
            }
        }
    }
}

float informationGain(int length, float *c0, float *c1) {
    float ig = 0.0f;

    for (int i = 0; i < length; ++i) {
        float c = c0[i] + c1[i];

        if (c0[i] != 0.0f) {
            ig += (c0[i]) * std::log2(c0[i]/c);
        }

        if (c1[i] != 0.0f) {
            ig += (c1[i]) * std::log2(c1[i]/c);
        }
    }
    return ig;
}
