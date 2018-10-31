#ifndef STATS_H
#define STATS_H

void reduceCounter(int divisions, float *in, int dimensions, float *out, int stride);
float informationGain(int length, float *c0, float *c1);

#endif
