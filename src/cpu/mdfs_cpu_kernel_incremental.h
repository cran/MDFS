#ifndef MDFS_CPU_KERNEL_INCREMENTAL_H
#define MDFS_CPU_KERNEL_INCREMENTAL_H

#include <cstddef>
#include <cstdint>

#include "entropy.h"
#include "mdfs_count_counters.h"


// only 1 and 2 decision classes are supported
// and only 1 and 2 dimensions
template <uint8_t n_decision_classes, uint8_t n_dimensions>
inline void process_tuple_incremental(
    const uint8_t *data,
    const uint8_t *decision,
    const size_t n_objects,
    const size_t n_classes,

    const size_t* tuple,

    float* counters,
    const size_t n_cubes,

    const float p[n_decision_classes],
    const float total,  // total of all counters; used only in no decision mode
    const size_t* d,

    const float H_Y,  // H(Y) (plain) entropy of decision
    const float* I_lower,  // 2D: i-indexed array of I(Y;X_i) mutual information of decision and the i-th var
                           // 2D no decision: i-indexed array of H(X_i) (plain) entropy of the i-th var

    float igs[n_dimensions],
    void*
) {
    count_counters<n_decision_classes, n_dimensions>(data, decision, n_objects, n_classes, tuple, counters, n_cubes, p, d);

    if (n_decision_classes > 1) {
        // H(Y|{X_i}) conditional entropy of decision given all tuple vars
        const float H_Y_given_all = conditional_entropy<n_decision_classes>(n_cubes, counters);

        if (n_dimensions == 1) {
            // I(Y;X_0) mutual information of decision and the current var
            igs[0] = H_Y - H_Y_given_all;
        } else if (n_dimensions == 2) {
            // I(Y;X_0 | X_1) mutual information of decision and var 0 given var 1
            // I(Y;X_1) == I_lower[tuple[1]]
            igs[0] = H_Y - I_lower[tuple[1]] - H_Y_given_all;
            // I(Y;X_1 | X_0) mutual information of decision and var 1 given var 0
            // I(Y;X_0) == I_lower[tuple[0]]
            igs[1] = H_Y - I_lower[tuple[0]] - H_Y_given_all;
        }
        // TODO: more dims
    } else {
        // H({X_i}) (plain) entropy of all tuple vars
        float H_all = entropy(total, n_cubes, counters);

        if (n_dimensions == 1) {
            // H(X_0) (plain) entropy of the current var
            igs[0] = H_all;
        } else if (n_dimensions == 2) {
            // I(X_0;X_1) mutual information
            igs[0] = igs[1] = I_lower[tuple[0]] + I_lower[tuple[1]] - H_all;
        }
        // TODO: more dims
    }
}

#endif
