#ifndef MDFS_CPU_KERNEL_H
#define MDFS_CPU_KERNEL_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "entropy.h"
#include "mdfs_count_counters.h"
#include "mdfs_reduce_counters.h"


// only 1 and 2 decision classes are supported
template <uint8_t n_decision_classes, uint8_t n_dimensions>
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

    const float p[n_decision_classes],
    const float total,  // total of all counters; used only in no decision mode
    const size_t* d,

    const float* H,  // entropies of single variables

    float igs[n_dimensions],
    void*
) {
    count_counters<n_decision_classes, n_dimensions>(data, decision, n_objects, n_classes, tuple, counters, n_cubes, p, d);

    // H(Y|{X_i}) conditional entropy of decision given all tuple vars
    float H_Y_given_all = 0.0f;
    if (n_decision_classes > 1) {
        H_Y_given_all = conditional_entropy<n_decision_classes>(n_cubes, counters);
    }
    // H({X_i}) (plain) entropy of all tuple vars
    float H_all = 0.0f;
    if (n_decision_classes == 1) {
        H_all = entropy(total, n_cubes, counters);
    }

    if (n_decision_classes == 1 && n_dimensions == 2) {  // optimised version
        // I(X_0;X_1) mutual information
        igs[0] = igs[1] = H[tuple[0]] + H[tuple[1]] - H_all;
        return;
    }

    for (size_t v = 0, stride = 1; v < n_dimensions; ++v, stride *= n_classes) {
        std::memset(counters_reduced, 0, sizeof(float) * n_cubes_reduced * n_decision_classes);
        reduce_counters(n_classes, n_cubes, counters, counters_reduced, stride);
        if (n_decision_classes > 1) {
            reduce_counters(n_classes, n_cubes, counters + n_cubes, counters_reduced + n_cubes_reduced, stride);
            // H(Y|{X_i!=X_k}) conditional entropy of decision given all tuple vars except the current one (X_k)
            float H_Y_given_all_except_current = conditional_entropy<n_decision_classes>(n_cubes_reduced, counters_reduced);
            // I(Y;X_k | {X_i!=X_k}) mutual information of decision and the current var given all the other tuple vars
            igs[v] = H_Y_given_all_except_current - H_Y_given_all;
        } else {
            // TODO: this can be optimised - simple entropies can be precomputed;
            // in 2D this means both, here TBD
            // H({X_i!=X_k}) (plain) entropy of all tuple vars except the current one
            float H_all_except_current = entropy(total, n_cubes_reduced, counters_reduced);
            float p_local = p[0] * n_cubes_reduced;
            count_counters<1, 1>(data, nullptr, n_objects, 0, tuple+v, counters_reduced, n_classes, &p_local, nullptr);
            // H(X_k) (plain) entropy of the current var
            float H_current = entropy(total, n_classes, counters_reduced);
            // I(X_k;{X_i!=X_k}) mutual information
            igs[v] = H_current + H_all_except_current - H_all;
        }
    }
}

#endif
