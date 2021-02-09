#ifndef COMMON_H
#define COMMON_H

#include <cstddef>
#include <list>
#include <vector>
#include <map>
#include <tuple>


class MDFSInfo {
public:
    size_t dimensions;
    size_t divisions;
    size_t discretizations;
    float pseudo;
    float ig_thr;
    int* interesting_vars;  // has to be sorted
    size_t interesting_vars_count;
    bool require_all_vars;
    double* I_lower;

    MDFSInfo(
        size_t dimensions,
        size_t divisions,
        size_t discretizations,
        float pseudo,
        float ig_thr,
        int* interesting_vars,
        size_t interesting_vars_count,
        bool require_all_vars,
        double* I_lower
    );
};


class TupleGenerator {
    size_t* nextTuple;
    const size_t n_dimensions;
    const size_t n_variables;
    const std::vector<size_t> interesting_vars;

public:
    TupleGenerator(size_t n_dimensions, size_t n_variables);
    TupleGenerator(size_t n_dimensions, const std::vector<size_t>& interesting_vars);
    ~TupleGenerator();

    bool hasNext() const;
    void next(size_t* out);
};


enum class MDFSOutputType { MaxIGs, MinIGs, MatchingTuples, AllTuples };

class MDFSOutput {
public:
    int *max_igs_tuples;
    int *dids;
    union {
        std::vector<float> *max_igs;
        std::map<std::tuple<std::vector<size_t>, size_t>, std::tuple<float, size_t>> *tuples;
        std::vector<float> *all_tuples;
    };

    MDFSOutput(MDFSOutputType type, size_t n_dimensions, size_t variable_count);
    ~MDFSOutput();

    const MDFSOutputType type;
    const size_t n_dimensions;
    const size_t n_variables;

    void setMaxIGsTuples(int *tuples, int *dids);
    void updateMaxIG(const size_t* tuple, float *igs, size_t discretization_id);
    void updateMinIG(const size_t* tuple, float *igs, size_t discretization_id);
    void copyMaxIGsAsDouble(double *copy) const;
    void addTuple(size_t i, float ig, size_t discretization_id, const size_t* vt);
    void updateAllTuplesIG(const size_t* tuple, float *igs, size_t discretization_id);
    size_t getMatchingTuplesCount() const;
    void copyMatchingTuples(int* matching_tuples_vars, double* IGs, int* matching_tuples) const;
    void copyAllTuples(int* matching_tuples_vars, double* IGs, int* matching_tuples) const;
    void copyAllTuplesMatrix(double* out_matrix) const;
};

#endif
