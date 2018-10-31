#ifndef COMMON_H
#define COMMON_H

#include <cstddef>
#include <list>
#include <vector>
#include <memory>

#include "dataset.h"

class MDFSInfo {
public:
    int dimensions;
    int divisions;
    int discretizations;
    float pseudo;
    float ig_thr;
    int* interesting_vars;
    size_t interesting_vars_count;
    bool require_all_vars;

    MDFSInfo(
        int dimensions,
        int divisions,
        int discretizations,
        float pseudo,
        float ig_thr,
        int* interesting_vars,
        size_t interesting_vars_count,
        bool require_all_vars
    );
};

class TupleGenerator;

class Tuple {
    std::vector<int> combination;

    // Initializes first tuple: (0, ..., 0) 
    Tuple(int dimensions);

    // Initializes consequent tuple to previous: (0, ..., x+1, ...)
    Tuple(const Tuple& previous, int variable_count);

public:
    int dimensions() const;
    int get(int index) const;
    std::vector<int>::const_iterator begin() const;
    std::vector<int>::const_iterator end() const;

    friend TupleGenerator;
};

class TupleGenerator {
    std::unique_ptr<Tuple> nextTuple;
    const int variable_count;

public:
    TupleGenerator(int dimensions, int variable_count);

    bool hasNext() const;
    std::unique_ptr<Tuple> next();
};

class MDFSTuple {
    int i;
    float ig;
    std::vector<int> v;

public:
    MDFSTuple(int i, float ig, std::vector<int>&& v);

    int getVariable() const;
    float getIG() const;
    int get(int i) const;
    size_t getDim() const;
};

enum class MDFSOutputType { MaxIGs, MatchingTuples };

class MDFSOutput {
    int *max_igs_tuples;
    union {
        std::vector<float> *max_igs;
        std::list<MDFSTuple> *tuples;
    };

public:
    MDFSOutput(MDFSOutputType type, int variable_count);
    ~MDFSOutput();

    const MDFSOutputType type;

    void setMaxIGsTuples(int *tuples);
    void updateMaxIG(const Tuple& tuple, float *digs);
    void copyMaxIGsAsDouble(double *copy) const;
    void addTuple(int i, float ig, const Tuple& vt);
    size_t getMatchingTuplesCount() const;
    void copyMatchingTuples(int* matching_tuples_vars, double* IGs, int* matching_tuples) const;
};

#endif
