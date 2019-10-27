#ifndef COMMON_H
#define COMMON_H

#include <cstddef>
#include <list>
#include <vector>
#include <memory>

#include "dataset.h"

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

    MDFSInfo(
        size_t dimensions,
        size_t divisions,
        size_t discretizations,
        float pseudo,
        float ig_thr,
        int* interesting_vars,
        size_t interesting_vars_count,
        bool require_all_vars
    );
};

class TupleGenerator;

class Tuple {
    std::vector<size_t> combination;

    // Initializes first tuple: (0, ..., 0)
    Tuple(size_t dimensions);

    // Initializes consequent tuple to previous: (0, ..., x+1, ...)
    Tuple(const Tuple& previous, size_t variable_count);

public:
    inline size_t dimensions() const {
        return this->combination.size() - 1;
    }
    inline size_t get(size_t index) const {
        return this->combination[index + 1];
    }
    inline std::vector<size_t>::const_iterator begin() const {
        return this->combination.begin() + 1;
    }
    inline std::vector<size_t>::const_iterator end() const {
        return this->combination.end();
    }

    friend TupleGenerator;
};

class TupleGenerator {
    std::unique_ptr<Tuple> nextTuple;
    const size_t variable_count;

public:
    TupleGenerator(size_t dimensions, size_t variable_count);

    bool hasNext() const;
    std::unique_ptr<Tuple> next();
};

class MDFSTuple {
    size_t i;
    float ig;
    std::vector<int> v;

public:
    MDFSTuple(size_t i, float ig, std::vector<int>&& v);

    size_t getVariable() const;
    float getIG() const;
    size_t get(size_t i) const;
    size_t getDim() const;
};

enum class MDFSOutputType { MaxIGs, MatchingTuples };

class MDFSOutput {
    int *max_igs_tuples;
    int *dids;
    union {
        std::vector<float> *max_igs;
        std::list<MDFSTuple> *tuples;
    };

public:
    MDFSOutput(MDFSOutputType type, size_t variable_count);
    ~MDFSOutput();

    const MDFSOutputType type;

    void setMaxIGsTuples(int *tuples, int *dids);
    void updateMaxIG(const Tuple& tuple, float *digs, int *dids);
    void copyMaxIGsAsDouble(double *copy) const;
    void addTuple(size_t i, float ig, const Tuple& vt);
    size_t getMatchingTuplesCount() const;
    void copyMatchingTuples(int* matching_tuples_vars, double* IGs, int* matching_tuples) const;
};

#endif
