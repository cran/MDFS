#include "mdfs_common.h"

#include <algorithm>

// TODO: find some more elegant solution :-)
#ifdef R_PKG
#include <R.h>
#define printf Rprintf
#else
#include <cstdio>
#endif

TupleGenerator::TupleGenerator(int dimensions, int variable_count) : nextTuple(new Tuple(dimensions)), variable_count(variable_count) { }

bool TupleGenerator::hasNext() const {
    return nextTuple->combination[0] == 0;
}

std::unique_ptr<Tuple> TupleGenerator::next() {
    std::unique_ptr<Tuple> thisTuple = std::move(nextTuple);
    nextTuple.reset(new Tuple(*thisTuple, variable_count));
    return thisTuple;
}

Tuple::Tuple(int dimensions) : combination(dimensions + 1) {
    combination[0] = 0;
    for (int d = 1; d <= dimensions; ++d)
        combination[d] = d - 1;
}

Tuple::Tuple(const Tuple& previous, int variable_count) : combination(previous.combination) {
    const int dimensions = this->dimensions();
    int d = dimensions;
    for (; d >= 0; --d) {
        ++combination[d];
        if (combination[d] < variable_count - (dimensions - d))
            break;
    }
    for (++d; d <= dimensions; ++d) {
        combination[d] = combination[d-1]+1;
    }
}

int Tuple::dimensions() const {
    return combination.size() - 1;
}

int Tuple::get(int i) const {
    return combination[i + 1];
}

std::vector<int>::const_iterator Tuple::begin() const {
    return combination.begin() + 1;
}

std::vector<int>::const_iterator Tuple::end() const {
    return combination.end();
}

MDFSTuple::MDFSTuple(int i, float ig, std::vector<int>&& v)
        : i(i), ig(ig), v(v) {}

int MDFSTuple::getVar() const {
    return i;
}

float MDFSTuple::getIG() const {
    return ig;
}

int MDFSTuple::get(int i) const {
    return v[i];
}

size_t MDFSTuple::getDim() const {
    return v.size();
}

void MDFSTuple::print() {
    printf("%d,%f,%d", i, ig, v[0]);
    for (auto u = v.begin() + 1; u != v.end(); u++) {
        printf(",%d", *u);
    }
    printf("\n");
}


MDFSOutput::MDFSOutput(MDFSOutputType type, int variable_count)
        : max_igs_tuples(nullptr), type(type) {
    switch(type) {
        case MDFSOutputType::MaxIGs:
            max_igs = new std::vector<float>(variable_count);
            break;
        case MDFSOutputType::MatchingTuples:
            tuples = new std::list<MDFSTuple>();
            break;
   }
}

MDFSOutput::~MDFSOutput() {
    switch(type) {
        case MDFSOutputType::MaxIGs:
            delete max_igs;
            break;
        case MDFSOutputType::MatchingTuples:
            delete tuples;
            break;
   }
}

void MDFSOutput::setTuples(int *tuples) {
    this->max_igs_tuples = tuples;
}

void MDFSOutput::print() {
    switch(type) {
        case MDFSOutputType::MaxIGs:
            printf("%f", (*max_igs)[0]);
            for (auto v = max_igs->begin() + 1; v != max_igs->end(); v++) {
                printf("\t%f", *v);
            }
            printf("\n");
            break;
        case MDFSOutputType::MatchingTuples:
            for (auto v = tuples->begin(); v != tuples->end(); v++) {
                v->print();
            }
            break;
   }
}

void MDFSOutput::updateMaxIG(const Tuple& tuple, float *digs) {
    if (max_igs_tuples == nullptr) {
        for (int i = 0; i < tuple.dimensions(); ++i) {
            int v = tuple.get(i);
            if (digs[i] > (*max_igs)[v]) {
                (*max_igs)[v] = digs[i];
            }
        }
    } else {
        for (int i = 0; i < tuple.dimensions(); ++i) {
            int v = tuple.get(i);
            if (digs[i] > (*max_igs)[v]) {
                (*max_igs)[v] = digs[i];
                std::copy(tuple.begin(), tuple.end(), max_igs_tuples + tuple.dimensions() * v);
            }
        }
    }
}

void MDFSOutput::copyMaxIGsAsDouble(double *copy) const {
    std::copy(max_igs->begin(), max_igs->end(), copy);
}

void MDFSOutput::addTuple(int i, float ig, const Tuple& vt) {
    tuples->emplace_back(i, ig, std::vector<int>(vt.begin(), vt.end()));
}

size_t MDFSOutput::getMatchingTuplesCount() const {
    return this->tuples->size();
}

void MDFSOutput::copyMatchingTuples(int* matching_tuples_vars, double* IGs, int* matching_tuples) const {
    size_t tuples_count = this->getMatchingTuplesCount();

    size_t i = 0; // moved up since multiple different type initializers are not allowed
    for (auto v = this->tuples->begin(); i < tuples_count; ++v, ++i) {
        matching_tuples_vars[i] = (*v).getVar();
        IGs[i] = (*v).getIG();
        for (size_t j = 0; j < (*v).getDim(); ++j) {
            matching_tuples[j * tuples_count + i] = (*v).get(j); // column-first
        }
    }
}
