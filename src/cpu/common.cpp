#include <algorithm>

#include "common.h"

/* MDFS Info */

MDFSInfo::MDFSInfo(
    size_t dimensions,
    size_t divisions,
    size_t discretizations,
    float pseudo,
    float ig_thr,
    int* interesting_vars,
    size_t interesting_vars_count,
    bool require_all_vars
) : dimensions(dimensions), divisions(divisions), discretizations(discretizations),
    pseudo(pseudo), ig_thr(ig_thr), interesting_vars(interesting_vars),
    interesting_vars_count(interesting_vars_count), require_all_vars(require_all_vars) {}


/* Tuple Generator */

TupleGenerator::TupleGenerator(size_t dimensions, size_t variable_count)
    : nextTuple(new Tuple(dimensions)), variable_count(variable_count) {}

bool TupleGenerator::hasNext() const {
    return this->nextTuple->combination[0] == 0;
}

std::unique_ptr<Tuple> TupleGenerator::next() {
    std::unique_ptr<Tuple> thisTuple = std::move(this->nextTuple);
    this->nextTuple.reset(new Tuple(*thisTuple, this->variable_count));

    return thisTuple;
}


/* Tuple */

Tuple::Tuple(size_t dimensions) : combination(dimensions + 1) {
    // Npte: this value becomes a "sentiel"
    this->combination[0] = 0;

    for (size_t d = 1; d <= dimensions; ++d) {
        this->combination[d] = d - 1;
    }
}

// Note: Would move it to the generator, so that it will initialize values and store state.
Tuple::Tuple(const Tuple& previous, size_t variable_count) : combination(previous.combination) {
    const int dimensions = this->dimensions();
    int d = dimensions;

    for (; d >= 0; --d) {
        ++(this->combination[d]);
        if (this->combination[d] < variable_count - (dimensions - d)) {
            break;
        }
    }

    for (++d; d <= dimensions; ++d) {
        this->combination[d] = this->combination[d-1]+1;
    }
}


/* MDFSTuple */

MDFSTuple::MDFSTuple(size_t i, float ig, std::vector<int>&& v)
    : i(i), ig(ig), v(v) {}

size_t MDFSTuple::getVariable() const {
    return this->i;
}

float MDFSTuple::getIG() const {
    return this->ig;
}

size_t MDFSTuple::get(size_t i) const {
    return this->v[i];
}

size_t MDFSTuple::getDim() const {
    return this->v.size();
}


/* MDFSOutput */

MDFSOutput::MDFSOutput(MDFSOutputType type, size_t variable_count)
    : max_igs_tuples(nullptr), type(type) {
    switch(type) {
        case MDFSOutputType::MaxIGs:
            this->max_igs = new std::vector<float>(variable_count);
            break;

        case MDFSOutputType::MatchingTuples:
            this->tuples = new std::list<MDFSTuple>();
            break;
   }
}

MDFSOutput::~MDFSOutput() {
    switch(type) {
        case MDFSOutputType::MaxIGs:
            delete this->max_igs;
            break;

        case MDFSOutputType::MatchingTuples:
            delete this->tuples;
            break;
   }
}

void MDFSOutput::setMaxIGsTuples(int *tuples, int *dids) {
    this->max_igs_tuples = tuples;
    this->dids = dids;
}

void MDFSOutput::updateMaxIG(const Tuple& tuple, float *digs, int *dids) {
    if (this->max_igs_tuples == nullptr) {
        for (size_t i = 0; i < tuple.dimensions(); ++i) {
            size_t v = tuple.get(i);

            if (digs[i] > (*(this->max_igs))[v]) {
                (*(this->max_igs))[v] = digs[i];
            }
        }
    } else {
        for (size_t i = 0; i < tuple.dimensions(); ++i) {
            size_t v = tuple.get(i);

            if (digs[i] > (*(this->max_igs))[v]) {
                (*(this->max_igs))[v] = digs[i];
                std::copy(tuple.begin(), tuple.end(), this->max_igs_tuples + tuple.dimensions() * v);
                this->dids[v] = dids[i];
            }
        }
    }
}

void MDFSOutput::copyMaxIGsAsDouble(double *copy) const {
    std::copy(this->max_igs->begin(), this->max_igs->end(), copy);
}

void MDFSOutput::addTuple(size_t i, float ig, const Tuple& vt) {
    this->tuples->emplace_back(i, ig, std::vector<int>(vt.begin(), vt.end()));
}

size_t MDFSOutput::getMatchingTuplesCount() const {
    return this->tuples->size();
}

void MDFSOutput::copyMatchingTuples(int* matching_tuples_vars, double* IGs, int* matching_tuples) const {
    size_t tuples_count = this->getMatchingTuplesCount();
    size_t i = 0;

    for (auto v = this->tuples->begin(); i < tuples_count; ++v, ++i) {
        matching_tuples_vars[i] = (*v).getVariable();
        IGs[i] = (*v).getIG();

        for (size_t j = 0; j < (*v).getDim(); ++j) {
            matching_tuples[j * tuples_count + i] = (*v).get(j); // column-first
        }
    }
}
