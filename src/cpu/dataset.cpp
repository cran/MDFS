#include "dataset.h"
#include "discretize.h"

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

RawDataInfo::RawDataInfo(size_t object_count, size_t variable_count)
	: object_count(object_count), variable_count(variable_count) {}

RawData::RawData(RawDataInfo data_file_info, const double* data, const int* decision)
	: info(data_file_info), data(data), decision(decision) {}

const double* RawData::getVariable(size_t var_index) const {
    return this->data + var_index * this->info.object_count;
}

DiscretizationInfo::DiscretizationInfo(uint32_t seed, size_t disc, size_t div, double range)
    : seed(seed), discretizations(disc), divisions(div), range(range) {}


DataSet::DataSet(void) {
}

DataSet::~DataSet() {
    // TODO: check each whether initialized or move to constructor or just redesign it...
    delete[] this->data;
    delete this->info;
}

void DataSet::loadData(RawData* rawdata, DiscretizationInfo discretization_info) {
    this->decision = rawdata->decision;

    size_t c0 = 0;

    for (size_t i = 0; i < rawdata->info.object_count; ++i) {
        if (this->decision[i] == 0) {
            ++c0;
        }
    }

    this->info = new DataSetInfo({
        discretization_info.discretizations,
        rawdata->info.object_count,
        rawdata->info.variable_count,
        {c0, rawdata->info.object_count-c0},
    });

    const auto allocSize = this->info->discretizations * this->info->object_count * this->info->variable_count;

    this->data = new uint8_t[allocSize];
    this->discretizeFile(rawdata, discretization_info);
}

void DataSet::discretizeVar(
    RawData* in,
    size_t var_index,
    DiscretizationInfo info
) {
    const double* in_data = in->getVariable(var_index);

    std::vector<double> sorted_in_data(in_data, in_data + in->info.object_count);
    std::sort(sorted_in_data.begin(), sorted_in_data.end());

    for (size_t d = 0; d < info.discretizations; ++d) {
        discretize(
            info.seed,
            d,
            var_index,
            info.divisions,
            in->info.object_count,
            in_data,
            sorted_in_data,
            this->getDiscretizationData(var_index, d),
            info.range
        );
    }
}

void DataSet::discretizeFile(
    RawData* in,
    DiscretizationInfo info
) {
    for (size_t v = 0; v < in->info.variable_count; ++v) {
        this->discretizeVar(in, v, info);
    }
}
