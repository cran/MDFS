#include "dataset.h"
#include "discretize.h"

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

RawDataInfo::RawDataInfo(int object_count, int variable_count)
	: object_count(object_count), variable_count(variable_count) {}

RawData::RawData(RawDataInfo data_file_info, const double* data, const int* decision)
	: info(data_file_info), data(data), decision(decision) {}

const double* RawData::getVariable(int var_index) const {
    return this->data + var_index * this->info.object_count;
}

DiscretizationInfo::DiscretizationInfo(uint32_t seed, int disc, int div, double range)
    : seed(seed), discretizations(disc), divisions(div), range(range) {}

DataSetInfo::DataSetInfo(int discretizations, int object_count, int variable_count)
    : discretizations(discretizations), object_count(object_count), variable_count(variable_count) {}

size_t DataSetInfo::getAllocSize() {
    // array size cast to size_t before multiplication due to possible int overflow
    return static_cast<std::size_t>(this->discretizations) * this->object_count * this->variable_count;
}

DataSet::DataSet(void) {
}

DataSet::~DataSet() {
    delete[] this->data;
    delete[] this->info;  // Should check whether is initialized?
}

void DataSet::loadData(RawData* rawdata, DiscretizationInfo discretization_info) {
    this->info = new DataSetInfo(discretization_info.discretizations, rawdata->info.object_count, rawdata->info.variable_count);
    this->data = new uint8_t[this->info->getAllocSize()];
    this->discretizeFile(rawdata, discretization_info);
}

void DataSet::discretizeVar(
    RawData* in,
    int var_index,
    DiscretizationInfo info
) {
    const double* in_data = in->getVariable(var_index);

    std::vector<double> sorted_in_data(in_data, in_data + in->info.object_count);
    std::sort(sorted_in_data.begin(), sorted_in_data.end());

    for (int d = 0; d < info.discretizations; ++d) {
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
    this->decision = in->decision;

    for (int v = 0; v < in->info.variable_count; ++v) {
        this->discretizeVar(in, v, info);
    }
}

uint8_t* DataSet::getDiscretizationData(int var_index, int dis_index) {
    std::size_t offset = this->info->object_count;
    offset *= (var_index * this->info->discretizations + dis_index);
    return this->data + offset;
}

int DataSet::get_dec_count(int value) {
    int counter = 0;

    for (int i = 0; i < this->info->object_count; ++i) {
        if (this->decision[i] == value) {
            counter++;
        }
    }

    return counter;
}

int DataSet::c0() {
    return this->get_dec_count(0);
}

int DataSet::c1() {
    return this->get_dec_count(1);
}
