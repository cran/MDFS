#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <cstdint>

class RawDataInfo {
public:
    RawDataInfo(size_t object_count, size_t variable_count);

    size_t object_count;
    size_t variable_count;
};

//Stored in VO way
class RawData {
public:
    RawData(RawDataInfo data_file_info);
    RawData(RawDataInfo data_file_info, const double* data, const int* decision);

    RawDataInfo info;
    const double* data;
    const int* decision;

    // returns pointer to array (info.object_count length) with requested variable
    const double* getVariable(size_t var_index) const;
};

class DiscretizationInfo {
public:
    DiscretizationInfo(uint32_t seed, size_t disc, size_t div, double range);

    uint32_t seed;
    size_t discretizations;
    size_t divisions;
    double range;
};

struct DataSetInfo {
    size_t discretizations;
    size_t object_count;
    size_t variable_count;
    size_t object_count_per_class[2];
};

// Stored in VDO way
class DataSet {
public:
    DataSet();
    ~DataSet();

    DataSetInfo* info;

    void loadData(
        RawData* rawdata,
        DiscretizationInfo info
    );

    void discretizeFile(
        RawData* in,
        DiscretizationInfo info
    );

    void discretizeVar(
        RawData* in,
        size_t var_index,
        DiscretizationInfo info
    );

    // This array can be perceived as matrix[variable][discretization][object]
    uint8_t *data;
    const int *decision;

    // returns pointer to array containing discretized v-variable with d-discretization
    inline uint8_t* getDiscretizationData(size_t var_index, size_t dis_index) const {
        return this->data + this->info->object_count * (var_index * this->info->discretizations + dis_index);
    }
};

#endif
