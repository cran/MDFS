#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <cstdint>

class RawDataInfo {
public:
    RawDataInfo(int object_count, int variable_count);

    int object_count;
    int variable_count;
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
    const double* getVariable(int var_index) const;
};

class DiscretizationInfo {
public:
    DiscretizationInfo(uint32_t seed, int disc, int div, double range);

    uint32_t seed;
    int discretizations;
    int divisions;
    double range;
};

class DataSetInfo {
public:
    DataSetInfo(int discretizations, int object_count, int variable_count);

    size_t getAllocSize();

    int discretizations;
    int object_count;
    int variable_count;
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
        int var_index,
        DiscretizationInfo info
    );

    // This array can be perceived as matrix[variable][discretization][object]
    uint8_t *data;
    const int *decision;

    // returns pointer to array containing discretized v-variable with d-discretization
    uint8_t *getDiscretizationData(int var_index, int dis_index);

    int get_dec_count(int value);
    int c0();
    int c1();
};

#endif
