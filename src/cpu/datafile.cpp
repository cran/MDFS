#include "datafile.h"

DataFileInfo::DataFileInfo(int o, int v)
        : object_count(o), variable_count(v) {}

DataFile::DataFile(DataFileInfo dfi, const double* data, const int* decision)
        : info(dfi), data(data), decision(decision) { }

const double* DataFile::getV(int v) const {
    return this->data + v * this->info.object_count;
}
