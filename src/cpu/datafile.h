#ifndef DATAFILE_H
#define DATAFILE_H

class DataFileInfo {
public:
    DataFileInfo(int o, int v);

    int object_count;
    int variable_count;
};

//Stored in VO way
class DataFile {
public:
    DataFile(DataFileInfo dfi);
    DataFile(DataFileInfo dfi, const double* data, const int* decision);

    DataFileInfo info;
    const double* data;
    const int* decision;

    /**
     * @param var requested variable number
     * @return pointer to array (info.object_count length) with requested variable
     */
    const double* getV(int var) const;
};

#endif
