#ifndef DISCRETIZEDFILE_H
#define DISCRETIZEDFILE_H

#include <cstdint>

class DiscretizedFileInfo {
public:
    DiscretizedFileInfo(int d, int o, int v);

    int discretizations;
    int object_count;
    int variable_count;
};

// Stored in VDO way
class DiscretizedFile {
public:
    DiscretizedFile(DiscretizedFileInfo dfi);
    ~DiscretizedFile();

    DiscretizedFileInfo info;
    /**
     * @brief this array can be perceived as matrix[variable][discretization][object]
     */
    uint8_t *data;
    const int *decision;

    /**
     * @param v variable number
     * @param d discretization number
     * @return pointer to array containing discretized v-variable with d-discretization
     */
    uint8_t *getVD(int v, int d);
    int c1();
    int c0();
};

#endif
