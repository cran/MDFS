#include "discretizedfile.h"

#include <cstddef>

DiscretizedFileInfo::DiscretizedFileInfo(int d, int o, int v)
        : discretizations(d), object_count(o), variable_count(v) {}

DiscretizedFile::DiscretizedFile(DiscretizedFileInfo dfi)
        : info(dfi) {
    // array size cast to size_t before multiplication due to possible int overflow
    this->data = new uint8_t[static_cast<std::size_t>(this->info.discretizations) * this->info.object_count * this->info.variable_count];
}

DiscretizedFile::~DiscretizedFile() {
    delete[] this->data;
}

uint8_t* DiscretizedFile::getVD(int v, int d) {
    std::size_t offset = this->info.object_count;
    offset *= (v * this->info.discretizations + d);
    return this->data + offset;
}

int DiscretizedFile::c1() {
    int c1 = 0;
    for (int i = 0; i < this->info.object_count; ++i)
        if (this->decision[i] == 1)
            c1++;
    return c1;
}

int DiscretizedFile::c0() {
    int c0 = 0;
    for (int i = 0; i < this->info.object_count; ++i)
        if (this->decision[i] == 0)
            c0++;
    return c0;
}
