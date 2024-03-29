//
// Created by jonossar on 3/9/24.
//
#include <cstring>
#include <map>
#include "../include/common.h"

#define IMG1_COMMON_IMPLEMENTATION
namespace Common {
    template double * initializeArray<double> (double val, int size);
    template float * initializeArray<float> (float val, int size);
    template long long int*initializeArray<long long int >(long long int val, int size);
    template unsigned int* initializeArray<unsigned int>(unsigned int val, int size);
    template unsigned char* initializeArray<unsigned char>(unsigned char val, int size);
}