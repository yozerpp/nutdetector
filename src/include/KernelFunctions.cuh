//
// Created by jonossar on 3/14/24.
//

#ifndef IMG1_KERNELFUNCTIONS_CUH
#define IMG1_KERNELFUNCTIONS_CUH
#ifndef IMG1_COMMON_H
#include "common.h"
#endif

#ifndef IMG1_IMAGE_H
#include "Image.h"
#endif


namespace KernelFunctions {
    __device__ void grayScale(unsigned char * in, unsigned char * out, unsigned int x,unsigned int y);
     __device__ void distribution(unsigned char* in, unsigned int* out, unsigned int x, unsigned int y);
     __device__ void dilate(const unsigned int kernelDim,unsigned char* in, unsigned char* out, unsigned int x, unsigned int y);
     __device__ void erode(const unsigned int kernelDim,unsigned char* in, unsigned char* out, unsigned int x, unsigned int y);
     __device__ void gaussian(const unsigned int kernelDim,const unsigned int sigma, unsigned char *in, unsigned char* out, unsigned int x, unsigned int y);
     __device__ void cluster(float * means, const unsigned int len,unsigned char* in, unsigned char* out, unsigned int x, unsigned int y);
        template<typename I,typename O>
    __host__
    O * kernelFunctionsWrapper(I *in,void(*func)(I*,O*, unsigned int, unsigned int),  const unsigned int x, const unsigned int y, const unsigned int z,
                               const unsigned int outSize,const unsigned int sharedMem=0);
}
#endif //IMG1_KERNELFUNCTIONS_CUH
