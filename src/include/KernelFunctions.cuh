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
    struct distribution {
    private:
        double *sum= nullptr;
    public:
        INLINE __host__ void prepare(unsigned int s);
        INLINE __host__ void destroy();
        INLINE __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned int *out, const double *kernel);
        INLINE __device__ void post(const unsigned int i, unsigned int *out);

    };

    struct erode {
    private:
        bool *flag = nullptr;
    public:
        INLINE  __host__ void prepare(unsigned int s);
        INLINE __host__ void destroy();
        INLINE __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned char *out, const double *kernel);

        INLINE __device__ void post(const unsigned int i, unsigned char *out);
    };

    struct dilate {
    private:
        bool *flag= nullptr;
    public:
        INLINE __host__ void prepare(unsigned int s);
        INLINE __host__ void destroy();
        INLINE __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned char *out, const double *kernel);

        INLINE  __device__ void post(const unsigned int i, unsigned char *out);
    };

    struct grayScale {
    public:
        INLINE __host__ void prepare(unsigned int s);
        INLINE __host__ void destroy();
        INLINE __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned char *out, const double *kernel);

        INLINE __device__ void post(const unsigned int i, unsigned char *out);
    };

    struct gaussianMean {
    private:
        double *sum= nullptr;
        Common::Mean* means= nullptr;
        unsigned int len;

        INLINE  __device__ bool isScarceMean(Common::Mean *mean);

        INLINE __device__ Common::Mean *closestMean(double value);

        INLINE __device__ bool isHigherMean(Common::Mean* mean);

    public:
        __host__ gaussianMean(Common::Mean *means, unsigned int size);
       INLINE __host__ void destroy();
       INLINE __host__ void prepare(unsigned int s);

       INLINE __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned char *out, const double *kernel);

       INLINE __device__ void post(const unsigned int i, unsigned char *out);
    };
    template<typename I, typename O, typename K,typename A>
    __global__  void kernelLoopWrapper(I* in, O* out, K* kernel, A &alg, const unsigned int x, const unsigned int y, const unsigned int kernelDim);
        template<typename I,typename O, typename K, typename A>
    __host__
    O * kernelFunctionsWrapper(I *in, K *kernel, A algo, const unsigned int x, const unsigned int y, const unsigned int z,
                               const unsigned int outSize, const unsigned int kernelDim);
}
#endif //IMG1_KERNELFUNCTIONS_CUH
