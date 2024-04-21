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


namespace Kernel {
    struct distribution {
    private:

    public:
        bool isConvolution=false;
        INLINE __host__ void prepare(const unsigned int s);
        INLINE __host__ void destroy();
        INLINE __device__ static void run(const  unsigned int& ix, const unsigned int& iy, const unsigned int& kix, const unsigned int& kiy, const unsigned int& rix, const unsigned int& riy, const unsigned int& x, const unsigned int& y, const unsigned char *in, const unsigned int& kernelDim,
                                          void *out, void *kernel);
        static INLINE __device__ void post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const unsigned char *in, unsigned int *out);
    };

    struct erode {
    private:
        unsigned int *flag = nullptr;
    public:
        bool isConvolution= true;
        INLINE  __host__ void prepare(const unsigned int s);
        INLINE __host__ void destroy();
        static INLINE __device__ void run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y, unsigned char *in, const unsigned int& kernelDim,
                            unsigned int *out, void *kernel);

        static INLINE __device__ void post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const unsigned int * in, unsigned char *out);
    };

    struct dilate {
    private:
        unsigned int *flag= nullptr;
    public:
        bool isConvolution= true;
        INLINE __host__ void prepare(const unsigned int s);
        INLINE __host__ void destroy();
        static INLINE __device__ void run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                            unsigned int *out, void *kernel);

        static INLINE  __device__ void post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y, const unsigned int * in, unsigned char *out);
    };

    struct grayScale {
    public:
        bool isConvolution=false;
        INLINE __host__ void prepare(const unsigned int s);
        INLINE __host__ void destroy();
        INLINE __device__ void run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                            void *out,  void *kernel);

        static INLINE __device__ void post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const unsigned char * in,unsigned char *out);
    };
    struct gaussian{
    public:
        bool isConvolution= true;
        INLINE __host__ void prepare(const unsigned int s);
        INLINE __host__ void destroy();
        static INLINE __device__ void run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                                   float *out,  const float *kernel);

        static INLINE __device__ void post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const float * in,unsigned char *out);

    };
    struct cluster{
    public:
        bool isConvolution=false;
        __host__ cluster(Common::Mean* means, unsigned int len);
        INLINE __host__ void prepare(const unsigned int s);
        INLINE __host__ void destroy();
        static INLINE __device__ void run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                                   void *out,  void *kernel);

        INLINE __device__ void post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const unsigned char * in,unsigned char *out);
    private:
        Common::Mean* means;
        const unsigned int len;
        INLINE __device__ Common::Mean *closestMean(float value);

        INLINE __device__ bool isHigherMean(Common::Mean* mean);
    };
//    struct gaussianMean {
//    private:
//        double *sum= nullptr;
//        Common::Mean* means= nullptr;
//        const unsigned int len;
//
//        INLINE  __device__ unsigned int isScarceMean(Common::Mean *mean);
//
//        INLINE __device__ Common::Mean *closestMean(double value);
//
//        INLINE __device__ unsigned int isHigherMean(Common::Mean* mean);
//
//    public:
//        __host__ gaussianMean(Common::Mean *means, const unsigned int size);
//       INLINE __host__ void destroy();
//       INLINE __host__ void prepare(const unsigned int s);
//
//       INLINE __device__ void run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
//                            float *out,  float *kernel);
//
//       INLINE __device__ void post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y, const float * in, unsigned char *out);
//    };
//    template<typename I, typename O, typename K,typename A>
//    __global__  void LinearWrapper(I* in, O* out, K* kernel, A &alg,  const unsigned int x,  const unsigned int y,  const unsigned int kernelDim);
    template<typename I, typename O, typename A,typename K=void,typename T=I>
    __host__ O *
    Executor(I *in, K *kernel, A && algo, unsigned int x, unsigned int y, unsigned int z,
             unsigned int outSize, unsigned int kernelDim);
}
#endif //IMG1_KERNELFUNCTIONS_CUH
