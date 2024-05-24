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

    /** @paragraph These declarations of algorithms/functors used in kernel, structure of these functions are they perform their operation on data through @code run() @code method. This file contains summaries about these methods in documentation together with summaries of other members of the functor/algorithm, you can find their definitions of these methods in @p Kernel.cu . **/

    /** calculates histogram of the input
     * this function calculates the weighted sum of every pixel @code i*data[i] @code where @p i is index and @p data[i] is strength of the pixel/element.
     * @tparam T type of data
     * @param origin minimum value in histogram
     * @param bound max value in histogram
     */
    template <typename T>
    struct distribution {
    private:
        const T origin;
        const T bound;
        double *sum= nullptr;
    public:

        __host__ distribution(T origin=(T)0, T bound=(T)255);
        inline __host__ void prepare(unsigned int s);
        inline __host__ void destroy();
        inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                            unsigned int *out, const double *kernel);
         inline __device__ void post(const unsigned int i,const T *in,unsigned int *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);

    };
    /** performs erosion
     * @param data in
     * this algorithm sets output pixel to @p 255 if any of the surrounding pixels in the input image is @p 255
     */
    struct erode {
    private:
        bool *flag = nullptr;
    public:
        inline  __host__ void prepare(unsigned int s);
        inline __host__ void destroy();
        inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned char *out, const double *kernel);

        inline __device__ void post(const unsigned int i, const unsigned char *in,unsigned char *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
    };
    /** performs dilation
     * @param data in
     * this algorithm sets output pixel to 0 if any of the surrounding pixels in the input image is 0
     */
    struct dilate {
    private:
        bool *flag= nullptr;
    public:
        inline __host__ void prepare(unsigned int s);
        inline __host__ void destroy();
        inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned char *out, const double *kernel);

        inline  __device__ void post(const unsigned int i, const unsigned char *in,unsigned char *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
    };
    /** calculates grayscale value of the input image
     *
 */
    struct grayScale {
    public:
        inline __host__ void prepare(unsigned int s);
        inline __host__ void destroy();
        inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                            unsigned char *out, const double *kernel);

        inline __device__ void post(const unsigned int i, const unsigned char *in,unsigned char *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
    };
    /** performs a gaussian filter on the image
     *
     * @tparam T
     */
    template <typename T>
    struct gaussian{
        float* sum;
        inline __host__ void destroy();
        inline __host__ void prepare(unsigned int s);
        inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                                   T *out, const float *kernel);
        inline __device__ void post(const unsigned int i, const T *in,T *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
    };
    /** this function assigns
     *
     * @tparam T filter type
     * @tparam O output type
     * @param max value to assign if the input element of type @p T @p isAboveThreshold
     * @param min if above condition is false @p min is assigned.
     * @param means means of the input data
     */
    template <typename T, typename O>
    struct cluster {
    private:
        const unsigned int threshold;
        O min;
        O max;
        Common::Mean* means= nullptr;
        unsigned int numMeans;
        inline __device__ Common::Mean *closestMean(float value);

        inline __device__ bool isAboveThreshold(Common::Mean* mean);

    public:
        /***
         *
         * @param means
         * @param size
         * @param min
         * @param max
         * @param onlyHighest supresses every value that is closest to a nonMax mean
         */
        __host__ explicit cluster(Common::Mean *means, unsigned int size=2, O min=(O)0,O max=(O)255, unsigned int threshold=1);
       inline __host__ void destroy();
       inline __host__ void prepare(unsigned int s);
       inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                            O *out, const double *kernel);
       inline __device__ void post(const unsigned int i, const T *in,O *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
    };
    template <typename T>
    struct multiply{
        T * val;
        multiply(T * val): val(val){}
        inline __host__ void destroy();
        inline __host__ void prepare(unsigned int s);
        inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                                   T*out, const double *kernel);
        inline __device__ void post(const unsigned int i, const T *in,T*out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
    };
    namespace edge{
        /** graident strcut holding value and direction of the gradient.
         *
         */
        struct gradient{
        public:
            float value;
            unsigned int axis;
            __host__ __device__ gradient(const float value,const unsigned int axis):value(value),axis(axis){}
            __host__ __device__ gradient(const gradient& other):value(other.value),axis(other.axis){}
            __host__ __device__ gradient(const gradient&& other):value(other.value),axis(other.axis){}
            __host__ __device__ gradient()=default;
            __host__ __device__ gradient(const double val):value(val),axis(0){}
            __host__ __device__ gradient(const float val):value(val),axis(0){}
            __host__ __device__ gradient(const int val):value(val),axis(0){}
            __device__ static inline void getNeighbours(const edge::gradient* all, edge::gradient* neighbours, bool orthogonal, unsigned int axis, const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int& y);
            __host__ __device__ explicit operator float()const{
                return value;
            }
            __host__ __device__ explicit operator double()const{
                return (double)value;
            }
            __host__ __device__ explicit operator unsigned int()const{
                return(unsigned int) value;
            }
            __host__ __device__ gradient operator =(const gradient& other){
                this->value=other.value;
                this->axis=other.axis;
                return *this;
            }
            __host__ __device__ gradient operator /(const gradient& other) const{
                return {this->value/other.value,axis};
            }
            __host__ __device__ gradient operator /=(const gradient& other){
                this->value/=other.value;
                return *this;
            }
            __host__ __device__ gradient operator *(const gradient& other)const{
                return {this->value*other.value, axis};
            }
            __host__ __device__ gradient operator +(const gradient& other)const{
                return {this->value+other.value, axis};
            }
            __host__ __device__ gradient operator *=(const gradient& other){
                this->value*=other.value;
                return (*this);
            }
            __host__ __device__ gradient operator -(const gradient& other)const{
                return {this->value-other.value, axis};
            }
            __host__ __device__ bool operator >(const gradient& other) const{
                return this->value>other.value;
            }
            __host__ __device__ bool operator <(const gradient& other)const{
                return this->value<other.value;
            }
            __host__ __device__ bool operator ==(const gradient& other)const{
                return this->value==other.value;
            }
            __host__ __device__ bool operator !=(const gradient& other)const{
                return this->value!=other.value;
            }
            __host__ __device__ bool operator >=(const gradient& other)const{
                return this->value >= other.value;
            }
            __host__ __device__ bool operator <=(const gradient& other)const {
                return this->value <= other.value;
            }
        };
        /** extract gradient of the image with sobel masks
         * @param masks pointer to sobel masks
         * @param sum temporary value to accumulate result of convolution
         */
        struct getGradient{

            float* sum;
            float* masks;
            __host__ getGradient(float **masks);
            inline __host__ void destroy();
            inline __host__ void prepare(unsigned int s);
            inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                                       gradient*out, const double *kernel);
            inline __device__ void post(const unsigned int i, const unsigned char *in,gradient *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
        };
        /** perform nonMaxSuppression on the input image.
         * this algorithm supresses (puts zero) to output if it's not the highest pixel with gradient in the corresponding pixel in the input image
         */
        struct nonMaxSuppress{
            const unsigned int x;
            const unsigned int y;
            __host__ nonMaxSuppress(unsigned int x, unsigned int y): x(x), y(y){}
            inline __host__ void destroy();
            inline __host__ void prepare(unsigned int s);
            inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const gradient *in,
                                       gradient*out, const double *kernel);
            inline __device__ void post(const unsigned int i,const gradient *in,gradient *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
        };
        /** perform hysteresis
         * this algorithm performs puts 1020 on output element if corresponding input element: either is part of the highest cluster, or is neighbour to a pixel that is part of the highest cluster.
         */
        struct hysteresis{
        private:
            unsigned int * count;
            Common::Mean* means;
            unsigned int len;
            inline __device__ unsigned int position(const gradient value);
        public:
            __host__ hysteresis(Common::Mean *means, unsigned int len, unsigned int* count);
            inline __host__ void destroy();
            inline __host__ void prepare(unsigned int s);
            inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const gradient *in,
                                       gradient *out, const double *kernel);
            inline __device__ void post(const unsigned int i, const gradient *in,gradient *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
        };
        /** this function accumulates the hough matrix from the input binary(edge detected) image.
         *
         */
        struct houghLine{
        public:
            inline __host__ void destroy();
            inline __host__ void prepare(unsigned int s);
            inline __device__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                                       unsigned int *out, const double *kernel);
            inline __device__ void post(const unsigned int i, const unsigned char *in,unsigned int *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y);
        };
    }
//    template<typename I, typename O, typename K,typename A>
//    __global__  void convolutionWrapper(I* in, O* out, K* kernel, A &alg, const unsigned int x, const unsigned int y, const unsigned int kernelDim);
/** interface to gpu kernel
 * this functions performs neccessary arrangements  for gpu execution (such as memory allocation, kernel configuration) and then calls @p convolutionWrapper .
 * @tparam I type of input
 * @tparam O type of output
 * @tparam K type of convolution kernel
 * @tparam A type of the algorithm/functor that will be executed inside kernel
 * @param in
 * @param out
 * @param kernel
 * @param alg functor instance to be executed
 * @param x
 * @param y
 * @param kernelDim dimension of kernel
 */
        template<typename I,typename O, typename K, typename A>
    __host__
    O * executor(I *in, K *kernel, A && algo, const unsigned int x, const unsigned int y, const unsigned int z,
                 const unsigned int outSize, const unsigned int kernelDim);
        template <typename T, typename O=T>
       __host__ O* normalize(T* in,unsigned int len,T * val=nullptr, T max=(T)255);
    template <typename T>
    __host__ T* sort(T* arr, unsigned int len);
}
//__device__ static inline Kernel::edge::gradient atomicExch(Kernel::edge::gradient* addr, Kernel::edge::gradient val){
//    atomicExch((float*)&(addr->value),(float) val.value);
//    atomicExch((unsigned int*)&(addr->ax), (unsigned int)val.ax);
//    return *addr;
//}
#endif //IMG1_KERNELFUNCTIONS_CUH
