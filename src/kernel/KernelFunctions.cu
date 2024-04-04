//
// Created by jonossar on 3/14/24.
//
#include <boost/thread.hpp>
#include "../include/KernelFunctions.cuh"
#include "../include/Preprocessor.h"

#define IMG1_KERNELFUNCTIONS_IMPLEMENTATION
//#define INLINE __forceinline__

namespace KernelFunctions {
    __device__ static INLINE char atomicAddChar(char *address, char val) {
        // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
        size_t long_address_modulo = (size_t) address & 3;
        // the 32-bit address that overlaps the same memory
        auto *base_address = (unsigned int *) ((char *) address - long_address_modulo);
        // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
        // The "4" signifies the position where the first byte of the second argument will end up in the output.
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
        unsigned int selector = selectors[long_address_modulo];
        unsigned int long_old, long_assumed, long_val, replacement;

        long_old = *base_address;

        do {
            long_assumed = long_old;
            // replace bits in long_old that pertain to the char address with those from val
            long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
            replacement = __byte_perm(long_old, long_val, selector);
            long_old = atomicCAS(base_address, long_assumed, replacement);
        } while (long_old != long_assumed);
        return __byte_perm(long_old, 0, long_address_modulo);
    }
    __device__
    static INLINE
    uint8_t
    atomicCASChar(uint8_t *const address,
                  uint8_t const compare,
                  uint8_t const value) {
        // Determine where in a byte-aligned 32-bit range our address of 8 bits occurs.
        uint8_t const longAddressModulo = reinterpret_cast< size_t >( address ) & 0x3;
        // Determine the base address of the byte-aligned 32-bit range that contains our address of 8 bits.
        uint32_t *const baseAddress = reinterpret_cast< uint32_t * >( address - longAddressModulo );
        uint32_t constexpr byteSelection[] = {0x3214, 0x3240, 0x3410,
                                              0x4210}; // The byte position we work on is '4'.
        uint32_t const byteSelector = byteSelection[longAddressModulo];
        uint32_t const longCompare = compare;
        uint32_t const longValue = value;
        uint32_t longOldValue = *baseAddress;
        uint32_t longAssumed;
        uint8_t oldValue;
        do {
            // Select bytes from the old value and new value to construct a 32-bit value to use.
            uint32_t const replacement = __byte_perm(longOldValue, longValue, byteSelector);
            uint32_t const comparison = __byte_perm(longOldValue, longCompare, byteSelector);

            longAssumed = longOldValue;
            // Use 32-bit atomicCAS() to try and set the 8-bits we care about.
            longOldValue = ::atomicCAS(baseAddress, comparison, replacement);
            // Grab the 8-bit portion we care about from the old value at address.
            oldValue = (longOldValue >> (8 * longAddressModulo)) & 0xFF;
        } while (compare == oldValue and
                 longAssumed != longOldValue); // Repeat until other three 8-bit values stabilize.

        return oldValue;
    }
    template<typename I, typename O>
    using kernelFunction =  void (*)(I*, O*, unsigned int, unsigned int, unsigned int idx_x, unsigned int idx_y);
    template<typename I,typename O>
    __global__ void wrapper(kernelFunction<I,O> f,I* inGpu, O* outGpu, unsigned int x, unsigned int y){
        f(inGpu, outGpu, x, y, threadIdx.x + blockIdx.x*blockDim.x,threadIdx.y + blockIdx.y*blockDim.y);
    }
    template<typename I, typename O>
    __host__ O *
    kernelFunctionsWrapper(I *in,void(* func)(I*,O*, unsigned int, unsigned int), const unsigned int x, const unsigned int y, const unsigned int z,
                           const unsigned int outSize, const unsigned int sharedMem) {
        dim3 threads(KERNEL_THREADS, KERNEL_THREADS, 1);
        double x2=Common::roundP2( x);
        double y2=Common::roundP2( y);
        x2=x2>y2?x2:y2;
        y2=y2>x2?y2:x2;
        unsigned int d1 = x2 / threads.x;
        unsigned int d2 = y2 / threads.y;
        dim3 blocks(d1, d2, 1);
        O *outGpu;
        I *inGpu;
        kernelFunction<I,O> f;
        cudaMalloc(&f, sizeof(func));
        cudaMemcpy(&f, &func, sizeof(kernelFunction<I,O>), cudaMemcpyHostToDevice);
        gc(cudaMalloc(&(outGpu), outSize * sizeof(O)));
        gc(cudaMemset(outGpu,  0, outSize * sizeof(O)));
        gc(cudaMalloc(&inGpu, x * y * z * sizeof(I)));
        gc(cudaMemcpy(inGpu, in, x * y * z * sizeof(I), cudaMemcpyHostToDevice));
        gc(cudaStreamSynchronize(thisStream));
        wrapper<<<blocks, threads,sharedMem,thisStream>>>(f,inGpu, outGpu, x,y);
        gc(cudaStreamSynchronize(thisStream));
        cudaError err = cudaGetLastError();
        O *out = new O[outSize];
        gc(cudaMemcpy(out, outGpu, outSize * sizeof(O), cudaMemcpyDeviceToHost));
        gc(cudaFree(outGpu));
        gc(cudaFree(inGpu));
        return out;
    }
    static INLINE __device__ float * closestMean(unsigned char in, float * means, const unsigned int len){
        float val=FLT_MAX;
        float * ret;
        for(unsigned int i=0; i<len; i++){
            float dist=abs(means[i] - in);
            val=val>dist?dist:val;
            ret=&means[i];
        }
        return ret;
    }
    static INLINE __device__ bool isLowerMean(float * mean, float * means, const unsigned int len){
        bool ret=true;
        for(unsigned int i=0; i<len; i++){
            if(*mean>means[i]) ret=false;
        }
        return ret;
    }
    __device__ void cluster(float *means, const unsigned int len,unsigned char* in, unsigned char* out, unsigned int x, unsigned int y){
        const unsigned int flatIdx=(threadIdx.y + blockIdx.y*blockDim.y)*x + threadIdx.x + blockIdx.x*blockDim.x;
        if(flatIdx>=x*y) return;
        if(isLowerMean(closestMean(in[flatIdx], means, len),means, len)) out[flatIdx]=0;
        else out[flatIdx]=255;
    }
    __device__ void grayScale(unsigned char * in, unsigned char * out,unsigned int x, unsigned int y){
        const unsigned int flatIdx=(threadIdx.y + blockIdx.y*blockDim.y)*x + threadIdx.x + blockIdx.x*blockDim.x;
        if(flatIdx>=x*y) return;
        atomicAddChar((char*)&out[flatIdx],(char) in[flatIdx*3] + in[flatIdx*3+1] + in[flatIdx*3+2]);
    }
    __device__ void distribution(unsigned char* in, unsigned int* out, unsigned int x, unsigned int y){
        const unsigned int flatIdx=(threadIdx.y + blockIdx.y*blockDim.y)*x + threadIdx.x + blockIdx.x*blockDim.x;
        if(flatIdx>=x*y) return;
        atomicAdd(&out[in[flatIdx]],1);
    }
    __device__ void dilate(const unsigned int kernelDim,unsigned char* in, unsigned char* out, unsigned int x, unsigned int y){
        const unsigned int idx_x=threadIdx.x + blockIdx.x*blockDim.x;
        const unsigned int idx_y=threadIdx.y + blockIdx.y*blockDim.y;
        if(idx_y>=y || idx_x >=x) return;
        for(int y=0; y<kernelDim; y++)
            for(int x=0; x<kernelDim; x++){
                if(in[(idx_y+y)*x + idx_x+x]==(unsigned char)0){
                    out[idx_y*x+ idx_x]=0;
                    return;
                }
            }
        out[idx_y*x + idx_x]=(unsigned char)255;
    }
     __device__ void erode(const unsigned int kernelDim,unsigned char* in, unsigned char* out, unsigned int x, unsigned int y){
        const unsigned int idx_x=threadIdx.x + blockIdx.x*blockDim.x;
        const unsigned int idx_y=threadIdx.y + blockIdx.y*blockDim.y;
        if(idx_y>=y || idx_x >=x) return;
        for(int y=0; y<kernelDim; y++)
            for(int x=0; x<kernelDim; x++){
                if(in[(idx_y+y)*x + idx_x+x]==(unsigned char)255){
                    out[idx_y*x+ idx_x]=255;
                    return;
                }
            }
        out[idx_y*x + idx_x]=(unsigned char)0;
    }
    static __global__ void _gaussian(const unsigned int i, const unsigned int dim,const unsigned int sigma,const bool vertical,const unsigned char* in, float *out, const unsigned int x, const unsigned int y){
        __shared__ float sum;
        const unsigned int idx=threadIdx.x + blockIdx.x*blockDim.x;
        if(idx>dim) return;
        if(idx==0) sum=0;
        __syncthreads();
        if(i + idx*vertical?x:1 > x*y) return;
        atomicAdd(&sum, in[i + idx*vertical?x:1]*(1/(sigma* sqrt(2*M_PI)))*(1/(exp(pow(idx,2)/(2*pow(sigma,2))))));
        __syncthreads();
        if(idx==0) {
            sum=sum>255?255:sum;
            atomicExch(&out[i + idx*vertical?x:1], sum);
        }
    }
    __device__ void gaussian(const unsigned int kernelDim,const unsigned int sigma, unsigned char *in, unsigned char* out, unsigned int x, unsigned int y ){
        const unsigned int flatIdx=(threadIdx.y + blockIdx.y*blockDim.y)*x + threadIdx.x + blockIdx.x*blockDim.x;
        if(flatIdx>=x*y) return;
        float * xMean;
        float * yMean;
        cudaMalloc(&xMean, sizeof(float));
        dim3 kernelThreads(kernelDim, 1,1);
        _gaussian<<<1,kernelThreads, sizeof(float)>>>(flatIdx, kernelDim,sigma, false, in, xMean, x,y);
        cudaMalloc(&yMean, sizeof(float ));
        _gaussian<<<1, kernelThreads, sizeof(float), cudaStreamTailLaunch>>>(flatIdx, kernelDim,sigma, true, in, yMean,x,y);
        atomicAddChar((char*)&out[flatIdx], (char)(*xMean)*(*yMean));
    }
    template unsigned int* KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned int>(unsigned char*, void (*)(unsigned char*, unsigned int*, unsigned int, unsigned int), unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    template unsigned char* KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char>(unsigned char*, void (*)(unsigned char*, unsigned char*, unsigned int, unsigned int), unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
}

