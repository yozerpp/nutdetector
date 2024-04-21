//
// Created by jonossar on 3/14/24.
//
#include <boost/thread.hpp>
#include "../include/Kernel.cuh"
#include "../include/Preprocessor.h"

#define IMG1_KERNELFUNCTIONS_IMPLEMENTATION
//#define inline __forceinline__

namespace Kernel {
    __device__
    static inline
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
    __device__ static inline char atomicAddChar(char *address, char val) {
        // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
        size_t long_address_modulo = (size_t) address & 3;
        // the 32-bit address that overlaps the same memory
        auto *base_address = (ui *) ((char *) address - long_address_modulo);
        // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
        // The "4" signifies the position where the first byte of the second argument will end up in the output.
        ui selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
        ui selector = selectors[long_address_modulo];
        ui long_old, long_assumed, long_val, replacement;

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
    //Configuration
    struct KernelConf{
        dim3 blocks;
        dim3 threads;
        KernelConf(dim3 b, dim3 t): blocks(b), threads(t){}
    };
    static __host__ inline KernelConf calculateConf(const ui x, const ui y){
        dim3 threads(KERNEL_THREADS, KERNEL_THREADS, 1);
        double x2=Common::roundP2( x);
        double y2=Common::roundP2( y);
        x2=x2>y2?x2:y2;
        y2=y2>x2?y2:x2;
        ui d1 = x2 / threads.x;
        ui d2 = y2 / threads.y;
        dim3 blocks(d1, d2);
        return {blocks,threads};
    }
    //KernelFunctor
    template<typename I, typename O, typename C>
    static __global__ void run(KernelFunctor<I,O,C> *instance){ //transfer instance to gpu memory before calling
        const ui idxy=threadIdx.y + blockDim.y*blockIdx.y;
        const ui idxx=threadIdx.x + blockDim.x*blockIdx.x;
        instance->operator()(idxx, idxy);
    }
    template <typename I, typename O, typename C>
     __host__ KernelFunctor<I,O,C>* KernelFunctor<I,O,C>::toHost(C* dev){
        auto* host= VirtualTransferable<C>::toHost(dev);
        cudaFree(dev);
        auto * o=(O*) malloc(sizeof(O)*host->outSize);
        cudaMemcpy(o, host->out, sizeof(O)*host->outSize, cudaMemcpyDeviceToHost);
        cudaFree(host->out);host->out=o;
        return host;
    }
    template <typename I, typename O, typename C>
    __host__ C* KernelFunctor<I,O,C>::toGpu() {
        I* tmp;
        Tensor<I>* t;
        cudaMalloc(&tmp, data->x*data->y*data->z*sizeof(I));
        cudaMalloc(&t, sizeof(Tensor<I>));
        cudaMemcpy(tmp, data->data, data->x*data->y*data->z*sizeof(I), cudaMemcpyHostToDevice);
        cudaMemcpy(t, data, sizeof(Tensor<I>), cudaMemcpyHostToDevice);
        return VirtualTransferable<C>::toGpu();
    }
    template <typename I,typename O, typename C>
    __host__ Kernel::KernelFunctor<I,O,C>:: KernelFunctor(Tensor<I> & in, ui outSize): VirtualTransferable<C>(), data(&in),outSize(outSize) {
        this->outSize=(outSize==0?(in.x*in.y*in.z):outSize);
        cudaMalloc(&(this->out), this->outSize*sizeof(O));
        cudaMemset(this->out,(O) 0, this->outSize*sizeof(O));
    }
    template <typename I,typename O, typename C>
    __host__ void Kernel::KernelFunctor<I,O,C>::operator delete(void* p){
        auto * that=static_cast<C*>(p);
        auto* t=(Tensor<I>*)malloc(sizeof(Tensor<I>));
        cudaMemcpy(t, that->data,sizeof(Tensor<I>), cudaMemcpyDeviceToHost);
        cudaFree(t->data);
        cudaFree(that->data);
        free(t);
        free(that);
    }
    template<typename I, typename O, class C>
    O* Kernel::KernelFunctor<I, O, C>::operator()() {
        KernelConf conf= calculateConf(data->x, data->y);
        auto * func=((KernelFunctor<I,O,C>*)this)->toGpu();
        run<<<conf.blocks,conf.threads>>>(func);
        auto * host= KernelFunctor<I,O,C>::toHost(func);
        auto * ret= static_cast<O*>(host->out);
        cudaFree(func);
        C::operator delete(host);
        return ret;
    }
    template<typename I, typename O, typename C>
    __device__ KernelFunctor<I,O,C>::KernelFunctor(KernelFunctor<I,O,C> & other): VirtualTransferable<C>(other),data(other.data), outSize(other.outSize), out(other.out){}
    //Convolution Functor
    template<typename I, typename O, typename T, typename C>
    static __global__ void layer2(ConvolutionFunctor<I,O,T,C> *instance, const ui ix, const ui iy){
        __shared__ T sum;
        const ui kix= threadIdx.x + blockIdx.x * blockDim.x;
        const ui kiy=threadIdx.y + blockDim.y * blockIdx.y;
        const int rix= ix + (kix - instance->kernelDim / 2);
        const int riy=iy + (kiy - instance->kernelDim / 2);
        if(rix>=instance->data->x || rix<0 || riy>=instance->data->y || riy<0) return;
        if(kix == 0) sum=(T)0;
        __syncthreads();
        instance->operator()(ix, iy, rix, riy,kix,kiy, &sum);
        __syncthreads();
        if(kix == 0) {
            if(std::is_same<T, uc>::value)
                atomicAddChar((char*) &(instance->temp[iy * instance->data->x + ix]), sum);
            else atomicAdd(&instance->temp[iy * instance->data->x + ix], sum);
        }
    }
    template<typename I, typename O, typename T,typename C>
    static __global__ void run(ConvolutionFunctor<I,O,T,C> *instance){
        ui idxy=threadIdx.y + blockDim.y*blockIdx.y;
        ui idxx=threadIdx.x + blockDim.x*blockIdx.x;
        layer2<<<dim3(1,1,1), dim3(instance->kernelDim,instance->kernelDim,1), sizeof(T), cudaStreamTailLaunch>>>(instance, idxx, idxy);
        instance->operator()(idxx, idxy);
    }
    template<typename I, typename O, typename T,typename C>
    __host__ Kernel::ConvolutionFunctor<I,O,T,C>::ConvolutionFunctor(Tensor<I>& in, ui kernelDim, ui outSize): KernelFunctor<I, O,C>(in, outSize), kernelDim(kernelDim) {
        cudaMalloc(&temp, sizeof(T)*this->outSize);
    }
    template<typename I, typename O, typename T,typename C>
    __host__ void Kernel::ConvolutionFunctor<I,O,T,C>::operator delete (void * p){
        auto * host=static_cast<C*>(p);
        cudaFree(host->temp);
        KernelFunctor<I,O,C>::operator delete(p);
    }
    template<typename I, typename O,typename T, typename C>
    O* Kernel::ConvolutionFunctor<I,O,T,C>::operator()(){
        auto * func=((KernelFunctor<I,O,C>*)this)->toGpu();
        KernelConf conf= calculateConf(this->data->x, this->data->y);
        run<<<conf.blocks,conf.threads>>>(func);
        cudaStreamSynchronize(thisStream);
        gc(cudaGetLastError());
        auto* host=KernelFunctor<I,O,C>::toHost(func);
        cudaFree(func);
        auto* ret=host->out;
        C::operator delete(host);
        return ret;
    }
    template<typename I, typename O, typename T,typename C>
    __device__ Kernel::ConvolutionFunctor<I,O,T,C>::ConvolutionFunctor(ConvolutionFunctor<I,O,T,C> & other): KernelFunctor<I, O,C>(other), temp(other.temp),kernelDim(other.kernelDim){}
    //GrayScale
    __host__  Kernel::GrayScale::GrayScale(Tensor<uc>& in): KernelFunctor<uc,uc,GrayScale>(in, in.x*in.y){}
    __device__ Kernel::GrayScale::GrayScale(GrayScale & other): KernelFunctor<unsigned char, unsigned char, GrayScale>(other){}
    __host__  uc* GrayScale:: operator()(){
        return KernelFunctor<uc,uc,GrayScale>::operator()();
    }
    __device__ inline void GrayScale::operator()(const ui ix,const ui iy){
        const ui flatIdx=iy*data->x + ix;
        if(flatIdx>=data->x*data->y) return;
        atomicAddChar((char *) &out[flatIdx], (char)(data->data[flatIdx * 3] + data->data[flatIdx * 3 + 1] + data->data[flatIdx * 3 + 2]));
    }
    //Dilate_Erode
        __host__ Dilate_Erode::Dilate_Erode(Tensor<uc>& in, Mode mode,ui kernelDim): ConvolutionFunctor<uc, uc, ui, Dilate_Erode>(in, kernelDim, in.x*in.y*in.z), mode(mode){}
        __device__ Dilate_Erode::Dilate_Erode(Dilate_Erode& other): ConvolutionFunctor<uc,uc,ui,Dilate_Erode>(other), mode(other.mode){}
        __device__ inline void Dilate_Erode::operator()(const ui ix, const ui iy, const int rix, const int riy, const ui _,const ui _a, ui* acc){
            if(data->data[riy * data->x + rix] == (uc) (mode == dilate ? 0 : 255))
                out[iy*data->x+ ix]=(mode==dilate?0:255);
            else
                out[iy*data->x +ix]=(mode==dilate?255:0);
        }
    __host__  uc* Dilate_Erode::operator()(){
        return KernelFunctor<uc,uc,Dilate_Erode>::operator()();
    }
        __device__ inline void Dilate_Erode::operator()(const ui ix,const ui iy){
        }
    //Gaussian
    __host__ Gaussian::Gaussian(Tensor<uc>& in, ui sigma, ui kernelDim): ConvolutionFunctor<uc,uc, float, Gaussian>(in, kernelDim, in.x*in.y*in.z), sigma(sigma), kernel(nullptr){
        cudaMalloc(&kernel, sizeof(float )* kernelDim*kernelDim);
        auto * k= Common::gaussianKernel<float>(kernelDim, sigma);
        cudaMemcpy(kernel, k, sizeof(float)*kernelDim*kernelDim, cudaMemcpyHostToDevice);
    }
    __host__  uc* Gaussian::operator()() {
        return KernelFunctor<uc,uc,Gaussian>::operator()();
    }
    __device__ Gaussian::Gaussian(Gaussian& other): ConvolutionFunctor<uc,uc,float,Gaussian>(other), sigma(other.sigma), kernel(other.kernel){}
    __host__ void Gaussian::operator delete(void * p){
        auto * host=static_cast<Gaussian*>(p);
        cudaFree(host->kernel);
        ConvolutionFunctor::operator delete(host);
    }
    __device__ inline void Gaussian::operator()(const ui ix,const ui iy,const int rix,const int riy,const ui kix, const ui kiy, float * sum){
        atomicAdd(sum,data->data[riy * data->x + rix] * kernel[kiy*kernelDim + kix]);
    }
    __device__ inline void Gaussian::operator()(const ui ix,const ui iy) {
        out[iy*data->x + ix]=(uc )temp[iy*data->x + ix];
    }
    //Cluster
    static inline __device__ float * closestMean(uc in, const float * means, const ui len){
        float val=FLT_MAX;
        float * ret;
        for(ui i=0; i<len; i++){
            float dist=abs(means[i] - in);
            val=val>dist?dist:val;
            *ret=means[i];
        }
        return ret;
    }
    static inline __device__ bool isLowerMean(const float * mean,const float * means, const ui len){
        bool ret=true;
        for(ui i=0; i<len; i++){
            if(*mean>means[i]) ret=false;
        }
        return ret;
    }
     __host__ Cluster::Cluster(Tensor<uc>& in,float* means,ui len): KernelFunctor<uc,uc,Cluster>(in, in.x*in.y*in.z),len(len), means(nullptr){
         cudaMalloc(&means, sizeof(float)*len);
         cudaMemcpy(this->means,means,sizeof(float)*len, cudaMemcpyHostToDevice);
    }
    __host__  uc * Cluster:: operator()(){
        return KernelFunctor<uc,uc,Cluster>::operator()();
    }
    __device__ Cluster::Cluster(Cluster& other): KernelFunctor<unsigned char, unsigned char,Cluster>(other), len(other.len), means(other.means){}
    __device__ inline void Cluster::operator()(const ui ix, const ui iy) {
        if(isLowerMean(closestMean(data->data[iy*data->x + ix], means, len),means, len)) out[iy*data->x+ ix]=0;
        else out[iy*data->x+ix]=255;
    }
    //Distribution
    __host__ Distribution::Distribution(Tensor<unsigned char> &in) : KernelFunctor<unsigned char, unsigned int, Distribution>(in, 256){}
    __device__ Distribution::Distribution(Distribution&other) : KernelFunctor<unsigned char, unsigned int, Distribution>(other){}
    __device__ void Distribution::operator()(const ui ix, const ui iy){
        atomicAdd(&(this->out[data->data[iy*data->x+ix]]),1);
    }
    __host__  ui* Distribution::operator()() {
        return KernelFunctor<uc,ui,Distribution>::operator()();
    }
}

