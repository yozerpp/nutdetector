//
// Created by jonossar on 3/14/24.
//
#include "../include/Kernel.cuh"
#include "device_atomic_functions.h"
#define IMG1_KERNELFUNCTIONS_IMPLEMENTATION

__device__ unsigned char atomicAdd(unsigned char*address,unsigned char val) {
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
namespace Kernel {
//    template<typename I, typename O,  typename A,typename K>
//    __global__  void
//    ConvolutionWrapper(I *in,O *out,K *kernel,A * alg, const unsigned int& ix, const unsigned int& iy,
//                       const unsigned int& x, const unsigned int& y,const unsigned int& kernelDim) {
//        const unsigned int kix=threadIdx.x + blockDim.x*blockIdx.x;
//        const unsigned int kiy=threadIdx.y + blockIdx.y*blockDim.y;
//        const int rix=(int)ix-kix;
//        const int riy=(int)iy-kiy;
//        if(rix>=x||riy>=y || rix<0 || riy<0) return;
//        __shared__ O temp;
//        if(rix==0 && riy==0) temp=(O)0;
//        __syncthreads();
//        alg->run(ix,iy,kix,kiy,rix,riy,x,y,in,kernelDim,&temp,kernel);
//        __syncthreads();
//        if(kix==0 && kiy==0) {
//            atomicAdd(&out[riy*x+ rix],temp);
//        }
//    }
#define KERNEL_LAUNCH_LIMIT 590000
    static __device__ unsigned int count;
    template<typename I, typename O,  typename A, typename K,typename T>
    __global__  void
    LinearWrapper(I *in,O *out,K *kernel,A * alg, T* temp,const unsigned int x, const unsigned int y,
                  const unsigned int kernelDim) {
        const unsigned int idx_y=threadIdx.y + blockIdx.y*blockDim.y;
        const unsigned int idx_x=threadIdx.x + blockIdx.x*blockDim.x;
        if (idx_y >= y || idx_x>=x) return;
        if(alg->isConvolution) {
            for (unsigned int i = 0; i < kernelDim; i++) {
                for (unsigned int j = 0; j < kernelDim; j++) {
                    int y_pos = (int) idx_y + i - kernelDim / 2;
                    int x_pos = (int) idx_x + j - kernelDim / 2;
                    x_pos = x_pos >= 0 ? (x_pos < x ? x_pos : (x - 1)) : 0;
                    y_pos = y_pos >= 0 ? (y_pos < y ? y_pos : (y - 1)) : 0;
                    alg->run(idx_x, idx_y, j, i, x_pos, y_pos, x, y, in, kernelDim, temp, kernel);
                }
            }
        }
        alg->post(idx_x, idx_y, x,y,temp,out);
//        const unsigned int idx_y=threadIdx.y + blockIdx.y*blockDim.x;
//        const unsigned int idx_x=threadIdx.x + blockIdx.x*blockDim.x;
//        if (idx_y >= y || idx_x>=x) return;
//        if(alg->isConvolution) {
//            if(idx_x==0&&idx_y==0) count=0;
//            __syncthreads();
//            while(count + kernelDim*kernelDim > KERNEL_LAUNCH_LIMIT/2);
//            atomicAdd_system(&count, kernelDim*kernelDim);
//            ConvolutionWrapper<I, T, A, K><<<1, dim3(kernelDim, kernelDim, 1), sizeof(T),cudaStreamFireAndForget>>>(in, temp,
//                                                                                                          kernel,
//                                                                                                          alg, idx_x,
//                                                                                                          idx_y,
//                                                                                                          x, y,
//                                                                                                          kernelDim);
//            atomicSub_system(&count, kernelDim*kernelDim);
//        }
//        gc(cudaGetLastError());
//        alg->post(idx_x, idx_y,x,y,temp,out);
    }
    template<typename I, typename O, typename A,typename K,typename T>
    __host__ O *
    Executor(I *in, K *kernel, A && algo, unsigned int x, unsigned int y, unsigned int z,
             unsigned int outSize, unsigned int kernelDim) {
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
        K *kernelGpu = nullptr;
        A *algoGpu;
        T* temp= nullptr;
        algo.prepare(x*y);
        gc(cudaGetLastError());
        gc(cudaMalloc(&algoGpu, sizeof(A)));
        gc(cudaMemcpy(algoGpu, &algo, sizeof(A), cudaMemcpyHostToDevice));
        int kernelSize = (int) round(pow(kernelDim, 2));
        gc(cudaMalloc(&(outGpu), outSize * sizeof(O)));
        gc(cudaMemset(outGpu,  (O)0, outSize * sizeof(O)));
        if (kernel != nullptr) {
            gc(cudaMalloc(&kernelGpu, (kernelSize) * sizeof(double)));
            gc(cudaMemcpy(kernelGpu, kernel, kernelSize * sizeof(double), cudaMemcpyHostToDevice));
        }
        gc(cudaMalloc(&inGpu, x * y * z * sizeof(I)));
        gc(cudaMemcpy(inGpu, in, x * y * z * sizeof(I), cudaMemcpyHostToDevice));
        if(typeid(T).operator!=(typeid(I))){
            cudaMalloc(&temp, sizeof(T)*x*y*z);
            cudaMemset(temp, (T)0, sizeof(T)*outSize);
        } else temp=reinterpret_cast<T*>(inGpu);
        LinearWrapper<I, O, A,K,T> <<<blocks, threads, 0, thisStream>>>(inGpu, outGpu, kernelGpu, algoGpu,temp, x, y,
                                                        kernelDim > 0 ? kernelDim : 1);
        cudaStreamSynchronize(thisStream);
        gc(cudaGetLastError());
        O *out = new O[outSize];
        gc(cudaMemcpy(out, outGpu, outSize * sizeof(O), cudaMemcpyDeviceToHost));
        gc(cudaFree(outGpu));
        gc(cudaFree(inGpu));
        if (kernel != nullptr)
            gc(cudaFree(kernelGpu));
        algo.destroy();
        gc(cudaFree(algoGpu));
        return out;
    }
     INLINE __host__ void distribution::prepare(const unsigned int s) {}
    INLINE __device__ void distribution:: run(const  unsigned int& ix,const unsigned int& iy, const unsigned int& kix, const unsigned int& kiy, const unsigned int& rix, const unsigned int& riy,const unsigned int& x, const unsigned int&y,  const unsigned char *in, const unsigned int& kernelDim,
                               void *out,  void *kernel) {}
                               
    INLINE __device__ void distribution::post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const unsigned char *in, unsigned int *out){
        atomicAdd_block(&out[(int) in[iy*x+ix]], 1);
    }
    INLINE __host__ void distribution:: destroy(){}
    __host__ void erode::prepare(unsigned int s) {}
    INLINE __host__ void erode::destroy(){}
    
    INLINE __device__ void erode::run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y, unsigned char *in,const unsigned int& kernelDim,
                               unsigned int *out, void *kernel){
        if (in[riy*x+rix] == 255)
            atomicExch_block(out, 1);
    }
    
    INLINE __device__ void erode::post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const unsigned int * in, unsigned char *out) {
        if (*in==1) atomicAdd((unsigned  char*)&out[iy*x+ix], (unsigned char) 255);
    }
    __host__ void dilate::prepare(unsigned int s) {}
   
    INLINE __device__ void dilate::run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                               unsigned int *out, void *kernel) {
        if(in[riy*x+rix]==0)
            atomicExch_block(out, 1);
    }

    INLINE  __device__ void dilate::post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y, const unsigned int * in, unsigned char *out){
        if (*in==0) atomicAdd((unsigned char*)&out[iy*x+ix],(unsigned char)255);
    }
    __host__ void dilate::destroy() {}
    __host__ void grayScale::prepare(unsigned int s) {}
    INLINE __host__ void grayScale::destroy(){}
    INLINE __device__ void grayScale::run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                               void *out,  void *kernel) {}
                               
    INLINE __device__ void grayScale::post(const unsigned int &ix, const unsigned int &iy, const unsigned int &x,
                                           const unsigned int &y, const unsigned char *in, unsigned char *out) {
        const unsigned int flatIdx=iy*x+ix;
        char val= (char)(in[flatIdx * 3] * 0.21 + in[flatIdx * 3 + 1]* 0.72 + in[flatIdx * 3 + 2]* 0.07);
        atomicAdd((unsigned char *) &out[flatIdx],(unsigned char) val);
    }
    INLINE __host__ void gaussian::prepare(const unsigned int s) {}
    INLINE __host__ void gaussian::destroy() {}
    
    INLINE __device__ void gaussian:: run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                               float *out,  const float *kernel){
        atomicAdd_block(out, kernel[kiy*kernelDim + kix] * (float)in[riy*x + rix]);
    }
    INLINE __device__ void gaussian::post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const float * in,unsigned char *out){
        atomicAdd((unsigned char*)&out[iy*x+ix], (unsigned char)*in);
    }
    __host__ cluster::cluster(Common::Mean *means, unsigned int len) : means(means), len(len){}
    INLINE __host__ void cluster::prepare(const unsigned int s){
        Common::Mean* tmp;
        cudaMalloc(&tmp, sizeof(Common::Mean)*len);
        cudaMemcpy(tmp, means, sizeof(Common::Mean)*len, cudaMemcpyHostToDevice);
        means=tmp;
    }
    INLINE __host__ void cluster::destroy(){
        cudaFree(means);
    }
    INLINE __device__ void cluster::run(const unsigned int ix, const unsigned int iy, const unsigned int kix, const unsigned int kiy, const unsigned int rix, const unsigned int riy,const unsigned int& x, const unsigned int& y,  const unsigned char *in, const unsigned int& kernelDim,
                                      void *out,  void *kernel){}

    INLINE __device__ void cluster::post(const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int&y,const unsigned char * in,unsigned char *out){
        if(isHigherMean(closestMean(in[iy*x+ ix])))
            atomicAdd((unsigned char*)&out[iy*x+ix], (unsigned char)255);
    }
    INLINE __device__ bool cluster::isHigherMean(Common::Mean *mean) {
        for (int i = 0; i < len; i++)
            if (mean->finalSum < means[i].finalSum)
                return false;
        return true;
    }
    INLINE __device__ Common::Mean *cluster::closestMean(float value) {
        Common::Mean *closest = nullptr;
        auto closestValue = (double) 256;
        for (int i = 0; i < len; i++)
            if (abs(value - means[i].finalSum) < closestValue) {
                closest = &means[i];
                closestValue = abs(value - means[i].finalSum);
            }
        return closest;
    }
//    INLINE __host__ void gaussianMean::prepare(unsigned int s) {
//        gc(cudaMalloc(&sum, s * sizeof(double)));
//        gc(cudaMemset(sum, 0.0, s * sizeof(double)));
//        Common::Mean *tmp=means;
//        gc(cudaMalloc(&means, sizeof(Common::Mean) * len));
//        gc(cudaMemcpy(means, tmp, sizeof(Common::Mean) * len, cudaMemcpyHostToDevice));
//    }
//    INLINE __host__ void gaussianMean::destroy() {
//        gc(cudaFree((means)));
//        gc(cudaFree((sum)));
//    }
//    __host__ gaussianMean::gaussianMean(Common::Mean *means, unsigned int size) {
//        this->means = means;
//        this->len = size;
//    }
//    INLINE __device__ void
//    gaussianMean::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
//                      unsigned char *out, const double *kernel) {
//        atomicAdd(&sum[i], in[k] * kernel[j]);
//    }
//    INLINE __device__ void gaussianMean::post(const unsigned int i, unsigned char *out) {
//        atomicExch(reinterpret_cast<float *>(&sum[i]), sum[i] <= 255.0 ? (sum[i] >= 0.0 ? sum[i] : 0.0) : 255.0);
//        atomicCASChar(&out[i], out[i], (uint8_t) (isHigherMean(closestMean(sum[i])) ? 255 : 0));
//    }
//    INLINE __device__ bool cluster::isScarceMean(Common::Mean *mean) {
//        for (int i = 0; i < len; i++)
//            if (mean->countSum < means[i].countSum)
//                return false;
//        return true;
//    }
//    template __global__ void kernelLoopWrapper<unsigned char, unsigned char, double, Kernel::gaussianMean>(unsigned char*, unsigned char*, double*, struct Kernel::gaussianMean*, const unsigned int, const unsigned int, const unsigned int );
//    template __global__ void kernelLoopWrapper<unsigned char ,unsigned char , double,Kernel::dilate> (unsigned char*, unsigned char*, double *, struct Kernel::dilate*, const unsigned int, const unsigned int, const unsigned int);
//    template __global__ void kernelLoopWrapper<unsigned char, unsigned char ,double,Kernel::erode> (unsigned char* , unsigned char* , double *, struct Kernel::erode* , const unsigned int, const unsigned int, const unsigned int);
//    template __global__ void kernelLoopWrapper<unsigned char, unsigned int , double,Kernel::distribution> (unsigned char* , unsigned int*, double *, struct Kernel::distribution*, const unsigned int, const unsigned int, const unsigned int);
//    template __global__ void kernelLoopWrapper<unsigned char, unsigned char, double, Kernel::grayScale>(unsigned char *, unsigned char*, double*, struct grayScale*, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char* Executor<unsigned char, unsigned char, Kernel::grayScale>(unsigned char* in, void *kernel, Kernel::grayScale && algo, unsigned int x, unsigned int y, unsigned int z,
                                                                                                unsigned int outSize, unsigned int kernelDim);
    template __host__ unsigned int* Executor<unsigned char, unsigned int, Kernel::distribution>(unsigned char* in, void *kernel, Kernel::distribution && algo, unsigned int x, unsigned int y, unsigned int z,
                                                                                               unsigned int outSize, unsigned int kernelDim);
    template __host__ unsigned char* Executor<unsigned char, unsigned char, Kernel::gaussian, float, float>(unsigned char* in, float *kernel, Kernel::gaussian && algo, unsigned int x, unsigned int y, unsigned int z,
                                                                                               unsigned int outSize, unsigned int kernelDim);
    template __host__ unsigned char* Executor<unsigned char, unsigned char, Kernel::cluster>(unsigned char* in, void *kernel, Kernel::cluster && algo, unsigned int x, unsigned int y, unsigned int z,
                                                                                               unsigned int outSize, unsigned int kernelDim);
    template __host__ unsigned char* Executor<unsigned char, unsigned char, Kernel::dilate, void, unsigned int>(unsigned char* in, void *kernel, Kernel::dilate && algo, unsigned int x, unsigned int y, unsigned int z,
                                                                                               unsigned int outSize, unsigned int kernelDim);
    template __host__ unsigned char* Executor<unsigned char, unsigned char, Kernel::erode, void, unsigned int>(unsigned char* in, void *kernel, Kernel::erode && algo, unsigned int x, unsigned int y, unsigned int z,
                                                                                               unsigned int outSize, unsigned int kernelDim);
}

