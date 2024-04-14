//
// Created by jonossar on 3/14/24.
//
#include "../include/Kernel.cuh"
#define IMG1_KERNELFUNCTIONS_IMPLEMENTATION
//#define INLINE __forceinline__

namespace Kernel {
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
    template<typename I, typename O, typename K, typename A>
    __global__  void
    kernelLoopWrapper(I *in, O *out, K *kernel, A * alg, const unsigned int x, const unsigned int y,
                      const unsigned int kernelDim) {
        const unsigned int idx_y=(threadIdx.y + blockIdx.y*blockDim.y);
        const unsigned int idx_x=threadIdx.x + blockIdx.x*blockDim.x;
        if (idx_y >= y || idx_x>=x) return;
        const unsigned int flatIdx=idx_y*x+idx_x;
        for(unsigned int i=0; i<kernelDim; i++) {
            for (unsigned int j = 0; j < kernelDim; j++) {
                int y_pos = (int) idx_y + i - kernelDim / 2;
                int x_pos = (int) idx_x + j - kernelDim / 2;
                x_pos = x_pos >= 0 ? (x_pos < x ? x_pos : (x - 1)) : 0;
                y_pos = y_pos >= 0 ? (y_pos < y ? y_pos : (y - 1)) : 0;
                alg->run(flatIdx, y_pos * x + x_pos, i * kernelDim + j, in, out, kernel);
            }
        }
        alg->post(flatIdx,out);
    }

    template<typename I, typename O, typename K, typename A>
    __host__ O *
    kernelFunctionsWrapper(I *in, K *kernel, A algo, const unsigned int x, const unsigned int y, const unsigned int z,
                           const unsigned int outSize, const unsigned int kernelDim) {
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
        K *kernelGpu;
        A *algoGpu;
        cudaStream_t stream;
        gc(cudaStreamCreate(&stream));
        algo.prepare(x*y);
        gc(cudaGetLastError());
        gc(cudaMalloc(&algoGpu, sizeof(A)));
        gc(cudaMemcpy(algoGpu, &algo, sizeof(A), cudaMemcpyHostToDevice));
        int kernelSize = (int) round(pow(kernelDim, 2));
        gc(cudaMalloc(&(outGpu), outSize * sizeof(O)));
        gc(cudaMemset(outGpu,  0, outSize * sizeof(O)));
        if (kernel != nullptr) {
            gc(cudaMalloc(&kernelGpu, (kernelSize) * sizeof(double)));
            gc(cudaMemcpy(kernelGpu, kernel, kernelSize * sizeof(double), cudaMemcpyHostToDevice));
        }
        gc(cudaMalloc(&inGpu, x * y * z * sizeof(I)));
        gc(cudaMemcpy(inGpu, in, x * y * z * sizeof(I), cudaMemcpyHostToDevice));
        gc(cudaStreamSynchronize(stream));
//        cudaDeviceSynchronize();
        kernelLoopWrapper<I, O, K, A> <<<blocks, threads>>>(inGpu, outGpu, kernelGpu, algoGpu, x, y,
                                                             kernelDim > 0 ? kernelDim : 1);
//        cudaDeviceSynchronize();
        gc(cudaStreamSynchronize(stream));
        gc(cudaStreamDestroy(stream));
        cudaError err = cudaGetLastError();
//        printf("%s::%d::%d\n", cudaGetErrorString(err), outSize, kernelSize);
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
     INLINE __host__ void distribution::prepare(const unsigned int s) {
        gc(cudaMalloc(&(this->sum), s * sizeof(double)));
        gc(cudaMemset((this->sum), 0, s * sizeof(double)));
    }
    INLINE __device__ void
    distribution::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                      unsigned int *out, const double *kernel) {
        atomicAdd(&sum[i], in[k] * kernel[j]);
    }
    INLINE __device__ void distribution::post(const unsigned int i, unsigned int *out) {
        atomicAdd(&out[(int) (sum[i] <= 255.0 ? sum[i] : 255.0)], 1);
    }
    INLINE __host__ void distribution:: destroy(){
        gc(cudaFree(sum));
    }
    __host__ void erode::prepare(unsigned int s) {
        gc(cudaMalloc(&flag, s * sizeof(bool)));
        gc(cudaMemset(flag, false, s * sizeof(bool)));
    }
    INLINE __host__ void erode::destroy(){
        gc(cudaFree(flag));
    }
    INLINE __device__ void
    erode::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
               unsigned char *out, const double *kernel) {
        if (in[k] == 255)
            flag[i] = true;
    }
    INLINE __device__ void erode::post(const unsigned int i, unsigned char *out) {
        if (flag[i]) atomicCASChar((uint8_t *) &(out[i]), (uint8_t) out[i], (uint8_t) 255);
        else atomicCASChar((uint8_t *) &(out[i]), (uint8_t) out[i], (uint8_t) 0);
    }
    __host__ void dilate::prepare(unsigned int s) {
        gc(cudaMalloc(&flag, s * sizeof(bool)));
        gc(cudaMemset(flag, false, s * sizeof(bool)));
    }
    INLINE __device__ void
    dilate::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                unsigned char *out, const double *kernel) {
        if (in[k] == 0)
            flag[i] = true;
    }
    INLINE __device__ void dilate::post(const unsigned int i, unsigned char *out) {
        if (flag[i]) atomicCASChar((uint8_t *) &(out[i]), (uint8_t) out[i], (uint8_t) 0);
        else atomicCASChar((uint8_t *) &(out[i]), (uint8_t) out[i], (uint8_t) 255);
    }
    INLINE __host__ void dilate::destroy(){
       gc(cudaFree(flag));
    }
    __host__ void grayScale::prepare(unsigned int s) {
    }
    INLINE __host__ void grayScale::destroy(){}
    INLINE __device__ void
    grayScale::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                   unsigned char *out, const double *kernel) {
        char val= (char)(in[i * 3] * 0.21 + in[i * 3 + 1]* 0.72 + in[i * 3 + 2]* 0.07);
        atomicAddChar((char *) &out[i], val);
    }
    INLINE __device__ void grayScale::post(const unsigned int i, unsigned char *out) {}

    INLINE __host__ void gaussianMean::prepare(unsigned int s) {
        gc(cudaMalloc(&sum, s * sizeof(double)));
        gc(cudaMemset(sum, 0.0, s * sizeof(double)));
        Common::Mean *tmp=means;
        gc(cudaMalloc(&means, sizeof(Common::Mean) * len));
        gc(cudaMemcpy(means, tmp, sizeof(Common::Mean) * len, cudaMemcpyHostToDevice));
    }
    INLINE __host__ void gaussianMean::destroy() {
        gc(cudaFree((means)));
        gc(cudaFree((sum)));
    }
    __host__ gaussianMean::gaussianMean(Common::Mean *means, unsigned int size) {
        this->means = means;
        this->len = size;
    }
    INLINE __device__ void
    gaussianMean::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                      unsigned char *out, const double *kernel) {
        atomicAdd(&sum[i], in[k] * kernel[j]);
    }
    INLINE __device__ void gaussianMean::post(const unsigned int i, unsigned char *out) {
        atomicExch(reinterpret_cast<float *>(&sum[i]), sum[i] <= 255.0 ? (sum[i] >= 0.0 ? sum[i] : 0.0) : 255.0);
        atomicCASChar(&out[i], out[i], (uint8_t) (isHigherMean(closestMean(sum[i])) ? 255 : 0));
    }
    INLINE __device__ bool gaussianMean::isScarceMean(Common::Mean *mean) {
        for (int i = 0; i < len; i++)
            if (mean->countSum < means[i].countSum)
                return false;
        return true;
    }
    INLINE __device__ bool gaussianMean::isHigherMean(Common::Mean *mean) {
        for (int i = 0; i < len; i++)
            if (mean->finalSum < means[i].finalSum)
                return false;
        return true;
    }
    INLINE __device__ Common::Mean *gaussianMean::closestMean(double value) {
        Common::Mean *closest = nullptr;
        auto closestValue = (double) 256;
        for (int i = 0; i < len; i++)
            if (abs(value - means[i].finalSum) < closestValue) {
                closest = &means[i];
                closestValue = abs(value - means[i].finalSum);
            }
        return closest;
    }
    template __global__ void kernelLoopWrapper<unsigned char, unsigned char, double, Kernel::gaussianMean>(unsigned char*, unsigned char*, double*, struct Kernel::gaussianMean*, const unsigned int, const unsigned int, const unsigned int );
    template __global__ void kernelLoopWrapper<unsigned char ,unsigned char , double,Kernel::dilate> (unsigned char*, unsigned char*, double *, struct Kernel::dilate*, const unsigned int, const unsigned int, const unsigned int);
    template __global__ void kernelLoopWrapper<unsigned char, unsigned char ,double,Kernel::erode> (unsigned char* , unsigned char* , double *, struct Kernel::erode* , const unsigned int, const unsigned int, const unsigned int);
    template __global__ void kernelLoopWrapper<unsigned char, unsigned int , double,Kernel::distribution> (unsigned char* , unsigned int*, double *, struct Kernel::distribution*, const unsigned int, const unsigned int, const unsigned int);
    template __global__ void kernelLoopWrapper<unsigned char, unsigned char, double, Kernel::grayScale>(unsigned char *, unsigned char*, double*, struct grayScale*, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * kernelFunctionsWrapper<unsigned char, unsigned char, double, Kernel::grayScale>(unsigned char *, double *, struct Kernel::grayScale , const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * kernelFunctionsWrapper<unsigned char, unsigned char, double, Kernel::dilate>(unsigned char *, double *, struct Kernel::dilate , const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * kernelFunctionsWrapper<unsigned char, unsigned char, double, Kernel::erode>(unsigned char *, double *, struct Kernel::erode , const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * kernelFunctionsWrapper<unsigned char, unsigned char, double, Kernel::gaussianMean>(unsigned char *, double *, struct Kernel::gaussianMean , const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned int * kernelFunctionsWrapper<unsigned char, unsigned int, double, Kernel::distribution>(unsigned char *, double *, struct Kernel::distribution , const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
}

