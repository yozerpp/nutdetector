//
// Created by jonossar on 3/14/24.
//
#include "../include/Kernel.cuh"
#include "../include/KernelCommons.cuh"
#define USE_THRUST false
#if USE_THRUST
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#endif
#include <cmath>

#define IMG1_KERNELFUNCTIONS_IMPLEMENTATION
//#define inline __forceinline__

namespace Kernel {

    /** this is a wrapper function to perform convolution in gpu.
     * it executes @p run() method of the algorithm instance @p alg during convolution and @p post() instance after convolution, for algorithms that doesnt require convolution only @p post() is executed (this is ensured by passing @p kernelDim=1) you can find more about algorithms/functors in @p Kernel.cuh header file
     * @tparam I in type
     * @tparam O out type
     * @tparam K kernel type
     * @tparam A type of the algorithm(i.e. grayScale)
     * @param in
     * @param out
     * @param kernel
     * @param alg algoritm/functor instance to be executed
     * @param x
     * @param y
     * @param kernelDim
     */
    template<typename I, typename O, typename K, typename A>
    __global__  void
    convolutionWrapper(const I *in, O *out, const K *kernel, A * alg, const unsigned int x, const unsigned int y,
                       const unsigned int kernelDim) {
        const int idx_y=threadIdx.y + blockIdx.y*blockDim.y;
        const int idx_x=threadIdx.x + blockIdx.x*blockDim.x;
        if (idx_y >= y || idx_x>=x) return;
        const unsigned int flatIdx=idx_y*x+idx_x;
        for( int i=0; i<kernelDim; i++) {
            for ( int j = 0; j < kernelDim; j++) {
                int y_pos =  idx_y + i - (int)kernelDim / 2;
                int x_pos = idx_x + j - (int)kernelDim / 2;
                x_pos = x_pos > 0 ? (x_pos < (x-1) ? x_pos : ((int)x - 1)) : 0;
                y_pos = y_pos > 0 ? (y_pos < (y-1) ? y_pos : ((int)y - 1)) : 0;
                alg->run(flatIdx, y_pos * x + x_pos, i * kernelDim + j, in, out, kernel);
            }
        }
        alg->post(flatIdx,in,out, idx_x, idx_y,x,y);
    }
    static std::mutex mtx;
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
    template<typename I, typename O, typename K, typename A>
    __host__ O *
    executor(I *in, K *kernel, A && algo, const unsigned int x, const unsigned int y, const unsigned int z,
             const unsigned int outSize, const unsigned int kernelDim) {
        if(streams.find(std::this_thread::get_id())==streams.end()) {
            auto *s =new cudaStream_t;
            cudaStreamCreate(s);
            streams.insert_or_assign(std::this_thread::get_id(),s);
        }
        dim3 threads(KERNEL_THREADS, KERNEL_THREADS, 1);
        cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, SIZE_MAX);
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
        algo.prepare(x*y);
        ERC(cudaGetLastError());
        ERC(cudaMalloc(&algoGpu, sizeof(A)));
        ERC(cudaMemcpy(algoGpu, &algo, sizeof(A), cudaMemcpyHostToDevice));
        int kernelSize = (int) round(pow(kernelDim, 2));
        ERC(cudaMalloc(&(outGpu), outSize * sizeof(O)));
        ERC(cudaMemsetAsync(outGpu, 0, outSize * sizeof(O), thisStream));
        if (kernel != nullptr) {
            ERC(cudaMalloc(&kernelGpu, (kernelSize) * sizeof(K)));
            ERC(cudaMemcpy(kernelGpu, kernel, kernelSize * sizeof(K), cudaMemcpyHostToDevice));
        }
        ERC(cudaMalloc(&inGpu, x * y * z * sizeof(I)));
        ERC(cudaMemcpy(inGpu, in, x * y * z * sizeof(I), cudaMemcpyHostToDevice));
//        if(kernelDim<10)
        {
            std::scoped_lock<std::mutex> lck(mtx);
            convolutionWrapper<I, O, K,
                    A> <<<blocks, threads, 0, thisStream>>>(inGpu, outGpu, kernelGpu, algoGpu, x, y,
                                                            kernelDim > 0 ? kernelDim : 1);
        }
//        else
//            cruncher(inGpu,outGpu,kernelGpu,algoGpu,x,y,kernelDim).crunch();
        cudaStreamSynchronize(thisStream);
        ERC(cudaGetLastError());
        O *out = new O[outSize];
        ERC(cudaMemcpy(out, outGpu, outSize * sizeof(O), cudaMemcpyDeviceToHost));
        ERC(cudaFree(outGpu));
        ERC(cudaFree(inGpu));
        if (kernel != nullptr)
            ERC(cudaFree(kernelGpu));
        algo.destroy();
        ERC(cudaFree(algoGpu));
        return out;
    }
    template<typename T, typename O, typename K>
    struct kernelFunctor {
    inline __host__ void prepare(const unsigned int s){}
    inline __host__ void destroy(){}
    inline __host__ void run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                             T *out, const double *kernel){}
        };
    template <typename T>
    distribution<T>::distribution(T origin,T bound):origin(origin), bound(bound){}
    template <typename T>
     inline __host__ void distribution<T>::prepare(const unsigned int s) {
        ERC(cudaMalloc(&(this->sum), s * sizeof(double)));
        ERC(cudaMemset((this->sum), 0, s * sizeof(double)));
    }
    template <typename T>
    inline __device__ void
    distribution<T>::run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                      unsigned int *out, const double *kernel) {}
    template <typename T>
    inline __device__ void distribution<T>::post(const unsigned int i,const T *in, unsigned int *out,const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y) {
        atomicAdd(&out[(unsigned int)(in[i] <= bound ? in[i] : bound)],1);
    }
    template <typename T>
    inline __host__ void distribution<T>:: destroy(){
        ERC(cudaFree(sum));
    }
    __host__ void erode::prepare(unsigned int s) {
        ERC(cudaMalloc(&flag, s * sizeof(bool)));
        ERC(cudaMemset(flag, false, s * sizeof(bool)));
    }
    inline __host__ void erode::destroy(){
        ERC(cudaFree(flag));
    }
    inline __device__ void
    erode::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
               unsigned char *out, const double *kernel) {
        if (in[k] == 255)
            flag[i] = true;
    }
    inline __device__ void erode::post(const unsigned int i, const unsigned char *in,unsigned char *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y) {
        if (flag[i]) out[i]=(unsigned char)255;
        else out[i]= (unsigned char)0;
    }
    __host__ void dilate::prepare(unsigned int s) {
        ERC(cudaMalloc(&flag, s * sizeof(bool)));
        ERC(cudaMemset(flag, false, s * sizeof(bool)));
    }
    inline __device__ void
    dilate::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                unsigned char *out, const double *kernel) {
        if (in[k] == 0)
            flag[i] = true;
    }
    inline __device__ void dilate::post(const unsigned int i, const unsigned char *in,unsigned char *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y) {
        if (flag[i]) out[i]=(unsigned char) 0;
        else out[i]=(unsigned char) 255;
    }
    inline __host__ void dilate::destroy(){
       ERC(cudaFree(flag));
    }
    __host__ void grayScale::prepare(unsigned int s) {
    }
    inline __host__ void grayScale::destroy(){}
    inline __device__ void
    grayScale::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                   unsigned char *out, const double *kernel) {}
    inline __device__ void grayScale::post(const unsigned int i, const unsigned char *in,unsigned char *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y) {
        unsigned int val= in[i * 3] * 0.21 + in[i * 3 + 1]* 0.72 + in[i * 3 + 2]* 0.07;
        atomicCAS((uint8_t*) &out[i],(uint8_t)0, (uint8_t)(val<=255?val:255));
    }
    __host__ Kernel::edge::hysteresis::hysteresis(Common::Mean* means, unsigned int len, unsigned int* count):means(means), len(len), count(count){
        std::sort(std::vector<Common::Mean>::iterator(means), std::vector<Common::Mean>::iterator(&(means[len])));
    }
    inline __host__ void Kernel::edge::hysteresis::destroy(){
        cudaFree(means);
    }
    inline __host__ void Kernel::edge::hysteresis::prepare(unsigned int s){
        using namespace Common;
        Mean * tmp;
        cudaMalloc(&tmp, sizeof(Mean)*len);
        cudaMemcpy(tmp,means, sizeof(Mean)*len,cudaMemcpyHostToDevice);
        means=tmp;
    }
    inline __device__ void Kernel::edge::hysteresis::run(const unsigned int i, const unsigned int k, const unsigned int j, const gradient *in,
                               gradient *out, const double *kernel){}
    inline __device__ void Kernel::edge::hysteresis::post(const unsigned int i, const gradient *in,gradient *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y){
        unsigned int pos= position(in[i]);
        if(pos ==len)
            out[i]=in[i];
        else if(pos <len && pos > 0) {
            gradient neighbours[2]={FLT_MAX,FLT_MAX};
            Kernel::edge::gradient::getNeighbours(in, neighbours, true, in[i].axis, ix, iy, x, y);
            bool f=false;
            for(int it=0;it<2; it++ ){
                unsigned int p=position(neighbours[it]);
                if(!f)f= (neighbours[it]!=FLT_MAX &&p==len);
            }
            if(f){
                out[i].value=means[len-1].finalSum;
                out[i].axis=in[i].axis;
                atomicAdd(count,1);
            } else out[i]=in[i];

        }
    }
    inline __device__ unsigned int Kernel::edge::hysteresis::position(const Kernel::edge::gradient value){
        unsigned int ret;
        float closestDist=FLT_MAX;
        for(unsigned int i=0; i < len; i++){
            float dist= value.value - means[i].finalSum;
            if(closestDist>abs(dist)){
                ret=dist>=0?i+1:i;
                closestDist=abs(dist);
            }
        }
        return ret;
    }
    template <typename T>
    __host__ void gaussian<T>::prepare(unsigned int s) {
        cudaMalloc(&sum, sizeof(float)* s);
        cudaMemset(sum,0, sizeof(float)*s);
    }
    template <typename T>
    inline __host__ void gaussian<T>::destroy(){
        cudaFree(sum);
    }
    template <typename T>
    inline __device__ void
    gaussian<T>::run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                      T *out, const float *kernel) {
        atomicAdd_system((float*)&sum[i], ((float)in[k])* kernel[j]);
    }
    template <typename T>
    inline __device__ void Kernel::gaussian<T>::post(const unsigned int i, const T *in,T *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y) {
//        atomicCAS((T*)&out[i], 0, (T) (sum[i] <= 255 ? (sum[i] >= 0 ? sum[i] : 0) : 255));
        out[i]=(unsigned char)(sum[i] <= 255 ? (sum[i] >= 0 ? sum[i] : 0) : 255);
    }
    template <typename T, typename O>
    inline __host__ void Kernel::cluster<T,O>::prepare(unsigned int s) {
        Common::Mean *tmp=means;
        ERC(cudaMalloc(&tmp, sizeof(Common::Mean) * numMeans));
        ERC(cudaMemcpy(tmp, means, sizeof(Common::Mean) * numMeans, cudaMemcpyHostToDevice));
        means=tmp;
    }
    template <typename T, typename O>
    inline __host__ void Kernel::cluster<T,O>::destroy() {
        ERC(cudaFree((means)));
    }
    template <typename T, typename O>
    __host__ Kernel::cluster<T,O>::cluster(Common::Mean *means, unsigned int size, O origin, O bound, unsigned int threshold):means(means), numMeans(size), max(bound), min(origin), threshold(threshold) {
        ;
    }
    template <typename T, typename O>
    inline __device__ void
    Kernel::cluster<T,O>::run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                         O *out, const double *kernel) {}
    template <typename T, typename O>
    inline __device__ void Kernel::cluster<T,O>::post(const unsigned int i,const T *in,O *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y) {
        out[i]= isAboveThreshold(closestMean((float) in[i])) ? max : min;
    }
//    template <typename T, typename O>
//    inline __device__ bool Kernel::cluster<T,O>::isScarceMean(Common::Mean *mean) {
//        for (int i = 0; i < numMeans; i++)
//            if (mean->countSum < means[i].countSum)
//                return false;
//        return true;
//    }
    template <typename T, typename O>
    inline __device__ bool Kernel::cluster<T,O>::isAboveThreshold(Common::Mean *mean) {
        bool ret=true;
        for (int i = 0; i < numMeans; i++){
            if(i <= threshold)
                ret= mean->finalSum >= means[i].finalSum;
        }
        return ret;
    }
    template <typename T, typename O>
    inline __device__ Common::Mean *Kernel::cluster<T,O>::closestMean(float value) {
        Common::Mean *closest = nullptr;
        auto closestDist = (float) FLT_MAX;
        for (int i = 0; i < numMeans; i++) {
            auto dist = abs(value - means[i].finalSum);
            if (dist < closestDist) {
                closest = &means[i];
                closestDist = dist;
            }
        }
        return closest;
    }
    __host__ edge::getGradient::getGradient(float **masks): masks((float*) malloc(sizeof(float) * SOBEL_MASK_SIZE * SOBEL_MASK_VARIANTS)){
        for(int i=0, k=0; i < SOBEL_MASK_VARIANTS; i++, k+=SOBEL_MASK_SIZE) {
            memcpy(&this->masks[k], masks[i], sizeof(int) * SOBEL_MASK_SIZE);
            delete[]masks[i];
        }
        delete masks;
    }

    inline __host__ void edge::getGradient::prepare(unsigned int s){
        float* temp;
        ERC(cudaMalloc(&temp, sizeof(float) * SOBEL_MASK_VARIANTS * SOBEL_MASK_SIZE));
        ERC(cudaMemcpy(temp, masks, sizeof(float) * SOBEL_MASK_VARIANTS * SOBEL_MASK_SIZE, cudaMemcpyHostToDevice));
        ERC(cudaMalloc(&sum, sizeof(float) * SOBEL_MASK_VARIANTS * s))
        cudaMemsetAsync(sum,0, sizeof(float)*SOBEL_MASK_VARIANTS*s,thisStream);
        cudaStreamSynchronize(thisStream);
        ERC(cudaGetLastError())
        delete[]masks;
        masks=temp;
    }
    inline __host__ void edge::getGradient::destroy(){
        cudaFree(masks);
        cudaFree(sum);
    }
    inline __device__ void edge::getGradient::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                                               Kernel::edge::gradient *out, const double *kernel){
        for(int off=0; off<SOBEL_MASK_VARIANTS; off++){
            sum[i*2+off]+=masks[off*SOBEL_MASK_SIZE+j]*in[k];
        }
    }

    inline __device__ void edge::getGradient::post(const unsigned int i, const unsigned char *in, Kernel::edge::gradient *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y){
        unsigned int direction;
        float angle=atan2f(sum[i*2+1],sum[i*2])*(float)(180.0/M_PI);
        switch ((unsigned int)(((((int)angle)+360)%180)/22.5)) {
            case 7:
            case 0:
                direction=GRADIENT_0;
                break;
            case 1:
            case 2:
                direction=GRADIENT_45;
                break;
            case 3:
            case 4:
                direction=GRADIENT_90;
                break;
            case 5:
            case 6:
                direction=GRADIENT_135;
                break;
        }
        out[i].value=hypotf(sum[i*2],sum[i*2+1]);
        out[i].axis= direction;
    }

    inline __host__ void edge::nonMaxSuppress::destroy() {}
    inline __host__ void edge::nonMaxSuppress::prepare(unsigned int s){}
    static inline __device__ bool inRange(const int& ix,const int& iy, const unsigned int& limx,const unsigned int& limy){
        return (ix >= 0 && ix < limx)&&(iy>=0 && iy<limy);
    }
    __device__ inline void edge::gradient::getNeighbours(const edge::gradient* all, edge::gradient* neighbours, bool orthogonal, unsigned int axis, const unsigned int& ix, const unsigned int& iy, const unsigned int& x, const unsigned int& y){
        int xOff[2]={0,0};
        int yOff[2]{0,0};
        if(orthogonal)
            axis=axis+2%4;
        switch (axis) {
            case GRADIENT_0:
                xOff[0]=-1;xOff[1]=1;
//                yOff[0]=yOff[1]=0;
                break;
            case GRADIENT_45:
                xOff[0]=-1;xOff[1]=1;
                yOff[0]=1;yOff[1]=-1;
                break;
            case GRADIENT_90:
//                xOff[0]=xOff[1]=0;
                yOff[0]=1;yOff[1]=-1;
                break;
            case GRADIENT_135:
                xOff[0]=1;xOff[1]=-1;
                yOff[0]=1;yOff[1]=-1;
            break;
        }
        if(inRange((int)ix+xOff[0],(int)iy+yOff[0],x,y))
            neighbours[0]= all[((int)iy + yOff[0]) * x + (int)ix + xOff[0]];
        if(inRange((int)ix+xOff[1],(int)iy+yOff[1],x,y))
            neighbours[1]= all[((int)iy + yOff[1]) * x + (int)ix + xOff[1]];
    }
    inline __device__ void edge::nonMaxSuppress::run(const unsigned int i, const unsigned int k, const unsigned int j, const gradient *in,
                                                     gradient*out, const double *kernel){}
    inline __device__ void edge::nonMaxSuppress::post(const unsigned int i, const gradient *in, gradient *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y){
        gradient neighbours[2]={-FLT_MAX,-FLT_MAX};
        Kernel::edge::gradient::getNeighbours(in,neighbours,false, in[i].axis, ix, iy, x, y);
        bool f=true;
        for(int it=0; it<2; it++)
             if(f) f=(in[i]>neighbours[it]);
        if(f)
            out[i]=in[i];
    }
    inline __host__ void edge::houghLine::destroy() {}
    inline __host__ void edge::houghLine::prepare(unsigned int s) {}
    inline __device__ void edge::houghLine::run(const unsigned int i, const unsigned int k, const unsigned int j, const unsigned char *in,
                               unsigned int *out, const double *kernel) {}
    static __global__ void calculateHoughLine(unsigned int * out, const unsigned int ix, const unsigned int iy) {
        const int angleIdx=((int)(threadIdx.x + blockIdx.x*blockDim.x));
        const int angle=(angleIdx+1);
        if(angle <=0 || angle >= 360) return;
        const int rho= ix * cos((angle / 360) * 2 * M_PI) + iy * sin((angle / 360) * 2 * M_PI);
        if(rho <= 0) return;
        atomicAdd_system(&out[rho * 360 + angleIdx],1);
    }
    inline __device__ void edge::houghLine::post(const unsigned int i, const unsigned char *in,unsigned int *out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y) {
        if(in[i]==255) {
            calculateHoughLine<<<dim3(1, 1, 1), dim3(360, 1, 1), 0, cudaStreamFireAndForget>>>(out, ix, iy);
//            ERC(cudaGetLastError())
        }
    }
    template<typename T>
    inline __host__ void multiply<T>::destroy(){
        cudaFree(val);
    }
    template<typename T>
    inline __host__ void multiply<T>::prepare(unsigned int s){
        T* tmp;
        cudaMalloc(&tmp, sizeof(T));
        cudaMemcpy(tmp, val, sizeof(T),cudaMemcpyHostToDevice);
        delete val;
        val=tmp;
    }
    template<typename T, typename O>
    __host__ O* normalize(T* arr, unsigned int len, T * val, T bound){
        auto * ret=(O*)malloc(sizeof(O)*len);
#if USE_THRUST
        thrust::device_vector<T> dev(numMeans);
        ERC(cudaMemcpy(dev.begin().base().get(),arr, sizeof(T)*numMeans,cudaMemcpyHostToDevice));
        T v{};
        if (val== nullptr) {
            T max=*thrust::max_element(thrust::device, dev.begin(), dev.end());
            v = ((T) 255) / max;
        }
        else v=*val;
        thrust::transform(thrust::device,dev.begin(),dev.end(),dev.begin(),thrust::placeholders::_1*v);
        ERC(cudaMemcpy(ret,dev.begin().base().get(), sizeof(T)*numMeans, cudaMemcpyDeviceToHost));
        dev.clear();
#else
        float v;
        if(val == nullptr) {
            T max = arr[0];
            for (int i = 0; i < len; i++) {
                if (arr[i] > max) max = arr[i];
            }
            v= (float)bound / (float)max;
        } else v=(float)*val;
        for(int i=0; i<len; i++) {
            ret[i]=std::ceil(v*(float)arr[i]);
        }
#endif
        return ret;
    }

#undef USE_THRUST
    template <typename T>
    __host__ T* sort(T* arr, unsigned int len){
//        T* dev;
//        cudaMalloc(&dev, sizeof(T)*numMeans);
//        thrust::copy(thrust::host,typename std::vector<T>::iterator(arr),typename std::vector<T>::iterator(&arr[numMeans]),dev.begin());
//        ERC(cudaMemcpy(dev,arr, sizeof(T)*numMeans,cudaMemcpyHostToDevice));
        auto * ret=new T[len];
        memcpy(ret, arr, sizeof(T)*len);
        std::sort(typename std::vector<T>::iterator(ret), typename std::vector<T>::iterator(&ret[len]));
//        cudaStreamSynchronize(thisStream);
//        ERC(cudaGetLastError())

//        ERC(cudaMemcpy(ret,dev, sizeof(T)*numMeans, cudaMemcpyDeviceToHost));
//        cudaFree(dev);
        return ret;
    }

    template<typename T>
    inline __device__ void multiply<T>::run(const unsigned int i, const unsigned int k, const unsigned int j, const T *in,
                                            T*out, const double *kernel){}
    template<typename T>
    inline __device__ void multiply<T>::post(const unsigned int i,const T *in, T*out, const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y){
        atomicExch(&out[i], in[i]*(*val));
    }
    //UNUSED CODE GOES THERE
//    template <typename I, typename O, typename K, typename A>
//    __global__ void convolutionWrapper(const I* in, O* out, const K* kernel, A* alg, const unsigned int ix,const unsigned int iy,const unsigned int x, const unsigned int y, const unsigned int kernelDim){
//        const int kiy=threadIdx.y + blockIdx.y*blockDim.y;
//        const int kix=threadIdx.x + blockIdx.x*blockDim.x;
//        if(kix>=kernelDim || kiy>=kernelDim) return;
//        int y_pos=iy+kiy-kernelDim/2;
//        int x_pos=ix+kix-kernelDim/2;
//        x_pos = x_pos > 0 ? (x_pos < (x-1) ? x_pos : ((int)x - 1)) : 0;
//        y_pos = y_pos > 0 ? (y_pos < (y-1) ? y_pos : ((int)y - 1)) : 0;
//        alg->run(iy*x + ix, (y_pos)*x+(x_pos),kiy*kernelDim+kix, in,out, kernel);
//    }
//    template <typename I, typename O, typename A>
//    static __global__ void postWrapper(const I* in, O* out, A* algo,const unsigned int ix, const unsigned int iy, const unsigned int x, const unsigned int y){
//        algo->post(iy*x+ix,in,out,ix,iy,x,y);
//    }
//
//    template <typename I, typename O, typename K, typename A>
//    static __global__ void _crunch(const I* in, O* out,const K* kernel, A* algo, const unsigned int lastX, const unsigned int lastY, const unsigned int x, const unsigned int y, const unsigned int kernelDim, const unsigned int xlim, const unsigned int ylim,const dim3 blocks, const dim3 threads,const unsigned int sharedMem=0){
//        unsigned int iy= threadIdx.y + blockIdx.y*blockDim.y;
//        unsigned int ix= threadIdx.x + blockIdx.x*blockDim.x;
//        if(ix >= xlim || iy >= ylim) return;
//        iy+=lastY;ix+=lastX;
//        if(ix >=x || iy>=y) return;
//        convolutionWrapper<I,O,K,A><<<blocks, threads,sharedMem,cudaStreamFireAndForget>>>(in, out, kernel, algo,ix, iy ,x,y, kernelDim);
//        postWrapper<I,O,A><<<1,1,0,cudaStreamTailLaunch>>>(in, out, algo, ix,iy,x,y);
//    }
//    template <typename I, typename O, typename K, typename A>
//    __host__ class cruncher {
//    public:
//        cruncher (const I* in,O* out,const K* kernel, A* algo, const unsigned int x, const unsigned int y,unsigned int kernelDim,unsigned int batchSize=0): in(in), lastX(0),lastY(0), out(out), kernel(kernel),algo(algo), x(x), y(y), kernelDim(kernelDim),
//                                                                                                                                                            size(batchSize==0?(x*y)/(Common::roundP2(kernelDim)*Common::roundP2(kernelDim)):batchSize), stepX(size*kernelDim/y), stepY(size*kernelDim/x){
//            cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, SIZE_MAX);
//            kernelDim=Common::roundP2(kernelDim);
//            threadsConv={kernelDim>KERNEL_THREADS?KERNEL_THREADS:kernelDim,kernelDim>KERNEL_THREADS?KERNEL_THREADS:kernelDim,1};
//            unsigned int b=kernelDim/threadsConv.x;
//            blocksConv={Common::roundP2(b), Common::roundP2(b),1};
//            unsigned int childDim=blocksConv.x*threadsConv.x;
//            size=x*y;
//            stepX=size/y;
//            stepY=size/x;
//            double x2=Common::roundP2( stepX);
//            double y2=Common::roundP2( stepY);
//            threads={x2>32?32:((unsigned int)x2), y2>32?32:((unsigned int)y2), 1};
//            unsigned int d1 =ceil(x2 / threads.x);
//            unsigned int d2 = ceil(y2 / threads.y);
//            blocks={d1, d2, 1};
//        }
//        void crunch(const unsigned int sharedMem=0){
//            dim3 thread(KERNEL_THREADS, KERNEL_THREADS, 1);
//            double x2=Common::roundP2( x);
//            double y2=Common::roundP2( y);
//            x2=x2>y2?x2:y2;
//            y2=y2>x2?y2:x2;
//            unsigned int d1 = x2 / thread.x;
//            unsigned int d2 = y2 / thread.y;
//            dim3 block(d1, d2, 1);
//            ERC(cudaGetLastError());
//            for(lastY=0; lastY < y;lastY+=stepY){
//                for(lastX=0; lastX<x; lastX+=stepX) {
//                    _crunch < I, O, K, A ><<<blocks, threads, sharedMem, thisStream>>>(in, out, kernel, algo, lastX, lastY, x, y, kernelDim, stepX, stepY, blocksConv, threadsConv);
//                    cudaStreamSynchronize(thisStream);
//                    ERC(cudaGetLastError());
//                }
//            }
//        }
//    private:
//        unsigned int lastX;
//        unsigned int lastY;
//        unsigned int size;
//        const unsigned int x;
//        const unsigned int y;
//        dim3 blocksConv;
//        dim3 threadsConv;
//        dim3 blocks;
//        dim3 threads;
//        const unsigned int kernelDim;
//        unsigned int stepX;
//        unsigned int stepY;
//        const I* in;
//        O* out;
//        const K* kernel;
//        A* algo;
//    };
    template unsigned int* Kernel::sort<unsigned int>(unsigned int*, unsigned int);
    template __host__ unsigned char* Kernel::normalize<unsigned int, unsigned char>(unsigned int*, unsigned int, unsigned int*, unsigned int);
    template __host__ Kernel::edge::gradient* sort<Kernel::edge::gradient>(Kernel::edge::gradient*,unsigned int);
    template __host__ Kernel::edge::gradient* normalize<Kernel::edge::gradient>(Kernel::edge::gradient*,unsigned int,Kernel::edge::gradient*, Kernel::edge::gradient);
    template __host__ Kernel::distribution<unsigned int>::distribution(unsigned int, unsigned int);
    template Kernel::distribution<Kernel::edge::gradient>::distribution(Kernel::edge::gradient, Kernel::edge::gradient);
    template Kernel::distribution<unsigned char>::distribution(unsigned char, unsigned char);
    template Kernel::cluster<unsigned int, unsigned char>::cluster(Common::Mean*, unsigned int, unsigned char, unsigned char, unsigned int);
    template Kernel::cluster<Kernel::edge::gradient, unsigned char>::cluster(Common::Mean*, unsigned int, unsigned char, unsigned char,unsigned int);
    template Kernel::cluster<unsigned char, unsigned char>::cluster(Common::Mean*, unsigned int, unsigned char, unsigned char, unsigned int);
//    template __global__ void convolutionWrapper<unsigned char, unsigned char, double, Kernel::cluster<unsigned char, unsigned char>>(unsigned char*, unsigned char*, double*, struct Kernel::cluster<unsigned char,unsigned char>*, const unsigned int, const unsigned int, const unsigned int );
//    template __global__ void convolutionWrapper<unsigned char ,unsigned char ,double,Kernel::dilate> (unsigned char*, unsigned char*, double *, struct Kernel::dilate*, const unsigned int, const unsigned int, const unsigned int);
//    template __global__ void convolutionWrapper<unsigned char, unsigned char ,double,Kernel::erode> (unsigned char* , unsigned char* , double *, struct Kernel::erode* , const unsigned int, const unsigned int, const unsigned int);
//    template __global__ void convolutionWrapper<unsigned char, unsigned int , double,Kernel::distribution<unsigned char>> (unsigned char* , unsigned int*, double *, struct Kernel::distribution<unsigned char>*, const unsigned int, const unsigned int, const unsigned int);
//    template __global__ void convolutionWrapper<unsigned char, unsigned char, double, Kernel::grayScale>(unsigned char *, unsigned char*, double*, struct grayScale*, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned int* executor<unsigned int, unsigned int, double, Kernel::distribution<unsigned int> >(unsigned int*, double*, Kernel::distribution<unsigned int>&&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    template __host__ unsigned int * executor<unsigned char, unsigned int, double, Kernel::edge::houghLine>(unsigned char *, double *, Kernel::edge::houghLine &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * executor<unsigned int, unsigned char, double, Kernel::cluster<unsigned int, unsigned char>>(unsigned int *, double *, Kernel::cluster<unsigned int,unsigned char> &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__  Kernel::edge::gradient* executor<Kernel::edge::gradient, Kernel::edge::gradient, double, Kernel::edge::hysteresis>(Kernel::edge::gradient*, double*, Kernel::edge::hysteresis&&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    template __host__ unsigned char * executor<edge::gradient, unsigned char, double, Kernel::cluster<edge::gradient, unsigned char>>(edge::gradient *, double *, Kernel::cluster<edge::gradient,unsigned char> &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned int * executor<edge::gradient, unsigned int, double, Kernel::distribution<edge::gradient>>(edge::gradient*, double *, Kernel::distribution<edge::gradient> &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ edge::gradient * executor<edge::gradient, edge::gradient, double, edge::nonMaxSuppress>(edge::gradient *, double *, edge::nonMaxSuppress &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ edge::gradient * executor<unsigned char, edge::gradient, double, edge::getGradient>(unsigned char *, double *, edge::getGradient &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * executor<unsigned char, unsigned char, float, Kernel::gaussian<unsigned char>>(unsigned char *, float *, Kernel::gaussian<unsigned char> &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * executor<unsigned char, unsigned char, double, Kernel::grayScale>(unsigned char *, double *, Kernel::grayScale &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * executor<unsigned char, unsigned char, double, Kernel::dilate>(unsigned char *, double *, Kernel::dilate &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * executor<unsigned char, unsigned char, double, Kernel::erode>(unsigned char*, double *, Kernel::erode &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned char * executor<unsigned char, unsigned char, double, Kernel::cluster<unsigned char,unsigned char>>(unsigned char *, double *, Kernel::cluster<unsigned char,unsigned char> &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    template __host__ unsigned int * executor<unsigned char, unsigned int, double, Kernel::distribution<unsigned char>>(unsigned char*, double *, Kernel::distribution<unsigned char> &&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
}

