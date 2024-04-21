//
// Created by jonossar on 3/14/24.
//

#ifndef IMG1_KERNEL_CUH
#define IMG1_KERNEL_CUH
#ifndef IMG1_COMMON_H
#include "common.h"
#endif
#ifndef IMG1_IMAGE_CUH
#include "Image.cuh"
#endif

namespace Kernel {
    template<typename I,typename  O, typename C>
    class KernelFunctor : public VirtualTransferable<C>{
    public:
       __host__ KernelFunctor(Tensor<I> & in, ui outSize=0);
        __device__ KernelFunctor(KernelFunctor& other);
        __host__ virtual O* operator()();
        O* out;
        __host__ void operator delete(void* p);
        __device__ virtual void operator()(ui ix, ui iy)=0;
        Tensor<I>* data;
        __host__ C* toGpu() override;
    protected:

        ui outSize;
        static __host__ KernelFunctor<I,O,C>* toHost(C* inst);

    private:

    };
    template<typename I,typename O,typename T, typename C>
    class ConvolutionFunctor : public KernelFunctor<I,O,C>{
    public:
        __host__ ConvolutionFunctor(Tensor<I>& in, ui kernelDim=3, ui outSize=0);
        __device__ ConvolutionFunctor(ConvolutionFunctor& other);
        __host__ O* operator()() override;
        T* temp;
        ui kernelDim;
        __device__ virtual void operator()(ui ix, ui iy) override=0;
        __device__ virtual void operator()(ui ix, ui iy, int rix, int riy, ui kix, ui kiy, T* acc)=0;
        __host__ void operator delete(void * p);
    protected:
    private:

    };
    class GrayScale : public KernelFunctor<uc, uc,GrayScale> {
    public:
        __host__ GrayScale(Tensor<uc>& in);
        __device__ GrayScale(GrayScale& other);
        __device__ void operator()(ui ix, ui iy) override;
        __host__  uc* operator()() override;
    protected:
        size_t _size() override{
            return sizeof(*this);
        }
    };
    class Cluster: public KernelFunctor<uc,uc, Cluster>{
    public:
        __host__ Cluster(Tensor<uc>& in,float *means,ui len=2);
        __device__ Cluster(Cluster& other);
        __device__ void operator()(ui ix, ui iy)override;
        __host__  uc* operator()() override;
    protected:
        size_t _size() override{
            return sizeof(*this);
        }
    private:
        float* means;
        ui len;

    };
    class Distribution : public KernelFunctor<uc,ui,Distribution>{
    public:
        __host__ Distribution(Tensor<uc>& in);
        __device__ Distribution(Distribution& other);
        __device__ void operator()(ui ix, ui iy) override;
        __host__  ui* operator()() override;
    protected:
        size_t _size() override{
            return sizeof(*this);
        }
    };
    class Gaussian: public ConvolutionFunctor<uc,uc,float, Gaussian>{
    public:
        __host__ Gaussian(Tensor<uc>& in,ui sigma, ui kernelDim=3);
        __device__ Gaussian(Gaussian& other);
        __device__ void operator()(ui ix, ui iy) override;
        __device__ void operator()(ui ix, ui iy, int rix, int riy, ui kix, ui kiy, float* acc) override;
        __host__  uc* operator()() override;
        __host__ void operator delete(void *p);
    protected:
        size_t _size() override{
            return sizeof(*this);
        }
    private:
        ui sigma;
        float* kernel;
    };
    class Dilate_Erode: public ConvolutionFunctor<uc,uc,ui, Dilate_Erode>{
    public:
        enum Mode{
            dilate,
            erode
        };
        __host__ Dilate_Erode(Tensor<uc>& in, Mode mode,ui kernelDim=3);
        __device__ Dilate_Erode(Dilate_Erode& other);
        __device__ void operator()(ui ix, ui iy) override;
        __device__ void operator()(ui ix, ui iy, int rix, int riy, ui kix, ui kiy, ui* acc) override;
        __host__  uc* operator()() override;
    protected:
        size_t _size() override{
            return sizeof(*this);
        }
    private:
        Mode mode;
    };
}
#endif //IMG1_KERNEL_CUH