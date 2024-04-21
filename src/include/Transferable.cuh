//
// Created by jonossar on 4/17/24.
//

#ifndef IMG1_TRANSFERABLE_CUH
#define IMG1_TRANSFERABLE_CUH

template<class T>
class Transferable {
public:
    bool inDevice;
    static __host__ T* toHost(T* dev){
        auto * ret=(T*) malloc(sizeof(T));
        cudaMemcpy(ret, dev, sizeof(T), cudaMemcpyDeviceToHost);
        ret->inDevice=true;
        cudaFree(dev);
        return ret;
    }
    __host__ Transferable():inDevice(false){}
    __host__ virtual T * toGpu() {
        T *ret;
        this->inDevice= true;
        cudaMalloc(&ret, this->_size());
        cudaMemcpy(ret, this, this->_size(), cudaMemcpyHostToDevice);
        this->inDevice=false;
        return ret;
    }
     virtual size_t _size(){
        return sizeof(*this);
    }
};
template <class T>
static __global__ void deviceInstance(T** out, T* other){
    *out=new T(*other);
}
template <class T>
 class VirtualTransferable: public Transferable<T>{
 public:
    __host__ VirtualTransferable(): Transferable<T>(){}
    __host__ virtual T * toGpu() override;
    virtual size_t _size() override{
        return sizeof(*this);
    }
    __device__ VirtualTransferable(VirtualTransferable<T>& other)=default;
};

#endif //IMG1_TRANSFERABLE_CUH
