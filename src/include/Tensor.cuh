//
// Created by jonossar on 4/16/24.
//

#ifndef IMG1_TENSOR_CUH
#define IMG1_TENSOR_CUH

#include "Transferable.cuh"

template <typename T>
class Tensor : public Transferable<Tensor<T>>{
public:
    T * data;
    unsigned int x;
    unsigned int y;
    unsigned int z;
    __host__ Tensor(T* in, unsigned int x, unsigned int y, unsigned int z): data(in), x(x), y(y), z(z){
    }
    __host__ Tensor() : x(0), y(0), z(0) {};
    __host__ Tensor(Tensor& other): data(other.data), x(other.x), y(other.y), z(other.z){}
    inline unsigned int size(){
        return (x>0?x:1)*(y>0?y:1)*(z>0?z:1);
    }
    __host__ ~Tensor(){
        free(data);
    }
    __host__ Tensor(T* data): data(data), x(0), y(0),z(0){}
    __host__ Tensor<T> * toGpu() override{
        T * tmp=this->data;
        cudaMalloc(&(this->data), sizeof(T)*this->size());
        cudaMemcpy(this->data,data, sizeof(T)*x*y*z, cudaMemcpyHostToDevice);
        Tensor<T>* ret=Transferable<Tensor<T>>::toGpu();
        this->data=tmp;
        return ret;
    }
    static __host__ Tensor<T> * toHost(Tensor<T> * t){
        auto* ret= Transferable<Tensor<T>>::toHost(t);
        T* tmp=(T*) malloc(ret->size());
        cudaMemcpy(tmp, ret->data, sizeof(T)*ret->x*ret->y*ret->z, cudaMemcpyDeviceToHost);
        free(ret->data);
        cudaFree(t);
        ret->data=tmp;
        return ret;
    }
protected:
    virtual size_t _size() override{
        return sizeof(*this);
    }
};


#endif //IMG1_TENSOR_CUH
