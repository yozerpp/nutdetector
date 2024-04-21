//
// Created by jonossar on 4/17/24.
//

#include "../include/Transferable.cuh"
#include "../include/Kernel.cuh"
template<typename T>
__host__ T* VirtualTransferable<T>:: toGpu(){
auto * dev=Transferable<T>::toGpu();
T** devPtr;
cudaMalloc(&devPtr, sizeof(void*));
deviceInstance<<<1,1>>>(devPtr,dev);
auto** hostPtr=(T**) malloc(sizeof(void*));
cudaMemcpy(hostPtr, devPtr, sizeof(void*), cudaMemcpyDeviceToHost);
cudaFree(dev);
cudaFree(devPtr);
return *hostPtr;
}
template Kernel::GrayScale* VirtualTransferable<Kernel::GrayScale>::toGpu();
template Kernel::Gaussian* VirtualTransferable<Kernel::Gaussian>::toGpu();
template Kernel::Distribution* VirtualTransferable<Kernel::Distribution>::toGpu();
template Kernel::Cluster* VirtualTransferable<Kernel::Cluster>::toGpu();
template Kernel::Dilate_Erode* VirtualTransferable<Kernel::Dilate_Erode>::toGpu();