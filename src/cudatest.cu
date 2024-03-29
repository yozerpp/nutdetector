//
// Created by jonossar on 3/8/24.
//
#include "stdio.h"
#include "include/Image.h"
#include "include/Preprocessor.h"
void __global__ kernelFunc(unsigned int x,unsigned int y,unsigned int x2, bool * out){
    unsigned int flatIdx=(threadIdx.y + blockDim.y*blockIdx.y)*x + (threadIdx.x + blockIdx.x*blockDim.x)*1;
    if(!(flatIdx>=x*y))
        out[flatIdx]= true;
}
unsigned int roundP2(unsigned int x){
//    unsigned int log=log2(x);
//    log= round(log);
//    if(log%2!=0)
//        log++;
    return (unsigned int) pow(2, ceil(log2(x)));
}
int __host__  main(){
auto * image=new Image("/home/jonossar/proj/img1/training/badem.jpg");
Preprocessing::Preprocessor * preprocessor=new Preprocessing::Preprocessor(2,256);
preprocessor->grayscale(image)->save("./out");
//    unsigned int x=851;
//    unsigned int threadDim=32;
//    unsigned int y=376;
//    unsigned int y2= roundP2(y);
//    unsigned int x2= roundP2(x);
//    y2=y2>x2?y2:x2;
//    x2=x2>y2?x2:y2;
//    printf("%d, %d", x2,y2);
//    bool * o;
//    freopen("../log", "w", stdout);
//    cudaMalloc(&o, sizeof(bool)*x*y);
//    cudaMemset(o, 0, sizeof(bool)*x*y);
//    cudaDeviceSynchronize();
//    kernelFunc<<<dim3(y2/threadDim, x2/threadDim, 1), dim3(threadDim, threadDim, 1)>>>(x,y,x2, o);
//    cudaDeviceSynchronize();
//    bool * o1=(bool *)malloc(sizeof(bool)*x*y);
//    cudaMemcpy(o1,o,sizeof(bool)*x*y, cudaMemcpyDeviceToHost);
//    printf("%s" , cudaGetErrorString(cudaGetLastError()));
//    for(int i=0; i<x*y; i++){
//        if (!o1[i]) {
//            printf("%d\n", i);
//        }
//    }
//    train();
//    test();
//    auto * preProcessor=new Preprocessing::Preprocessor(2, 256);
//    for (const auto &file : std::filesystem::directory_iterator(INPUT_DIR)) {
//        auto * img=new Image(file.path().c_str());
//        img=preProcessor->polarize(img, Preprocessing::GAUSSIAN);
//        img->save(OUTPUT_DIR);
//        auto * labelLength=new unsigned int;
//        ObjectLabel* labels=Detector::Detect(img, labelLength);
//        for(int i=0; i<*labelLength; i++)
//            Extractor::extract(labels[i], img);
//        img->save(OUTPUT_DIR);
//    }
}