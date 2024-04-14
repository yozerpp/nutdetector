//
// Created by jonossar on 3/23/24.
//

#include <boost/thread.hpp>
#include <iostream>
#include "../include/Extractor.cuh"
#define IMG1_FEATUREXTRACTOR_IMPLEMENTATION
#define FILENAME "FeatureExctractor.cu"
namespace Extractor{
    static INLINE __device__ bool kernelIsntWhite(const unsigned char * addr){
        return *addr!=(unsigned char)255 || *(addr+1)!=(unsigned char)255 || *(addr+2)!=(unsigned char)255;
    }
    __host__ void checkErrors(bool * o, unsigned int i){
        for(unsigned int j=0; j<i; i++){
            if(!o[j]) std::cout << j;
        }
    }
    __global__ void weightedSum(const unsigned int p,const unsigned int q, double *out, const Common::ObjectLabel * label,unsigned char *data,const unsigned int x, const unsigned int y) {
        __shared__ double blockSum;
        unsigned int idx_y=blockIdx.y*blockDim.y+ threadIdx.y;
        unsigned int idx_x=blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int realIdx = (label->y_start +idx_y ) * x + label->x_start + idx_x;
//        if(idx_x==0 && idx_y==0) {
//            blockSum=0;
//        }
//        __syncthreads();
        if(idx_x>=label->x_len || idx_y>=label->y_len) return;
//        o[idx_y*label->x_len + idx_x]= true;
        if (kernelIsntWhite(&data[realIdx * 3])){
            double val=pow<double>(idx_y, p)*pow<double>(idx_x, q);
//            printf("%f=%d^%d * %d^%d\n", val, idx_y, p, idx_x, q);
            atomicAdd(out, val);
        }
//       __syncthreads();
//        if(idx_x==0 && idx_y==0) {
//            atomicAdd(out, blockSum);
//        }
    }
    __global__ void centralMoment(const unsigned int p,const unsigned int q,const double * xNormal,const double * yNormal, double *out,const Common::ObjectLabel * label,unsigned char *data,const unsigned int x, const unsigned int y) {
        __shared__ double blockSum;
        unsigned int idx_y=blockIdx.y*blockDim.y+ threadIdx.y;
        unsigned int idx_x=blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int realIdx = (label->y_start +idx_y ) * x + label->x_start + idx_x;
//        if(idx_x==0 && idx_y==0) {
//            blockSum=0;
//        }
//        __syncthreads();
        if(idx_x>=label->x_len || idx_y>=label->y_len) return;
        if (kernelIsntWhite(&data[realIdx * 3])) {
            double val = pow<double>((idx_y) - *yNormal, p) *
                        pow<double>((idx_x) - *xNormal, q);
            atomicAdd(out, val);
        }
//        __syncthreads();
//        if(idx_x==0 && idx_y==0) {
//            atomicAdd(out, blockSum);
//        }
    }
__host__ Moment123::Moment123(Common::ObjectLabel *label, Image * image) {
    double x2=Common::roundP2( label->x_len);
    double y2=Common::roundP2( label->y_len);
    x2=x2>y2?x2:y2;
    y2=y2>x2?y2:x2;
    threads=dim3(KERNEL_THREADS,KERNEL_THREADS,1);
    unsigned int d1 = x2 / threads.x;
    unsigned int d2 = y2 / threads.y;
    blocks= dim3(d1, d2, 1);
    Common::ObjectLabel* gpuLabel;
//    gc(cudaMemset(xNormal, (double)0, sizeof(double )));
//    gc(cudaMemset(yNormal, (double)0, sizeof(double )));
//    gc(cudaMemset(totalPixel, (double)0, sizeof(double )));
    gc(cudaMalloc(&gpuLabel, sizeof(Common::ObjectLabel)));
    gc(cudaMemcpy(gpuLabel, label, sizeof(Common::ObjectLabel), cudaMemcpyHostToDevice));
    this->label=gpuLabel;
    this->image=image;
    xNormal=0;
    yNormal=0;
    totalPixel=0;
    }
    __host__ void Moment123::calculateNormals() {
        double * xNormalD;
        double * yNormalD;
        double * totalD;
        gc(cudaMalloc(&xNormalD, sizeof(double)));
        gc(cudaMalloc(&yNormalD, sizeof(double)));
        gc(cudaMalloc(&totalD, sizeof(double)));
        gc(cudaMemcpy(xNormalD, &yNormal, sizeof(double ), cudaMemcpyHostToDevice));
        gc(cudaMemcpy(yNormalD, &xNormal, sizeof(double ), cudaMemcpyHostToDevice));
        gc(cudaMemcpy(totalD, &totalPixel, sizeof(double ), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        weightedSum <<<blocks, threads, sizeof(double)>>>(1, 0, yNormalD, label, image->data, image->x, image->y);
        gc(cudaGetLastError());
//    bool * oH = (bool*)malloc(sizeof(bool)*label->x_len*label->y_len);
//    cudaMemcpy(oH, out, sizeof(bool)*label->x_len*label->y_len, cudaMemcpyDeviceToHost);
//    checkErrors(oH, label->x_len*label->y_len);
        weightedSum <<<blocks, threads, sizeof(double )>>>(0, 1, xNormalD, label, image->data, image->x, image->y);
        gc(cudaGetLastError());
        weightedSum <<<blocks, threads, sizeof(double)>>>(0, 0, totalD, label, image->data, image->x, image->y);
        gc(cudaGetLastError());
        cudaDeviceSynchronize();
        gc(cudaMemcpy(&xNormal, xNormalD, sizeof(double ), cudaMemcpyDeviceToHost));
        gc(cudaMemcpy(&yNormal, yNormalD, sizeof(double), cudaMemcpyDeviceToHost));
        gc(cudaMemcpy(&totalPixel, totalD, sizeof(double ), cudaMemcpyDeviceToHost));
        xNormal/=totalPixel;
        yNormal/=totalPixel;
        cudaFree(xNormalD);
        cudaFree(yNormalD);
        cudaFree(totalD);
    }
    void __host__ Moment123::calculateCentrals(){
        double up;
        double down;
        double *xNormalD;
        double*  yNormalD;
        double * upD;
        double* downD;
        gc(cudaMalloc(&upD, sizeof(double )));
        gc(cudaMalloc(&downD, sizeof(double )));
        gc(cudaMalloc(&xNormalD, sizeof(double )));
        gc(cudaMalloc(&yNormalD, sizeof(double )));
        cudaMemcpy(xNormalD, &xNormal, sizeof(double ), cudaMemcpyHostToDevice);
        cudaMemcpy(yNormalD, &yNormal, sizeof(double), cudaMemcpyHostToDevice);
        for (int i = 0; i <= MOMENT_MAX; i++) {
            for (int j = 0; j <= MOMENT_MAX; j++) {
                if (i * j < 3 && (i != j || i * j == 1) && i + j != 1) {
                    up=0;
                    down=0;
                    cudaMemcpy(upD, &up, sizeof(double ), cudaMemcpyHostToDevice);
                    cudaMemcpy(downD, &down, sizeof(double ), cudaMemcpyHostToDevice);
                    cudaDeviceSynchronize();
                    centralMoment <<<blocks, threads, sizeof(double )>>>(i, j, xNormalD, yNormalD, upD, label,
                                                                       image->data, image->x, image->y);
                    gc(cudaGetLastError());
                    centralMoment <<<blocks, threads, sizeof(double )>>>(0, 0, xNormalD, yNormalD, downD, label,
                                                                       image->data, image->x, image->y);
                    gc(cudaGetLastError());
                    cudaDeviceSynchronize();
                    gc(cudaMemcpy(&up, upD, sizeof(double), cudaMemcpyDeviceToHost));
                    gc(cudaMemcpy(&down, downD, sizeof(double), cudaMemcpyDeviceToHost));
                    double val = up / pow(down, i + j);
                    if(std::isnan(val)) printf("NaN: %f\n", val);
                    moments.insert_or_assign(std::to_string(i) + "," + std::to_string(j), val);
                }
            }
        }
        gc(cudaFree(upD));
        gc(cudaFree(downD));
    }
    __host__ void Moment123::calculate() {
        calculateNormals();
        calculateCentrals();
        gc(cudaFree(label));
    }
    __host__ long double * extractOne(Common::ObjectLabel * label, Image *image) {
        auto *m = new Moment123(label,image);
        auto * out=new long double[FEATURES_LENGTH];
        m->calculate();
        for(int i=0; i<FEATURES_LENGTH; i++){
            long double val=calculateFeature(m,i);;
            if(std::isnan(val)) printf("NaN: %Lf", val);
            out[i]=val;
        }
        m->moments.clear();
        delete m;
        return out;
    }
    __host__ long double ** extractFeatures(Common::ObjectLabel * objects, Image * image, unsigned int labelLength){
        auto ** out= new long double *[labelLength];
        unsigned char * gpuData;
        unsigned char * tmp=image->data;
        gc(cudaMalloc(&gpuData, sizeof(unsigned char)*image->x*image->y*image->channels));
        gc(cudaMemcpy(gpuData, image->data, sizeof(unsigned char)*image->x*image->y*image->channels, cudaMemcpyHostToDevice));
        image->data=gpuData;
        for(unsigned int i=0; i<labelLength; i++) {
            out[i] = extractOne(&objects[i], image);
        }
        gc(cudaFree(image->data));
        image->data=tmp;
        return out;
    }
    __host__ long double calculateFeature(Moment123 * m, unsigned int i){
        long double o;
        if (i == 0)o = m->getMoment(2, 0) + m->getMoment(0, 2);
        else if (i == 1)
            o = pow(m->getMoment(2, 0) - m->getMoment(0, 2), 2.0) +
                4 * pow(m->getMoment(1, 1), 2);
        else if (i == 2)
            o = pow(m->getMoment(3, 0) - 3 * m->getMoment(1, 2), 2) +
                pow(3 * m->getMoment(2, 1) - m->getMoment(0, 3), 2);
        else if (i == 3)
            o = pow(m->getMoment(3, 0) + m->getMoment(1, 2), 2) +
                pow(m->getMoment(2, 1) + m->getMoment(0, 3), 2);
        else if (i == 4)
            o = (m->getMoment(3, 0) - 3 * m->getMoment(1, 2)) * (m->getMoment(3, 0) + m->getMoment(1, 2)) *
                (pow(m->getMoment(3, 0) + m->getMoment(1, 2), 2) -
                 3 * pow(m->getMoment(2, 1) + m->getMoment(0, 3), 2)) +
                (3 * m->getMoment(2, 1) - m->getMoment(0, 3)) * (m->getMoment(2, 1) + m->getMoment(0, 3)) *
                (3 * pow(m->getMoment(3, 0) + m->getMoment(1, 2), 2) -
                 pow(m->getMoment(2, 1) + m->getMoment(0, 3), 2));
        else if (i == 5)
            o = (m->getMoment(2, 0) - m->getMoment(0, 2)) * (pow(m->getMoment(3, 0) + m->getMoment(1, 2), 2) -
                                                             pow(m->getMoment(2, 1) + m->getMoment(0, 3), 2)) +
                4 * m->getMoment(1, 1) * (m->getMoment(3, 0) + m->getMoment(1, 2)) *
                (m->getMoment(2, 1) + m->getMoment(0, 3));
        else if(i==6)
            o = (3 * m->getMoment(2, 1) - m->getMoment(0, 3)) * (m->getMoment(3, 0) + m->getMoment(1, 2)) *
                (pow(m->getMoment(3, 0) + m->getMoment(1, 2), 2) -
                 3 * pow(m->getMoment(2, 1) + m->getMoment(0, 3), 2)) -
                (m->getMoment(3, 0) - 3 * m->getMoment(1, 2)) * (m->getMoment(2, 1) + m->getMoment(0, 3)) *
                (3*pow(m->getMoment(3, 0) + m->getMoment(1, 2), 2) - pow(m->getMoment(2, 1) + m->getMoment(0, 3), 2));
        return o;
    }
}