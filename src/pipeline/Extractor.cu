//
// Created by jonossar on 3/23/24.
//

#include <iostream>
#include "../include/Extractor.cuh"
#include "../include/KernelCommons.cuh"

#define IMG1_FEATUREXTRACTOR_IMPLEMENTATION
#define FILENAME "FeatureExctractor.cu"
namespace Extractor{
    static inline __device__ bool kernelIsntWhite(const unsigned char * addr){
        return *addr!=(unsigned char)255 || *(addr+1)!=(unsigned char)255 || *(addr+2)!=(unsigned char)255;
    }
    __host__ void checkErrors(bool * o, unsigned int i){
        for(unsigned int j=0; j<i; i++){
            if(!o[j]) std::cout << j;
        }
    }
    /** calculate weighted sum, it corresponds to @p Npq in the mathematical Hu moments formula
     *
     * @param p
     * @param q
     * @param out
     * @param label
     * @param data
     * @param x
     * @param y
     */
    __global__ void weightedSum(const unsigned int p, const unsigned int q, double *out, const Common::ObjectPosition * label, unsigned char *data, const unsigned int x, const unsigned int y) {
//        __shared__ double blockSum;
        unsigned int idx_y=blockIdx.y*blockDim.y+ threadIdx.y;
        unsigned int idx_x=blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int realIdx = (label->y_start +idx_y ) * x + label->x_start + idx_x;
//        if(idx_x==0 && idx_y==0) {
//            blockSum=0;
//        }
//        __syncthreads();
        if(idx_x>=label->x_len || idx_y>=label->y_len) return;
//        o[idx_y*_detect->x_len + idx_x]= true;
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
    /** calculate central moments, it corresponds to Upq in the mathematical Hu moments formula
     *
     * @param p
     * @param q
     * @param xNormal
     * @param yNormal
     * @param out
     * @param label
     * @param data
     * @param x
     * @param y
     */
    __global__ void centralMoment(const unsigned int p, const unsigned int q, const double * xNormal, const double * yNormal, double *out, const Common::ObjectPosition * label, unsigned char *data, const unsigned int x, const unsigned int y) {
//        __shared__ double blockSum;
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
    __host__ Moment::~Moment(){
        cudaFree(this->label);
        moments.clear();
    }
    /** allocate gpu memory and reserve configuration arguements for future kernel calls
     *
     * @param label
     * @param image
     */
__host__ Moment::Moment(Common::ObjectPosition *label, Image * image):image(image) {
    double x2=Common::roundP2( label->x_len);
    double y2=Common::roundP2( label->y_len);
    x2=x2>y2?x2:y2;
    y2=y2>x2?y2:x2;
    threads=dim3(KERNEL_THREADS,KERNEL_THREADS,1);
    unsigned int d1 = x2 / threads.x;
    unsigned int d2 = y2 / threads.y;
    blocks= dim3(d1, d2, 1);
    Common::ObjectPosition* gpuLabel;
//    ERC(cudaMemset(xNormal, (double)0, sizeof(double )));
//    ERC(cudaMemset(yNormal, (double)0, sizeof(double )));
//    ERC(cudaMemset(totalPixel, (double)0, sizeof(double )));
    ERC(cudaMalloc(&gpuLabel, sizeof(Common::ObjectPosition)));
    ERC(cudaMemcpy(gpuLabel, label, sizeof(Common::ObjectPosition), cudaMemcpyHostToDevice));
    this->label=gpuLabel;
    xNormal=0;
    yNormal=0;
    totalPixel=0;
    if(streams.find(std::this_thread::get_id())==streams.end()){
        auto * stream=new cudaStream_t;
        cudaStreamCreate(stream);
        streams.insert_or_assign(std::this_thread::get_id(),stream);
    }
    }
    /** calculate normals this function calculates @p N10/N00 and @p M01/M00
     *
     */
    __host__ void Moment::calculateNormals() {
        double * xNormalD;
        double * yNormalD;
        double * totalD;
        ERC(cudaMalloc(&xNormalD, sizeof(double)));
        ERC(cudaMalloc(&yNormalD, sizeof(double)));
        ERC(cudaMalloc(&totalD, sizeof(double)));
        ERC(cudaMemcpy(xNormalD, &yNormal, sizeof(double ), cudaMemcpyHostToDevice));
        ERC(cudaMemcpy(yNormalD, &xNormal, sizeof(double ), cudaMemcpyHostToDevice));
        ERC(cudaMemcpy(totalD, &totalPixel, sizeof(double ), cudaMemcpyHostToDevice));
        weightedSum <<<blocks, threads, sizeof(double),thisStream>>>(1, 0, yNormalD, label, image->data, image->x, image->y);
        cudaStreamSynchronize(thisStream);
        ERC(cudaGetLastError());
//    bool * oH = (bool*)malloc(sizeof(bool)*label->x_len*_detect->y_len);
//    cudaMemcpy(oH, out, sizeof(bool)*label->x_len*_detect->y_len, cudaMemcpyDeviceToHost);
//    checkErrors(oH, label->x_len*_detect->y_len);
        weightedSum <<<blocks, threads, sizeof(double ), thisStream>>>(0, 1, xNormalD, label, image->data, image->x, image->y);
        cudaStreamSynchronize(thisStream);
        ERC(cudaGetLastError());
        weightedSum <<<blocks, threads, sizeof(double), thisStream>>>(0, 0, totalD, label, image->data, image->x, image->y);
        cudaStreamSynchronize(thisStream);
        ERC(cudaGetLastError());
        cudaDeviceSynchronize();
        ERC(cudaMemcpy(&xNormal, xNormalD, sizeof(double ), cudaMemcpyDeviceToHost));
        ERC(cudaMemcpy(&yNormal, yNormalD, sizeof(double), cudaMemcpyDeviceToHost));
        ERC(cudaMemcpy(&totalPixel, totalD, sizeof(double ), cudaMemcpyDeviceToHost));
        xNormal/=totalPixel;
        yNormal/=totalPixel;
        cudaFree(xNormalD);
        cudaFree(yNormalD);
        cudaFree(totalD);
    }
    /** calculate centrals this function calculates the central moments @p Upq/U^(p+q) and saves it in the map @p moments
     *
     */
    void __host__ Moment::calculateCentrals(){
        double up;
        double down;
        double *xNormalD;
        double*  yNormalD;
        double * upD;
        double* downD;
        ERC(cudaMalloc(&upD, sizeof(double )));
        ERC(cudaMalloc(&downD, sizeof(double )));
        ERC(cudaMalloc(&xNormalD, sizeof(double )));
        ERC(cudaMalloc(&yNormalD, sizeof(double )));
        cudaMemcpy(xNormalD, &xNormal, sizeof(double ), cudaMemcpyHostToDevice);
        cudaMemcpy(yNormalD, &yNormal, sizeof(double), cudaMemcpyHostToDevice);
        for (int i = 0; i <= MOMENT_MAX; i++) {
            for (int j = 0; j <= MOMENT_MAX; j++) {
                if (i * j < 3 && (i != j || i * j == 1) && i + j != 1) {
                    up=0;
                    down=0;
                    cudaMemcpy(upD, &up, sizeof(double ), cudaMemcpyHostToDevice);
                    cudaMemcpy(downD, &down, sizeof(double ), cudaMemcpyHostToDevice);
                    centralMoment <<<blocks, threads, sizeof(double ), thisStream>>>(i, j, xNormalD, yNormalD, upD, label,
                                                                       image->data, image->x, image->y);
                    cudaStreamSynchronize(thisStream);
                    ERC(cudaGetLastError());
                    centralMoment <<<blocks, threads, sizeof(double ), thisStream>>>(0, 0, xNormalD, yNormalD, downD, label,
                                                                       image->data, image->x, image->y);
                    cudaStreamSynchronize(thisStream);
                    ERC(cudaGetLastError());
                    ERC(cudaMemcpy(&up, upD, sizeof(double), cudaMemcpyDeviceToHost));
                    ERC(cudaMemcpy(&down, downD, sizeof(double), cudaMemcpyDeviceToHost));
                    double val = up / pow(down, i + j);
                    moments.insert_or_assign(std::to_string(i) + "," + std::to_string(j), val);
                }
            }
        }
        ERC(cudaFree(upD));
        ERC(cudaFree(downD));
    }
    static std::mutex mtx;
    /** wrapper function, see above
     *
     */
    __host__ void Moment::calculate() {
        if(streams.find(std::this_thread::get_id())==streams.end()) {
            auto *s =new cudaStream_t;
            cudaStreamCreate(s);
            streams.insert_or_assign(std::this_thread::get_id(),s);
        }
        std::scoped_lock lck(mtx);
        calculateNormals();
        calculateCentrals();
    }

#pragma region extract
using namespace std;
    /** this function uses @p Moment to extract the features of one object from given image and object position details and stores them in a matrix.
     *
     * @param label
     * @param image
     * @return matrix containing the features
     */
    __host__ Matrix<double> extractOne(Common::ObjectPosition * label, Image *image) {
        Moment moment(label, image);
        Matrix<double> out(FEATURES_LENGTH,1);
        moment.calculate();
        for(int i=0; i<FEATURES_LENGTH; i++){
            double val=calculateFeature(&moment,i);;
            if(isnan(val)) printf("NaN: %f", val);
            out.operator[](i)=val;
        }
        return out;
    }
    /** this function accumulates results of extraction of every individual object in the image to a matrix (see @p extractOne )
     *
     * @param objects
     * @param image
     * @param labelLength
     * @return
     */
    __host__ Matrix<double> extractFeatures(Common::ObjectPosition * objects, Image * image, unsigned int labelLength){
        printf("--Started extracting--\n");
        Matrix<double> out(FEATURES_LENGTH,0,nullptr);
        unsigned char * gpuData;
        unsigned char * tmp=image->data;
        ERC(cudaMalloc(&gpuData, sizeof(unsigned char) * image->x * image->y * image->channels));
        ERC(cudaMemcpy(gpuData, image->data, sizeof(unsigned char) * image->x * image->y * image->channels, cudaMemcpyHostToDevice));
        image->data=gpuData;
        for(unsigned int i=0; i<labelLength; i++) {
            out.merge(extractOne(&objects[i], image));
        }
        ERC(cudaFree(image->data));
        image->data=tmp;
        return out;
    }
    __host__ double calculateFeature(Moment * m, unsigned int i){
        double o;
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
        else throw std::exception();
        return o;
    }

#pragma endregion
}