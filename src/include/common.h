//
// Created by jonossar on 3/9/24.
//

#ifndef IMG1_COMMON_H
#define IMG1_COMMON_H
#include <cmath>
#include <random>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include "Image.h"
#include "cuda_runtime_api.h"
#define OUTPUT_DIR "./out/"
#define SHOW_STEPS true
#define ps(image, s) {printSteps(image, s);}
#define COLLUSION {(unsigned char)0,(unsigned char)0,(unsigned char)0}
#define WHITE {(unsigned char)255,(unsigned char)255,(unsigned char)255}
#define FEATURES_LENGTH 7
#define  INLINE inline
#define KERNEL_THREADS 32
#define gc(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define thisStream *(streams.at(pthread_self()))
inline std::random_device rd;
inline std::mt19937 gen(rd());
inline std::uniform_real_distribution<double> dist01(0,1);
inline std::map<pthread_t, cudaStream_t*> streams{};
static inline unsigned int numThreads=0;
inline boost::mutex mtx;
inline boost::condition_variable cond;
namespace Common {
    struct KernelStruct {
        int dimension=0;
        double *data= nullptr;
    };

    struct Color{
        unsigned char r=0;
        unsigned char g=0;
        unsigned char b=0;
         INLINE bool operator==(const Color& other) const{
            return this->r==other.r && this->g==other.g && this->b==other.b;
        }
        INLINE bool operator!=(const Color& o) const{
            return this->r!=o.r || this->g!=o.g || this->b!=o.b;
        }
        Color(){};
        INLINE Color(unsigned char r, unsigned char g, unsigned char b){
            this->r=r;
            this->g=g;
            this->b=b;
        }
    };

    struct ObjectLabel{
        unsigned int x_start;
        unsigned int x_end;
        unsigned int y_start;
        unsigned int y_end;
        unsigned int x_len;
        unsigned int y_len;
        INLINE ObjectLabel(unsigned int x1, unsigned int y1){
            x_start=x1;y_start=y1;
            x_end=x1;y_end=y1;
            x_len=0;y_len=0;
        }
    };

    struct Mean{
        double finalSum=0.0;
        double weightedSum=0.0;
        double countSum=0.0;
    //        Mean (): finalSum(0), weightedSum(0), countSum(0) {
    //            finalSum=0.0;weightedSum=0.0;countSum=0.0;
    //        }
    };
    static inline unsigned int roundP2(unsigned int x){
        return pow(2, ceil(log2(x)));
    }
    template<typename T>
    static INLINE T *initializeArray(T value, int length) {
        T *arr = new T[length];
        for(int i=0; i<length; i++)
            arr[i]=value;
        return arr;
    }
    __host__ __device__ static INLINE double ** initialize2DArray(double value, const unsigned int x,const unsigned int y){
        auto ** ret=(double **) malloc(sizeof(double *)*y);
        for (unsigned int i=0; i<y; i++){
            auto * arr=(double *) malloc(sizeof(double )* x);
            for(unsigned int j=0; j<x; j++)
                arr[j]=value;
            ret[i]=arr;
        }
        return ret;
    }
    static INLINE bool isNotWhite(unsigned char * in){
        return *in!=(unsigned char)255|| *(in+1)!=(unsigned char)255 || *(in+2)!=(unsigned char)255;
    }
    static INLINE void writeColor(unsigned char * loc, Color *col){
        *loc=col->r;
        *(loc+1)=col->g;
        *(loc+2)=col->b;
    }
    static INLINE Color * randomColor() {
        Color * c = new Color(dist01(gen) * 255,dist01(gen) * 255,dist01(gen) * 255);
        if (c->r == 255 && c->g == 255 && c->b == 255){
            c->r=dist01(gen)*255;
        }
        return c;
    }
    template<typename T>
    static INLINE __host__ T* gaussianKernel(unsigned int len, unsigned int sigma, unsigned int dimension=2, int center=-1){
        center=center>=0?center:(len / 2);
        T* out=new T[len];
        T sum=(T)0;
        for(int i=0; i < len; i++){
            out[i]=1/(sigma * sqrt(2*M_PI)) * exp(-0.5 *pow((i-center)*1.0 / sigma,2));
            sum+=out[i];
        }
        for(int i=0; i < len; i++) out[i]/=sum;
        if(dimension==2){
            T* tmp = new T[len * len];
            for(int i=0; i < len; i++)
                for(int j=0; j < len; j++)
                    tmp[i * len + j]= out[i] * out[j];
            delete[] out;
            out=tmp;
        }
        return out;
    }
}
inline void __host__ __device__ gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
#ifndef  __NVCC__
        fprintf(stderr,"%s::%s::%d\n", cudaGetErrorString(code), file, line);
#else
        printf("%s::%s::%d\n", cudaGetErrorString(code), file, line);
#endif
#ifndef __NVCC__
        if (abort) exit(code);
#endif
    }
}
inline void printSteps(Image* image,std::string &&s){
    image->fileName.fileBaseName.append("_" + s);
    image->save(OUTPUT_DIR);
}
#endif //IMG1_COMMON_H
