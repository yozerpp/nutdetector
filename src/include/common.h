//
// Created by jonossar on 3/9/24.
//

#ifndef IMG1_COMMON_H
#define IMG1_COMMON_H
#include <cmath>
#include <random>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread.hpp>
#include <thread>
#include <iostream>
#include <device_launch_parameters.h>
#include <condition_variable>
#include <atomic>
#include "Image.h"
#define STEP_OUT_DIR "./images/steps"
#define prntstps(image, s) {printSteps(image, s);}
#define api(reg, f,...) {<reg>(f, __VA_ARGS__);}
#define FEATURES_LENGTH 7
#define KERNEL_THREADS 32
//#define getLock {std::unique_lock<std::mutex>lck(mtx); if(active.load() > 0) critical.wait(lck, []{return active.load()==0;});}
//#define enter {std::unique_lock<std::mutex>lck(mtx); active++; if()};


static inline bool SHOW_STEPS=true;
namespace Common {
    inline std::random_device rd;
    inline std::mt19937 gen(rd());
    inline std::uniform_real_distribution<float> dist01(0, 1);
    inline std::map<std::thread::id, std::vector<std::thread>> threads{};
    inline std::string STEP_DIR="./images/steps";
    inline std::string TRAINING_DIR="./images/training";
    inline std::string TESTING_DIR="./images/test";
    inline std::string EDGE_DIR="./images/edge";
    inline std::string OUTPUT_DIR="./images/out";
    inline void waitChildren() {
        if(threads.count(std::this_thread::get_id())> 0) {
            for (auto &t: Common::threads.at(std::this_thread::get_id())) {
                if (t.joinable())
                    t.join();
            }
        }
    }
    struct Color {
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;

        inline bool operator==(const Color &other) const {
            return this->r == other.r && this->g == other.g && this->b == other.b;

        }

        inline bool operator!=(const Color &o) const {
            return this->r != o.r || this->g != o.g || this->b != o.b;
        }

        Color() = default;

        inline Color(unsigned char r, unsigned char g, unsigned char b) {
            this->r = r;
            this->g = g;
            this->b = b;
        }
    };
    struct ObjectPosition {
        unsigned int x_start;
        unsigned int x_end;
        unsigned int y_start;
        unsigned int y_end;
        unsigned int x_len;
        unsigned int y_len;
        ObjectPosition()=default;
        explicit inline ObjectPosition(unsigned int x1, unsigned int y1) {
            x_start = x1;
            y_start = y1;
            x_end = x1;
            y_end = y1;
            x_len = 0;
            y_len = 0;
        }
    };
    struct Mean {
        float finalSum = (float)0;
        float weightedSum = (float)0;
        float countSum = (float)0;
        __host__ __device__ bool operator > (const Mean& other) const{
            return finalSum>other.finalSum;
        }
        __host__ __device__ bool operator < (const Mean& other) const{
            return finalSum<other.finalSum;
        }
        __host__ __device__ bool operator == (const Mean& other) const{
            return finalSum==other.finalSum;
        }
#define MAX_TRY 15
#define SHOW_FAILURE false
        static inline Mean * getMeans(unsigned int* distr, unsigned int range= 255, unsigned int numMeans= 2){
            auto *means=new Common::Mean[numMeans];
            bool firstRun=true;
            bool fail;
            unsigned int count=0;
            std::stringstream bf("");
            do{
                if(SHOW_FAILURE) std::cout << bf.str();
                bf.str("");
                fail=false;
                if(!firstRun && SHOW_FAILURE) bf << "failed previous attempt, trying again\n";
                else firstRun=false;
                auto* initials=new float[numMeans];
                bf << "initialized kMean values: ";
                for (int i = 0; i < numMeans; i++) {
                    float val= dist01(Common::gen);
                    initials[i] = val * range;
                    bf << initials[i] <<", ";
                    means[i].finalSum=initials[i];
                }
                bf << std::endl;
                calculateMeans(means, distr, range, numMeans);
                bf<< "converged values: ";
                for (int i = 0; i < numMeans; i++)
                    bf  <<means[i].finalSum << ", ";
                bf << "\n";
                for(int i=0; i<numMeans; i++)
                    for(int j=0; j<numMeans;j++)
                        if(initials[i]==means[j].finalSum) {
                            if(++count > MAX_TRY) {
                                fprintf(stderr, "exceeded the maximum retry amount, exiting...\n");
                                waitChildren();
                                abort();
                            }
                            fail = true;
                        }
                delete[] initials;
            } while(fail);
                std::cout << bf.str();
            std::sort(std::vector<Common::Mean>::iterator(means), std::vector<Common::Mean>::iterator(&means[numMeans]));
            return means;
        }
#undef MAX_TRY
    private:
        static inline float averageKmeans(Mean* means,unsigned int numMeans) {
            float sum = 0.0;
            for (int i = 0; i < numMeans; i++)
                sum += means[i].finalSum;
            return sum / numMeans;
        }

        static inline void calculateMeans(Mean *means, unsigned int* distribution, float range, unsigned int numMeans=2) {
            auto *oldMeans = new float[numMeans];
            do {
                for (int i = 0; i < numMeans; i++)
                    oldMeans[i] = means[i].finalSum;
                findNewKmeans(means, distribution, numMeans, range);
            } while (!converged(means,oldMeans,numMeans));
            delete[] oldMeans;
        }

        static inline void findNewKmeans(Mean *means,const unsigned int* distribution, unsigned int numMeans=2, int range=255) {
            for (int i = 0; i < numMeans; i++) {
                means[i].countSum = 0;
                means[i].weightedSum = 0;
            }
            for (int i = 0; i <= range; i++) {
                Common::Mean *closest = findClosestKmean(means,(float)i,numMeans,(float)range);
                closest->weightedSum += (float)(distribution[i] * i);
                closest->countSum += (float)(distribution[i]);
            }
            for (int i = 0; i < numMeans; i++)
                means[i].finalSum = (means[i].countSum > 0 ? (means[i].weightedSum * 1.0f / means[i].countSum)
                                                           : means[i].finalSum);
        }

        static inline Common::Mean *findClosestKmean(Mean *means,float value,unsigned int numMeans,float range) {
            Common::Mean *closest = nullptr;
            auto closestDist = (float) range + 1;
            for (int i = 0; i < numMeans; i++)
                if (abs(value - means[i].finalSum) < closestDist) {
                    closest = &means[i];
                    closestDist = abs(value - means[i].finalSum);
                }
            return closest;
        }

        static inline bool converged(Mean *means,const float *values,unsigned int numMeans) {
            bool yes = true;
            for (int i = 0; i < numMeans; i++)
                if (abs(means[i].finalSum - values[i]) > 0.01)
                    yes = false;
            return yes;
        }

       static inline bool isHigherMean(Mean *means,Common::Mean *mean,unsigned int numMeans=2) {
            for (int i = 0; i < numMeans; i++)
                if(mean->finalSum < means[i].finalSum)
                    return false;
            return true;
        }
    };

    static inline unsigned int roundP2(unsigned int x) {
        return pow(2, ceil(log2(x)));
    }

    template<typename T>
    static inline T *initializeArray(T value, unsigned long int length) {
        T *arr = new T[length];
        for (int i = 0; i < length; i++)
            arr[i] = value;
        return arr;
    }

    __host__ __device__ static inline double **
    initialize2DArray(double value, const unsigned int x, const unsigned int y) {
        auto **ret = (double **) malloc(sizeof(double *) * y);
        for (unsigned int i = 0; i < y; i++) {
            auto *arr = (double *) malloc(sizeof(double) * x);
            for (unsigned int j = 0; j < x; j++)
                arr[j] = value;
            ret[i] = arr;
        }
        return ret;
    }

    static __forceinline__ bool isWhite(const unsigned char *in) {
        return *in == (unsigned char) 255 && *(in + 1) == (unsigned char) 255 && *(in + 2) == (unsigned char) 255;
    }

    static inline void writeColor(unsigned char *loc, const Color& col) {
        *loc = col.r;
        *(loc + 1) = col.g;
        *(loc + 2) = col.b;
    }

    static inline Color randomColor() {
        Color ret(dist01(gen) * 255, dist01(gen) * 255, dist01(gen) * 255);
        if (ret.r == 255 && ret.g == 255 && ret.b == 255) {
            ret.r = dist01(gen) * 255;
        }
        return ret;
    }

    template<typename T>
    static inline __host__ T *
    gaussianKernel(unsigned int dim, unsigned int sigma, unsigned int dimension = 1, int center = -1) {
        center = center >= 0 ? center : (dim / 2);
        T *out = new T[dim];
        T sum = (T) 0;
        for (int i = 0; i < dim; i++) {
            out[i] = 1 / (sigma * sqrt(2 * M_PI)) * exp(-0.5 * pow((i - center) * 1.0 / sigma, 2));
            sum += out[i];
        }
        for (int i = 0; i < dim; i++) out[i] /= sum;
        if (dimension == 2) {
            T *tmp = new T[dim * dim];
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    tmp[i * dim + j] = out[i] * out[j];
            delete[] out;
            out = tmp;
        }
#define SHOW_KERNEL false
#if SHOW_KERNEL
    for(int i=0; i<dim; i++){
            for(int j=0; j< dim; j++){
                std::cout << out[i * dim + j] << (j != dim?", " : " ");
            }
        std::cout<< std::endl;
        if(dimension==1 ) break;
        }
#endif
#undef SHOW_KERNEL
         return out;
    }
#define SOBEL_MASK_DIM 3
#define SOBEL_MASK_SIZE (SOBEL_MASK_DIM*SOBEL_MASK_DIM)
#define SOBEL_MASK_VARIANTS 2
template <typename T=float>
    static inline __host__ T **sobelKernels(){
        T* x=new T[SOBEL_MASK_SIZE];
        T* y=new T[SOBEL_MASK_SIZE];
        T** ret=(T**) malloc(sizeof(T*)*SOBEL_MASK_VARIANTS);
        for(int i=0; i<SOBEL_MASK_DIM; i++)
            for(int j=0; j<SOBEL_MASK_DIM; j++){
                T v=(i==SOBEL_MASK_DIM/2?2:1)*(j==SOBEL_MASK_DIM/2?0:(j<SOBEL_MASK_DIM/2?-1:1));
                x[i*SOBEL_MASK_DIM+j]=v;
                y[j*SOBEL_MASK_DIM+i]=-v;
            }
        ret[0]=x;ret[1]=y;
//            for (int i = 0; i < 2; i++) {
//                for (int j = 0; j < SOBEL_MASK_DIM; j++) {
//                    for (int k = 0; k < SOBEL_MASK_DIM; k++)
//                        printf("%d, ", ret[i][j * SOBEL_MASK_DIM + k]);
//                    printf("\n");
//                }
//                printf("\n\n");
//            }
        return ret;
    }
    template <typename T=float>
    static inline __host__ T* sobelKernels1d(){
        T* ret=new T[SOBEL_MASK_SIZE*SOBEL_MASK_VARIANTS];
        T **d2=sobelKernels();
        for(int i=0, k=0; i < SOBEL_MASK_VARIANTS; i++, k+=SOBEL_MASK_SIZE) {
            memcpy(&ret[k], d2[i], sizeof(T) * SOBEL_MASK_SIZE);
            delete[]d2[i];
        }
//        for(int i=0; i<SOBEL_MASK_SIZE*SOBEL_MASK_VARIANTS; i++){
//            if(i==SOBEL_MASK_SIZE) printf("\n");
//            printf("%d, ", ret[i]);
//        }
        return ret;
    }

    //    threads.clear();
    //    auto ts=threads.equal_range(pthread_self());
    //    if(ts.first!=ts.second) {
    //        for (auto it = ts.first; it != ts.second; ++it) {
    //            if (it->second.joinable())
    //                it->second.join();
    //        }
    //        threads.erase(pthread_self());
    //    }
    template <typename T>
    void fork(T&& t){
//        try{
//            threads.at(std::this_thread::get_id());
//        } catch (const std::out_of_range& e){
//           threads.insert_or_assign(std::this_thread::get_id(), std::vector<std::thread>());
//        }
        threads[std::this_thread::get_id()].push_back(std::thread([t]()->void{
            t();
            waitChildren();
        }));
    }

//static inline std::vector<std::thread> threads{};
}
#define GRADIENT_0 0
#define GRADIENT_45 1
#define GRADIENT_90 2
#define GRADIENT_135 3

static inline void printSteps(Image *image, const char *step) {
    image->fileName.fileBaseName.append("_").append(std::string(step));
    image->save(STEP_OUT_DIR);
}

#define apiWrpr(f, section, ...) \
{ \
    getLock<section>(); \
    decltype(ret(f)) ret = f(__VA_ARGS__); \
    releaseLock<section>(); \
    return ret; \
}
//namespace detail {
//    template< class T, bool is_function_type = false >
//    struct add_pointer {
//        using type = typename std::remove_reference<T>::type*;
//    };
//
//    template< class T >
//    struct add_pointer<T, true> {
//        using type = T;
//    };
//
//    template< class T, class... Args >
//    struct add_pointer<T(Args...), true> {
//        using type = T(*)(Args...);
//    };
//
//    template< class T, class... Args >
//    struct add_pointer<T(Args..., ...), true> {
//        using type = T(*)(Args..., ...);
//    };
//}
//namespace detail
//{
//    template <std::size_t Ofst, class Tuple, std::size_t... I>
//    constexpr auto slice_impl(Tuple&& t, std::index_sequence<I...>)
//    {
//        return std::forward_as_tuple(
//                std::get<I + Ofst>(std::forward<Tuple>(t))...);
//    }
//}
//
//template <std::size_t I1, std::size_t I2, class Cont>
//constexpr auto tuple_slice(Cont&& t)
//{
//    static_assert(I2 >= I1, "invalid slice");
//    static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2,
//                  "slice index out of bounds");
//
//    return detail::slice_impl<I1>(std::forward<Cont>(t),
//                                  std::make_index_sequence<I2 - I1>{});
//}
//namespace types {
//    struct EOStream{
//        using type= EOStream;
//    };
//    template<typename First, typename... Rest>
//    struct first {
//        using type = First;
//    };
//    template<typename>
//    struct strip;
//
//    template<typename ...T>
//    struct strip<std::tuple<T...>>
//    {
//        using type = std::array<T...>;
//    };
//
//};
//template <typename O,typename... A>
//struct castFunc{
//    O (*f)(A...);
//
//public:
//    explicit castFunc(O (*f)(A...)): f(f){}
//    template <typename... Ar>
//    auto cast() {
//        return reinterpret_cast< O (*)(Ar...)>(f);
//    }
//};
//#define CRITICAL 1
//#define SAFE 0
//void getLock(int section);
//void releaseLock(int section);
//template <int section=SAFE,typename F, typename... Ar>
//static inline auto apiWrapper( F* f,Ar... arg) -> std::invoke_result_t<decltype(castFunc(f).template cast<Ar...>()),Ar...>{
//    auto casted= castFunc (f).template cast<Ar...>();
//    getLock(section);
//    auto ret= casted(std::forward<Ar>(arg)...);
//    releaseLock(section);
//    return ret;
//}
//
//template <int section=SAFE,typename F>
//static inline auto apiWrapper( std::function<F>&& f) -> std::invoke_result_t<F>{
//    getLock(section);
//    auto ret= f();
//    releaseLock(section);
//    return ret;
//}
//
////template <int section=SAFE,typename F, typename... Ar>
//static inline auto ( F&& f, Ar... args){
//    auto casted= castFunc (f).cast(args...);
//    getLock(section);
//    auto ret= casted(args...);
//    releaseLock(section);
//    return ret;
//}
//template <int section=SAFE,typename F, typename... Ar>
//static inline auto ( F&& f,Ar... args){
//    auto casted= castFunc (&f).cast(args...);
//    getLock(section);
//    auto ret= casted(args...);
//    releaseLock(section);
//    return ret;
//}
#endif //IMG1_COMMON_H
