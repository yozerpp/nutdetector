//
// Created by jonossar on 3/15/24.
//
#include <boost/bind/bind.hpp>
#include "../include/Preprocessor.h"
#define KERNEL_DIM 35
#define KERNEL_SIGMA 2

#define IMG1_PREPROCESSOR_IMPLEMENTATION
//
// Created by jonossar on 3/9/24.
//
namespace Preprocessing {
    const unsigned int dilate_kernel=5;
    const unsigned int erode_kernel=3;
    Image * Preprocessor:: grayscale(Image * image){

        auto * out= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char>(image->data, KernelFunctions::grayScale, image->x, image->y, image->channels, image->x*image->y*image->channels);
        printf("turned into grayscale\n");
        delete[] image->data;
        image->data=out;
        image->channels=1;
        return image;
    }
    Image *Preprocessor:: dilate(Image *image, int kernelDim) {
        auto f = [kernelDim](unsigned char *i, unsigned char *o, unsigned int x, unsigned int y) { return KernelFunctions::dilate(kernelDim, i,o,x,y); };
        auto* o=KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char>(image->data,
                                                                                      (void (&&)(unsigned char *,
                                                                                                 unsigned char *,
                                                                                                 unsigned int,
                                                                                                 unsigned int)) std::move(
                                                                                              f), image->x, image->y, image->channels, image->x * image->y * image->channels);
        printf("dilated\n");
        delete[] image->data;
        image->data=o;
        if(RENAME_IMAGE) image->fileName.fileBaseName.append("_dilate");
        return image;
    }
    Image * Preprocessor:: erode(Image * image, int kernelDim){
        auto f = [kernelDim](unsigned char *i, unsigned char *o, unsigned int x, unsigned int y) { return KernelFunctions::erode(kernelDim, i,o,x,y); };
        auto * out= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char>(image->data,
                                                                                          (void (&&)(unsigned char *,
                                                                                                     unsigned char *,
                                                                                                     unsigned int,
                                                                                                     unsigned int)) std::move(
                                                                                                  f), image->x, image->y, image->channels, image->x * image->y * image->channels);
        printf("eroded\n");
        delete[] image->data;
        image->data=out;
        if(RENAME_IMAGE) image->fileName.fileBaseName.append("_erode");
        return image;
    }
    Image * Preprocessor::polarize(Image *image, Distribution distr) {
        printf("--Started Binary Conversion--\n");
        means = new Common::Mean[numMeans];
        if (image->channels != 1)
            image= grayscale(image);
        if (distr == STRAIGHT ){
            distribution=Common::initializeArray((unsigned int) 0, pixelRange);
            for (int i = 0; i < image->x * image->y; i++)
                distribution[image->data[i]]++;
        }
        else {
            distribution= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned int>(image->data, KernelFunctions::distribution, image->x, image->y, image->channels, image->x*image->y*image->channels);
        }
        printf("initialized kMean values: ");
        for (int i = 0; i < numMeans; i++) {
            means[i].finalSum = dist01(gen) * (pixelRange - 1);
            printf("%f, ", means[i].finalSum);
        }
        printf("\n");
        calculateMeans();
        delete[] distribution;
        printf("converged values: ");
        for (int i = 0; i < numMeans; i++)
            printf("%f, ", means[i].finalSum);
        printf("\n");
        if (distr == STRAIGHT)
            for (int i = 0; i < image->x * image->y; i++){
                image->data[i]=isHigherMean(findClosestKmean(image->data[i])) ? (pixelRange - 1) : 0;
            }
        else if (distr == GAUSSIAN) {
            auto  f=[=] __global__ (unsigned char* i, unsigned char* o, unsigned int x, unsigned int y){ return KernelFunctions::gaussian(KERNEL_DIM, KERNEL_SIGMA, i, o, x,y);};
//                auto f=boost::bind(KernelFunctions::gaussian, KERNEL_DIM, KERNEL_SIGMA, boost::arg<2>(), boost::arg<3>(),boost::arg<4>(), boost::arg<5>());
             auto *newData= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char>(image->data,
                                                                                                  reinterpret_cast<void (*)(
                                                                                                          unsigned char *,
                                                                                                          unsigned char *,
                                                                                                          unsigned int,
                                                                                                          unsigned int)>(&f), image->x, image->y, image->channels, image->x * image->y * image->channels);
//            for (int i=0; i<image->x*image->y; i++){
//                newData[i] = isLowerMean(findClosestKmean(newData[i])) ? (unsigned char)(pixelRange-1) :(unsigned char)0 ;
//            }

            auto g=[=](unsigned char* i, unsigned char* o, unsigned int x, unsigned int y){ return KernelFunctions::cluster(reinterpret_cast<float *>(Common::meanData(means, numMeans)), numMeans,i, o, x, y);};
//            auto g=boost::bind(KernelFunctions::gaussian, Common::meanData(means, numMeans), numMeans, boost::arg<2>(), boost::arg<3>(),boost::arg<4>(), boost::arg<5>());

            auto *newNewData= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char>(newData,                                                                                                   reinterpret_cast<void (*)(
                    unsigned char *,
                    unsigned char *,
                    unsigned int,
                    unsigned int)>(&g), image->x, image->y, image->channels, image->x * image->y * image->channels);
            delete[]newData;
            delete[] image->data;
            image->data = newNewData;
        }
        if(RENAME_IMAGE) image->fileName.fileBaseName.append(distr == STRAIGHT ? "_straight" : "_gaussian");
        printf("finished binary conversion\n");
        delete[]means;
        return image;
    }
    Preprocessor:: Preprocessor(int numMeans, int pixelRange) {
        this->pixelRange = pixelRange;
        this->numMeans = numMeans;
    }
}