//
// Created by jonossar on 3/15/24.
//
#include "../include/Preprocessor.h"


#define IMG1_PREPROCESSOR_IMPLEMENTATION
//
// Created by jonossar on 3/9/24.
//
namespace Preprocessing {
    const unsigned int dilate_kernel=5;
    const unsigned int erode_kernel=3;
    Image * Preprocessor:: grayscale(Image * image){

        auto * out= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char, double, KernelFunctions::grayScale>(
                image->data, nullptr,
                KernelFunctions::grayScale(), image->x,
                image->y, image->channels,
                image->x * image->y, 0);
        printf("turned into grayscale\n");
        delete[] image->data;
        image->data=out;
        image->channels=1;
        return image;
    }
    Image *Preprocessor:: dilate(Image *image, int kernelDim) {

        auto * out= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char, double, KernelFunctions::dilate>(
                image->data, nullptr,
                KernelFunctions::dilate(), image->x,
                image->y, image->channels,
                image->x * image->y, kernelDim);
        printf("dilated\n");
        delete[] image->data;
        image->data=out;
        if(RENAME_IMAGE) image->fileName.fileBaseName.append("_dilate");
        return image;
    }
    Image * Preprocessor:: erode(Image * image, int kernelDim){
        auto * out= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char, double, KernelFunctions::erode>(
                image->data, nullptr,
                KernelFunctions::erode(), image->x,
                image->y, image->channels,
                image->x * image->y, kernelDim);
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
            this->kernel = Common::createGaussianKernel(15, 3);
//                cpuDistr(image);
            distribution= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned int, double, KernelFunctions::distribution>(
                    image->data,
                    kernel->data,
                    KernelFunctions::distribution(),
                    image->x,
                    image->y, 1,
                    pixelRange,
                    kernel->dimension);
            delete[] this->kernel->data;
            delete this->kernel;
        }
        printf("initialized kMean values: ");
        for (int i = 0; i < numMeans; i++) {
            means[i].finalSum = dist01(gen) * (pixelRange - 1);
            printf("%f, ", means[i].finalSum);
        }
        printf("\n");
        calculateMeans();
        printf("converged values: ");
        for (int i = 0; i < numMeans; i++)
            printf("%f, ", means[i].finalSum);
        printf("\n");
        if (distr == STRAIGHT)
            for (int i = 0; i < image->x * image->y; i++){
                image->data[i]=isHigherMean(findClosestKmean(image->data[i])) ? (pixelRange - 1) : 0;
            }
        else if (distr == GAUSSIAN) {
            this->kernel = Common::createGaussianKernel(11, 2);
             auto *newData= KernelFunctions::kernelFunctionsWrapper<unsigned char, unsigned char, double, KernelFunctions::gaussianMean>(
                     image->data,
                     kernel->data,
                     KernelFunctions::gaussianMean(means, numMeans),
                     image->x,
                     image->y, 1,
                     image->x *
                     image->y,
                     kernel->dimension);
//            for (int i=0; i<image->x*image->y; i++){
//                newData[i] = isHigherMean(findClosestKmean(newData[i])) ? (unsigned char)(pixelRange-1) :(unsigned char)0 ;
//            }
            delete[] image->data;
            delete[]  this->kernel->data;
            delete[] distribution;
            delete this->kernel;
            image->data = newData;
        }
        if(RENAME_IMAGE) image->fileName.fileBaseName.append(distr == STRAIGHT ? "_straight" : "_gaussian");
//        image= erode(image,erode_kernel);
//        image= dilate(image,dilate_kernel);
        printf("finished binary conversion\n");
        delete[]means;
        return image;
    }
    Preprocessor:: Preprocessor(int numMeans, int pixelRange) {
        this->pixelRange = pixelRange;
        this->numMeans = numMeans;
    }

}