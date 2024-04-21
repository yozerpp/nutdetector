//
// Created by jonossar on 3/15/24.
//
#include <boost/bind/bind.hpp>
#include "../include/Preprocessor.h"
#include "Kernel.cuh"
#include "Model.h"

#define KERNEL_DIM 35
#define KERNEL_SIGMA 2

#define IMG1_PREPROCESSOR_IMPLEMENTATION
//
// Created by jonossar on 3/9/24.
//
using namespace Kernel;
namespace Preprocessing {
    const unsigned int dilate_kernel=5;
    const unsigned int erode_kernel=3;
    Image & Preprocessor:: grayscale(Image& image){
        GrayScale f(image);
        auto * ret= (f());
        free(image.data);
        image.data=ret;
        image.z=1;
        if(SHOW_STEPS){
            image.fileName.fileBaseName.append("_g");
            image.save(OUTPUT_DIR);
        }
        printf("turned into grayscale\n");

        return image;
    }
    Image & Preprocessor:: dilate(Image &image, int kernelDim) {
        Dilate_Erode f(image,Kernel::Dilate_Erode::dilate, kernelDim);
        auto* ret= f();
        free(image.data);
        image.data=ret;
        if(SHOW_STEPS) { 
            image.fileName.fileBaseName.append("_d");
            image.save(OUTPUT_DIR);
        }
        printf("dilated\n");
        return image;
    }
    Image & Preprocessor:: erode(Image& image, int kernelDim){
        Dilate_Erode f(image, Dilate_Erode::Mode::erode, kernelDim);
        auto * ret= f();
        free(image.data);
        image.data=ret;
        if(SHOW_STEPS){
            image.fileName.fileBaseName.append("_e");
            image.save(OUTPUT_DIR);
        }
        printf("eroded\n");
        return image;
    }
    Image & Preprocessor::polarize(Image& image, Distribution distr) {
        printf("--Started Binary Conversion--\n");
        means = new Common::Mean[numMeans];
        if (image.z != 1)
            grayscale(image);
        if (distr == STRAIGHT ){
            Kernel::Distribution dist(image);
            distribution=dist();
        }
        else {
            Gaussian g(image,1); //TODO DYNAMICALLY DETERMINE SIGMA AND KERNEL SIZE
            uc* t=g();
            free(image.data);
            image.data=t;
            Kernel::Distribution dist(image);
            distribution=dist();
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
        if (distr == STRAIGHT){
            Cluster f(image,Common::meanData(means,numMeans),numMeans);
            uc* t=f();
            free(image.data);
            image.data=t;
        }
        else if (distr == GAUSSIAN) {
            Gaussian f(image,1);
            uc* t=f();
            free(image.data);
            image.data=t;
        }
        if(SHOW_STEPS) {
            image.fileName.fileBaseName.append("_bin").append(distr == STRAIGHT ? "(straight)" : "(gaussian)");
        }
        printf("finished binary conversion\n");
        delete[]means;
        return image;
    }
    Preprocessor:: Preprocessor(int numMeans, int pixelRange) {
        this->pixelRange = pixelRange;
        this->numMeans = numMeans;
    }
}