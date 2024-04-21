//
// Created by jonossar on 3/15/24.
//
#include "../include/Preprocessor.h"


#define IMG1_PREPROCESSOR_IMPLEMENTATION
//
// Created by jonossar on 3/9/24.
//
namespace Preprocessing {
    static const unsigned int dilate_kernel=5;
    static const unsigned int erode_kernel=3;
    static const unsigned int gaussian_kernel_len=5;
    static const unsigned int gaussian_kernel_sigma=2;
    Image* gaussianFilter(Image* image){
        auto * kernel = Common::gaussianKernel<float>(gaussian_kernel_len,gaussian_kernel_sigma);
        auto * tmp= Kernel::Executor<unsigned char,unsigned char,Kernel::gaussian, float, float>(image->data, kernel,Kernel::gaussian(),image->x, image->y, image->channels, image->size(),gaussian_kernel_len);
        delete[] kernel;
        delete[]image->data;image->data=tmp;
        if(SHOW_STEPS) ps(image,"ga");
        return image;
    }
    Image * Preprocessor:: grayscale(Image * image){

        auto * out= Kernel::Executor<unsigned char,unsigned char>(image->data, (void*)nullptr,Kernel::grayScale(),image->x, image->y, image->channels, image->size(),0);
        printf("turned into grayscale\n");
        delete[] image->data;
        image->data=out;
        image->channels=1;
        if(SHOW_STEPS) ps(image, "g");
        return image;
    }
    Image *Preprocessor:: dilate(Image *image, int kernelDim) {
        auto * out= Kernel::Executor<unsigned char,unsigned char, Kernel::dilate, void, unsigned int>(image->data, (void*)nullptr,Kernel::dilate(),image->x, image->y, image->channels, image->size(),0);
        printf("dilated\n");
        delete[] image->data;
        image->data=out;
        if(SHOW_STEPS) ps(image, "d");
        return image;
    }
    Image * Preprocessor:: erode(Image * image, int kernelDim){
        auto * out= Kernel::Executor<unsigned char,unsigned char, Kernel::erode, void, unsigned int>(image->data, (void*)nullptr,Kernel::erode(),image->x, image->y, image->channels, image->size(),0);
        printf("eroded\n");
        delete[] image->data;
        image->data=out;
        if(SHOW_STEPS) ps(image, "e");
        return image;
    }
    Image * Preprocessor::polarize(Image *image, Distribution distr) {
        printf("--Started Binary Conversion--\n");
        means = new Common::Mean[numMeans];
        if (image->channels != 1)
            image= grayscale(image);
        if (distr == STRAIGHT ){
            distribution=Kernel::Executor<unsigned char,unsigned int>(image->data, (void*)nullptr,Kernel::distribution(),image->x, image->y, image->channels, 256,0);
        }
        else {
            image= gaussianFilter(image);
            distribution= Kernel::Executor<unsigned char,unsigned int>(image->data, (void*)nullptr,Kernel::distribution(),image->x, image->y, image->channels, 256,0);
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
        unsigned char*out;
        if (distr == STRAIGHT){
             out=Kernel::Executor<unsigned char,unsigned char>(image->data, (void*)nullptr,Kernel::cluster(means, numMeans),image->x, image->y, image->channels, image->size(),0);
            if(SHOW_STEPS) ps(image, "b(s)");
        }
        else {
            image = gaussianFilter(image);
            out = Kernel::Executor<unsigned char, unsigned char>(image->data, (void *) nullptr,
                                                                 Kernel::cluster(means, numMeans), image->x, image->y,
                                                                 image->channels, image->size(), 0);
            if(SHOW_STEPS) ps(image, "b(g)");
        }
        image->data = out;
        printf("finished binary conversion\n");
        delete[] distribution;
        delete[]means;
        return image;
    }
    Preprocessor:: Preprocessor(int numMeans, int pixelRange) {
        this->pixelRange = pixelRange;
        this->numMeans = numMeans;
    }

}