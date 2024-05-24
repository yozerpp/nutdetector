//
// Created by jonossar on 3/15/24.
//
#include <iostream>
#include "../include/Preprocessor.h"
#include "cuda_runtime_api.h"
#include "common.h"

#define IMG1_PREPROCESSOR_IMPLEMENTATION
#define DIL_KERNEL_CONST 8
#define GAUSSIAN_KERNEL_CONST 6
namespace Preprocessing {
    /** perform gaussian filtering on image
     * this function first creates a gaussian kernel with acquired parameters @p sigma and @p kernelDim (see @p Common::gaussianKernel ) then it applies the kernel to image (see @p Kernel::gaussian for implementation details)
     * @param image
     * @param sigma sigma of the kernel
     * @param kernelDim
     * @return
     */
    Image* gaussian(Image* image, unsigned int sigma, unsigned int kernelDim){
        if(sigma==0 || kernelDim==0){
            kernelDim= pow(image->size(),1.0/GAUSSIAN_KERNEL_CONST);
            kernelDim= kernelDim%2==0?kernelDim-1:kernelDim;
            sigma=kernelDim*3;
        }
        auto* kernel=Common::gaussianKernel<float>(kernelDim,sigma,2);
        auto * out= Kernel::executor<unsigned char, unsigned char, float, Kernel::gaussian<unsigned char>>(image->data, kernel,
                                                                                             Kernel::gaussian<unsigned char>(),
                                                                                             image->x, image->y,
                                                                                             image->channels,
                                                                                             image->x * image->y,
                                                                                             kernelDim);
        delete[]image->data;
        delete[] kernel;
        image->data=out;
        return image;
    }
    /** turns image to grayscale (see @p Kernel::grayScale ) for implementation detail
     *
     * @param image
     * @return
     */
    Image * grayscale(Image * image){
        auto * out= Kernel::executor<unsigned char, unsigned char, double, Kernel::grayScale>(
                image->data, nullptr,
                Kernel::grayScale(), image->x,
                image->y, image->channels,
                image->x * image->y, 0);
        printf("turned into grayscale\n");
        delete[] image->data;
        image->data=out;
        image->channels=1;
        if(SHOW_STEPS) prntstps(image, "grayscale")
        return image;
    }
    /** extract K-Means from an image
     * this function first acquires histrogram of the image (see @p Kernel::distribution for details about algorithm implementation)
     * then it acquires means from the acquired histogram (see @p Common::Mean::getMeans)
     * @tparam T
     * @param image
     * @param in
     * @param numMeans
     * @param maxVal range of the histogram
     * @return
     */
    template <typename T>
    Common::Mean * getMeans(Image* image, T* in, unsigned int numMeans, T maxVal){
        auto* distribution=Kernel::executor<T,unsigned int,double,Kernel::distribution<T>>(
                in, nullptr, Kernel::distribution<T>(0, maxVal), image->x, image->y, image->channels, image->size(), 0
        );
        auto * means=Common::Mean::getMeans(distribution, (double)maxVal,numMeans);
        delete[]distribution;
        return means;
    }
    /** Wrapper function for @p Kernel::cluster
     *
     * @tparam T
     * @param image
     * @param filter
     * @param means
     * @param numMeans
     * @return
     */
    template <typename T=unsigned char>
    unsigned char* cluster(Image* image, T* filter,Common::Mean* means, unsigned int numMeans){
       return Kernel::executor<T,unsigned char,double,Kernel::cluster<T,unsigned char>>(
                filter, nullptr,std::forward<Kernel::cluster<T,unsigned char>>(Kernel::cluster<T,unsigned char>(means,numMeans)),image->x,image->y,image->channels,image->size(),0
        );
    }
    /**
     * this function takes a filter array which maps one to one with pixels in the image(i.e filter[6] corresponds to image->data[6]). it first acquires kMeans from the input then decides whether a pixel will be suppressed or not by in which cluster the corresponding filter pixel is.
     * @tparam T type of filter array
     * @param image input
     * @param filter filter array to decide whether pixel will be black or white
     * @param numMeans number of Kmeans to divide the filter array into
     * @param suppressVal value to write if a pixel will be suppressed
     * @return binary data
     */
    template <typename T=unsigned char>
    unsigned char* _binary(Image* image, T* filter, unsigned int numMeans, T suppressVal= (T)255){
        auto * means= getMeans<T>(image,filter,numMeans,suppressVal);
        auto * out= Preprocessing::cluster<T>(image,filter, means,numMeans);
        delete[]means;
        return out;
    }
    static inline unsigned int dilErKernelDim(unsigned int a){
        return pow(a, 1.0/DIL_KERNEL_CONST);
    }
    /** perform erosion. Kernel dimension is evaluated from the size of the image(see @p dilErKernelDim ) you can see algorithm implementation in @p Kernel::dilate
 *
 * @param image
 * @param kernelDim
 * @return
 */
    Image *dilate(Image *image, int kernelDim) {
        if(kernelDim==0) kernelDim= dilErKernelDim(image->size());
        auto * out= Kernel::executor<unsigned char, unsigned char, double, Kernel::dilate>(
                image->data, nullptr,
                Kernel::dilate(), image->x,
                image->y, image->channels,
                image->x * image->y, kernelDim);
        delete[] image->data;
        image->data=out;
        printf("dilated\n");
        if(SHOW_STEPS) prntstps(image, "dilate")
        return image;
    }
    /** perform erosion. Kernel dimension is evaluated from the size of the image(see @p dilErKernelDim ) you can see algorithm implementation in @p Kernel::erode
     *
     * @param image
     * @param kernelDim
     * @return
     */
    Image * erode(Image * image, int kernelDim){
        if(kernelDim==0) kernelDim= dilErKernelDim(image->size());
        auto * out= Kernel::executor<unsigned char, unsigned char, double, Kernel::erode>(
                image->data, nullptr,
                Kernel::erode(), image->x,
                image->y, image->channels,
                image->x * image->y, kernelDim);
        delete[] image->data;
        image->data=out;
        printf("eroded\n");
        if(SHOW_STEPS) prntstps(image, "erode")
        return image;
    }
    /** function to save intermediate steps in edge detection pipeline. It first normalizes gradients [0-1020) to [0-255) and converts them to unsigned char.
     *
     * @param in
     * @param image
     * @param step
     */
    static inline void convertAndSave(Kernel::edge::gradient* in, Image* image, const char* step){
        auto* image1= new Image(*image);
        auto* cpy=(Kernel::edge::gradient*) malloc(sizeof(Kernel::edge::gradient)*image->size());
        memcpy(cpy, in,sizeof(Kernel::edge::gradient)*image->size());
        image->fileName.fileBaseName.append("_").append(step);
        Common::fork([image1=image1,cpy=cpy, step=step](){
            auto * norm=Kernel::normalize<Kernel::edge::gradient>(cpy,image1->size());
            auto * data=new unsigned char [image1->size()];
            for(int i=0; i<image1->size(); i++)
                data[i]=((unsigned char)norm[i].value);
            image1->data=data;
            prntstps(image1, step);
            delete[] image1->data;
            delete image1;
            free(cpy);
            free(norm);
        });
    }
    /** extract gradients from image (see @p Kernel::edge::gradient for details about algorithm implementation)
     *
     * @param image
     * @return extracted gradients
     */
    Kernel::edge::gradient * getGradients(Image* image){
        auto* ret= Kernel::executor<unsigned char, Kernel::edge::gradient, double, Kernel::edge::getGradient>(
                image->data, nullptr, Kernel::edge::getGradient(Common::sobelKernels()),
                image->x, image->y, image->channels, image->size(), SOBEL_MASK_DIM
        );
#if SHOW_STEPS
        convertAndSave(ret, image, "gradient");
#endif
        printf("extracted gradients\n");
        return ret;
    }
    /** wrapper for non-max suppression (see @p Kernel::edge::nonMaxSuppression for how the algorithm is implemented)
     *
     * @param in
     * @param image
     * @return filtered gradients
     */
    static inline Kernel::edge::gradient* nonMaxSuppression(Kernel::edge::gradient* in, Image* image){
        using namespace Kernel::edge;
       auto *ret= Kernel::executor<gradient, gradient, double, nonMaxSuppress>(
               in, nullptr, nonMaxSuppress(image->x, image->y), image->x, image->y, image->channels, image->size(), 0
       );
#if SHOW_STEPS
        convertAndSave(in, image, "suppress");
#endif
        printf("suppressed\n");
        return ret;
    }
    /** wrapper function for kernel call to @p hysteresis (see @p Kernel::edge::hysteresis )
     *
     * @param filter gradients to be applied as filter
     * @param means K-Means acquired from previous steps
     * @param image -just uses it for x, y coordinates- no mutations to rgb data
     * @return
     */
    static Kernel::edge::gradient* hysteresis(Kernel::edge::gradient* filter, Common::Mean* means,Image*image) {
        unsigned int* count;
        cudaMallocManaged((void**)&count, sizeof(unsigned int));
#define MAX_PASS 10
        unsigned int c=0;
        do {
            *count=0;
            filter = Kernel::executor<Kernel::edge::gradient, Kernel::edge::gradient, double, Kernel::edge::hysteresis>(
                    filter, nullptr, Kernel::edge::hysteresis(means, 2, count), image->x, image->y, image->channels,
                    image->size(), 0
            );
        } while (*count>0 && c++<MAX_PASS);
        cudaFree(count);
        printf("applied hysteresis\n");
        convertAndSave(filter,image,"hysteresis");
        return filter;
#undef  MAX_PASS

    }
#define GRADIENT_MAX_VAL ((gradient)1019)
/** edge detection
 * This function first applies @p gaussian filter to image, then extracts gradients with sobel mask convolution, then it applies @p hysteresis by grouping gradient values 2 groups
 * finally, it uses result of the hysteresis as a filter to convert input image to a binary image that is white in edges (see doc for @p _binary (not @p binary) for more info).
 * @param image
 * @return a binary image that is white on edges
 */
    Image* edge(Image* image){
        using namespace Kernel::edge;
        if(image->channels!=1) image= grayscale(image);
        image= gaussian( image); // Gaussian filter
        printf("applied Gaussian filter\n");
        if(SHOW_STEPS) prntstps(image, "gaussian")
        auto * gradients= getGradients(image); //Extract gradients with sobel mask convolution
        auto * suppressed= nonMaxSuppression(gradients,image); // Perform non-mask suppression to iamge
        delete[]gradients;
        Common::Mean* means= getMeans(image,suppressed,2,GRADIENT_MAX_VAL); // K-Means cluster the gradients
        auto* hysteresis=  Preprocessing::hysteresis(suppressed,means,image); // perform hysteresis
        delete[]means;
        delete[] suppressed;
        auto * out= _binary<gradient>(image, hysteresis,2,GRADIENT_MAX_VAL);
        delete[]image->data;
        image->data=out;
        prntstps(image,"binary");
        if(SHOW_STEPS) prntstps(image, "edge")
        printf("highlighted edges\n");
        return image;
    }
    /** Convert a RGB image to binary
     * this function first applies gaussian, then converts to binary (see @p _binary), then applies dilation and erosion.
     * @param image
     * @return
     */
        Image * binary(Image *image) {
        printf("--Started Binary Conversion--\n");
        if(image->channels!=1) grayscale(image);
        image= gaussian(image); // gaussian
        if(SHOW_STEPS) prntstps(image, "gaussian");
        auto * o=_binary<unsigned char>(image,image->data,2,(unsigned char)255);
        delete[] image->data;
        image->data=o;
        if (SHOW_STEPS) prntstps(image, "binary")
        image= dilate(image);
        image= erode(image);
        printf("finished binary conversion\n");
        return image;
    }
    template unsigned char* Preprocessing::_binary<Kernel::edge::gradient>(Image*, Kernel::edge::gradient*, unsigned int,Kernel::edge::gradient);
    template unsigned char* Preprocessing::_binary<unsigned char>(Image*, unsigned char*, unsigned int, unsigned char);
}
