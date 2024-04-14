//
// Created by jonossar on 3/9/24.
//

#ifndef IMG1_PREPROCESSOR_H
#define IMG1_PREPROCESSOR_H

#include <cstring>
#include <cfloat>
#ifndef IMG1_COMMON_H
#include "common.h"
#endif
#ifndef IMG1_KERNELFUNCTIONS_CUH
#include "Kernel.cuh"
#endif
#ifndef IMG1_IMAGE_H
#include "Image.h"
#endif


namespace Preprocessing {
    enum Distribution {
        STRAIGHT,
        GAUSSIAN
    };
    Common::KernelStruct *createGaussianKernel(int dimension, double mean, double sigma);

    class Preprocessor{
    private:
        int numMeans;
        int pixelRange;
        unsigned int *distribution;
        Common::Mean *means;
        Common::KernelStruct *kernel;
    public:
        Preprocessor(int numMeans, int pixelRange);

        Image * grayscale(Image * image);

        Image *dilate(Image *image, int kernelDim);

        Image * erode(Image * image, int kernelDim);

        Image *polarize(Image *image, Distribution distr) ;
    private:
        double averageKmeans() {
            double sum = 0.0;
            for (int i = 0; i < numMeans; i++)
                sum += means[i].finalSum;
            return sum / numMeans;
        }

        void calculateMeans() {
            auto *oldMeans = new double[numMeans];
            do {
                for (int i = 0; i < numMeans; i++)
                    oldMeans[i] = means[i].finalSum;
                findNewKmeans();
            } while (!converged(oldMeans));
            delete[] oldMeans;
        }

        void findNewKmeans() {
            for (int i = 0; i < numMeans; i++) {
                means[i].countSum = 0;
                means[i].weightedSum = 0;
            }
            for (int i = 0; i <= 255; i++) {
                Common::Mean *closest = findClosestKmean(i);
                closest->weightedSum += distribution[i] * i;
                closest->countSum += distribution[i];
            }
            for (int i = 0; i < numMeans; i++)
                means[i].finalSum = (means[i].countSum > 0 ? (means[i].weightedSum * 1.0 / means[i].countSum)
                                                           : means[i].finalSum);
        }

        Common::Mean *findClosestKmean(double value) {
            Common::Mean *closest = nullptr;
            auto closestValue = (double) pixelRange;
            for (int i = 0; i < numMeans; i++)
                if (abs(value - means[i].finalSum) < closestValue) {
                    closest = &means[i];
                    closestValue = abs(value - means[i].finalSum);
                }
            return closest;
        }

        bool converged(const double *values) {
            bool yes = true;
            for (int i = 0; i < numMeans; i++)
                if (abs(means[i].finalSum - values[i]) > 0.01)
                    yes = false;
            return yes;
        }

        bool isHigherMean(Common::Mean *mean) {
            for (int i = 0; i < numMeans; i++)
                if(mean->finalSum < means[i].finalSum)
                    return false;
            return true;
        }
    };
}
#endif //IMG_PREPROCESSOR_H