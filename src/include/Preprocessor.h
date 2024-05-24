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
    Image* gaussian(Image* image, unsigned int sigma=0, unsigned int kernelDim=0);

    Image * grayscale(Image * image);

    Image *dilate(Image *image, int kernelDim=0);

    Image * erode(Image * image, int kernelDim=0);

    Image *binary(Image *image);
    Image* edge(Image* image);
}
#endif //IMG_PREPROCESSOR_H