//
// Created by jonossar on 3/19/24.
//

#ifndef IMG1_DETECTOR_H
#define IMG1_DETECTOR_H
#include <sys/resource.h>
#ifndef IMG1_COMMON_H
#include "common.h"
#endif
#include <boost/thread.hpp>
class Detector {
public:
    Detector(Image * image);
    Common::ObjectLabel * detect(unsigned int* labelLength);
private:
    Common::ObjectLabel * currentPos;
    Common::Color * currentColor;
    static const unsigned int kernelDim=3;
    const unsigned char* data;
    Image * image;
    unsigned char * out;
    unsigned int x;
    unsigned int y;
    void label(unsigned int i);
}; // Detector

#endif //IMG1_DETECTOR_H
