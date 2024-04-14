//
// Created by jonossar on 3/21/24.
//

#ifndef IMG1_MODEL_H
#define IMG1_MODEL_H

#include "json.hpp"
#include <fstream>
#include "opencv2/opencv.hpp"
#include "Image.h"
#include "Preprocessor.h"
#include "Detector.h"
#include "Extractor.cuh"

#define MODEL_DATA_DIR std::string("./model/")
#define MODEL_DATA std::string("model.json")
#define INPUT_DIR (char *) "./in/"
#define OUTPUT_DIR (char *) "./out/"
#define TRAINING_DIR (char *) "./training/"
#define TESTING_DIR (char *) "./test/"
namespace Model {
//void train(Image * data, const char * label);
void train(const char* trainDir);
Image * infer(Image * data);
    enum Normalization{
        MINMAX,
        STANDARD,
        NONE
    };
};


#endif //IMG1_MODEL_H
