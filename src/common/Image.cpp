//
// Created by jonossar on 3/8/24.
//
#include "../include/Image.h"
#ifndef IMG1_COMMON_H
#include "../include/common.h"
#endif
#define IMG1_IMAGE_IMPLEMENTATION
#define MAX_SAVING_THREADS 2
void Image::save(std::string writeDir){
    auto* i=new Image(*this);
    Common::fork([i=i, writeDir=writeDir](){
        if(!std::filesystem::exists(writeDir)) std::filesystem::create_directories(writeDir.c_str());
        std::string absolutePath=(std::string(writeDir)).append("/").append(i->fileName.fileBaseName).append(".").append(i->fileName.fileExt);
        stbi_write_jpg(absolutePath.c_str(), i->x, i->y, i->channels, i->data, 100);
        delete i;
    });
}