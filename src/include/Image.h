//
// Created by jonossar on 3/8/24.
//

#ifndef IMG1_IMAGE_H
#define IMG1_IMAGE_H

#include <string>
#include <cstring>
#include "stb_image.h"
#include "stb_image_write.h"
typedef struct{
    std::string fileBaseName;
    std::string fileExt;
} FileName;
class Image{
public:
    unsigned char * data;
    int x;
    int y;
    int channels;
    FileName fileName;
    Image(const char * filePath) {
        FILE * f= fopen(filePath, "r");
        auto* tmp = stbi_load_from_file(f, &x, &y, &channels, 3);
        this->data=new unsigned char [x*y*channels];
        std::memmove(this->data, tmp, x*y*channels);
        free(tmp);
        initFileName(filePath);
        fclose(f);
    }
    Image (char * filePath, int desired_channels){
        FILE * f= fopen(filePath, "r");
        auto* tmp = stbi_load_from_file(f, &x, &y, &channels, desired_channels);
        this->data=new unsigned char [x*y*channels];
        std::memmove(this->data, tmp, x*y*channels);
        free(tmp);
        initFileName(filePath);
        fclose(f);
    }
    Image(unsigned char * data, int x, int y, int ch, FileName fileName){
        this->data=data;
        this->x=x;
        this->y=y;
        this->channels=ch;
        this->fileName=std::move(fileName);
    }
    void save(const char * writeDir){
        std::string absolutePath=(std::string(writeDir)).append(fileName.fileBaseName).append(fileName.fileExt);
        stbi_write_jpg(absolutePath.c_str(), x, y, channels, data, 100);
    }
    void setFileName(const char * name){
        this->fileName.fileBaseName=name;
    }
private:
    void initFileName(const char * filePath){
        std::string filePathStr(filePath);
        std::string fileNameStr= filePathStr.substr(filePathStr.find_last_of("/")+1);
        fileName.fileBaseName=fileNameStr.substr(0,fileNameStr.find('.'));
        fileName.fileExt=fileNameStr.substr(fileNameStr.find('.'));
    }
};
#endif //IMG_IMAGE_H