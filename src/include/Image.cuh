//
// Created by jonossar on 3/8/24.
//

#ifndef IMG1_IMAGE_CUH
#define IMG1_IMAGE_CUH

#include <string>
#include "stb_image.h"
#include "stb_image_write.h"
#include "../include/Tensor.cuh"
#include "../include/common.h"

typedef struct{
    std::string fileBaseName;
    std::string fileExt;
} FileName;
class Image : public Tensor<unsigned char>{
public:
    FileName fileName;
    __host__ Image(const char * filePath){
        this->data=stbi_load(filePath, reinterpret_cast<int *>(&x), reinterpret_cast<int *>(&y),
                             reinterpret_cast<int *>(&z), 3);
        initFileName(filePath);
    }
    __host__ Image(Image& other): Tensor<unsigned char>(other), fileName(other.fileName){
    }
    __host__ Image():Tensor<unsigned char>(), fileName(FileName()){}
    __host__ Image(unsigned char * data, unsigned int x, unsigned int y, unsigned int ch, FileName&& fileName): Tensor<unsigned char>(data, x,y,ch){
        this->fileName=fileName;
    }
    void save(const char * writeDir){
        std::string absolutePath=(std::string(writeDir)).append(fileName.fileBaseName).append(fileName.fileExt);
        stbi_write_jpg(absolutePath.c_str(), x, y, z, data, 100);
    }
protected:
    virtual size_t _size(){
        return sizeof(*this);
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