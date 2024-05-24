//
// Created by jonossar on 3/8/24.
//

#ifndef IMG1_IMAGE_H
#define IMG1_IMAGE_H

#include <string>
#include <cstring>
#include "stb_image.h"
#include "stb_image_write.h"

#include <filesystem>
#include <utility>
#include <thread>
typedef struct FileName{
    std::string fileBaseName;
    std::string fileExt;
    ~FileName(){
        fileExt.clear(); fileBaseName.clear();
    }
    explicit FileName(const FileName& other): fileBaseName(other.fileBaseName), fileExt(other.fileExt){}
    explicit FileName(FileName&& other) noexcept: fileBaseName(std::move(other.fileBaseName)), fileExt(std::move(other.fileExt)){
    }
    FileName& operator=(const FileName& other){
        this->fileBaseName=other.fileBaseName;this->fileExt=other.fileExt;
        return *this;
    }
    FileName& operator=(FileName&& other){
        this->fileBaseName=std::move(other.fileBaseName);this->fileExt=std::move(other.fileExt);
        return *this;
    }
    FileName(std::string&& base, std::string&& ext="jpg"): fileBaseName(base) ,fileExt(ext){}
    FileName()=default;
} FileName;
class Image{
public:
    unsigned char * data;
    int x;
    int y;
    int channels;
    FileName fileName;
private :    static inline FileName getFileName(const char* path){
        return getFileName(std::string (path));
    }
    static inline FileName getFileName(std::string path){
        std::string fullName=path.substr(path.find_last_of('/')+1);
        auto dot=fullName.find_last_of('.') + 1;
        return {fullName.substr(0, fullName.length()- dot), fullName.substr(dot)};
    }
public:
    Image (const char * filePath, int desired_channels): fileName(getFileName(filePath)){
        FILE * f= fopen(filePath, "r");
        auto* tmp = stbi_load_from_file(f, &x, &y, &channels, desired_channels);
        this->data=new unsigned char [x*y*channels];
        for(int i=0; i<x*y*channels; i++) this->data[i]=tmp[i];
        free(tmp);
        fclose(f);
    }
    Image(const char * filePath): Image(filePath, 3){}
    Image(const Image& other): x(other.x) ,y(other.y), channels(other.channels), data(new unsigned char[other.size()]), fileName(other.fileName){
    for(int i=0; i<other.size(); i++) this->data[i]=other.data[i];
}
    Image(Image&& other): x(other.x), y(other.y) , channels(other.channels), data(other.data), fileName(std::move(other.fileName)){}
    Image(unsigned char * data, int x, int y, int ch, FileName&& fileName) :data(data), x(x), y(y), channels(ch), fileName(std::move(fileName)){}
    void save(std::string writeDir);
    void setFileName(std::string name){
        this->fileName.fileBaseName=fileName.fileBaseName=name.substr(0, name.find('.'));
        this->fileName.fileExt=name.substr(name.find('.'));
    }
    [[nodiscard]] constexpr inline unsigned int size() const{
        return x*y*channels;
    }
private:
};
#endif //IMG_IMAGE_H