//
// Created by jonossar on 3/8/24.
//
#include "include/Image.h"
#include "include/Preprocessor.h"
#include "include/Kernel.cuh"
#include "include/Model.h"
#include "opencv2/opencv.hpp"
#include "common.h"

//static inline const int sLen=2;
//static inline Model::DistanceMethod storageMethods[sLen]={Model::DistanceMethod::Mean, Model::DistanceMethod::Covariance};
//static inline Model::Model* model=new Model::Model(true, true,sLen, storageMethods);
void init(){}
void clean(){
    Common::waitChildren();
}
static inline int getRange(Image * img){
    return (int)ceil(hypot(img->x-1,img->y-1) ) +1;
}
unsigned int* getHough(Image* img){
    int range= getRange(img);
    auto * hough= Kernel::executor<unsigned char,unsigned int,double,Kernel::edge::houghLine>(
            img->data,nullptr,Kernel::edge::houghLine(),img->x,img->y,img->channels, range*360,0
    );
    auto * cpy=(unsigned int*) malloc(sizeof(unsigned int)*range*360);
    memcpy(cpy,hough, sizeof(unsigned int)*range*360);
    img->fileName.fileBaseName.append("_hough");
    Image(Kernel::normalize<unsigned int, unsigned char>(cpy,range*360),360,range,1,FileName(img->fileName)).save(STEP_OUT_DIR);
    free(cpy);
    return hough;
}
unsigned char* houghLine(Image* img){
#define numMeans 3
    int range= getRange(img);
    auto* hough=getHough(img);
    unsigned int max=hough[0];
    for(int i=0; i<range*360; i++){
        if(hough[i] > max) max=hough[i];
    }
#define maxLines 30
    auto * sorted=Kernel::sort(hough,range*360);
    auto  threshold=sorted[ (int)(range*360  - maxLines)];
    auto * filtered= new unsigned char [range* 360];
    for(int i=0; i< range*360; i++) if(hough[i] > threshold) filtered[i]=255; else filtered[i]=0;
//    auto * dist= Kernel::executor<unsigned int, unsigned int, double, Kernel::distribution<unsigned int>>(
//            hough, nullptr, Kernel::distribution<unsigned int>(0, max), 360, range, 1, max, 0
//    );
//    Common::Mean* means= Common::Mean::getMeans(dist, max,numMeans);
//    auto * filtered= Kernel::executor<unsigned int, unsigned char, double, Kernel::cluster<unsigned int, unsigned char>>(
//            hough, nullptr, Kernel::cluster<unsigned int, unsigned char>(means, numMeans, 0, 255, (int)(numMeans)), 360, range, 1, range * 360, 0
//    );
#undef numMeans
    if(SHOW_STEPS) {
        img->fileName.fileBaseName.append("_filtered");
        Image(filtered, 360, range, 1, FileName(img->fileName)).save(STEP_OUT_DIR);
    }
    return filtered;
}
    void matrixTests(){
    using Matrix=Matrix<Model::dataType>;
    Matrix::SHOW_MATRIX_RESULTS=false;
    Matrix m1(3,5,{5,0,9,8,7,6,6,7,8,4,5,10,2,1,7}); // centering
    auto centered=   m1.center();
    cout <<"centered: " << endl << centered;
    cout << "mean: "<< endl  << centered.mean();
    Matrix m2(3, 3,{3,4,3,6,8,3,5,12,5});
    auto projection= Matrix::projection(m2, 0.6, false);
    cout << "projection: " <<endl << projection;
    auto projected=m2*projection;
    cout<< "projected: "<<endl <<projected; // output should be {1,2,8}
    Matrix m3(3,3, {1,0,-5,0,1,-5,0,0,1});
    auto inv=m3.inverse();
    assert(inv==Matrix(3,3,{1,0,5,0,1,5,0,0,1}));
//    Matrix normal(2,3,{3,3,4,4,12,12});
//    assert(normal.normalize(Y)==Matrix(2,2,{3.0/5,3.0/5,4.0,5.0,4.0/5}));
}
    cv::Point * getLinePoints(const int r,const  int t, const Image& img){
        auto * ret= new cv::Point[2];
        double houghAng= (t/360.0)*2*M_PI;
        double initX= r * cos(houghAng);
        double initY= r * sin(houghAng);
        double lineAng;
        switch (t /90) {
            case 0:
                lineAng=(t+ 90);
                break;
            case 1:
            case 3:
                lineAng=t-90;
                break;
            default: return nullptr;
    }
    lineAng =(lineAng/360) * 2*M_PI;
    double stepX= cos(lineAng), stepY= sin(lineAng);
    if((initX < 0 && stepX <0) || (initY <0 && stepY <0)) {
        return nullptr;
    }
    while(initX <0 || initY <0 ){
        if(initX >= img.x || initY >= img.y) return nullptr;
        initX+= stepX, initY +=stepY;
    }
    double xStart=initX, yStart=initY;
    while(xStart >= 0 && yStart >= 0 && xStart < img.x && yStart < img.x) xStart += stepX, yStart-= stepY;
    xStart -= stepX,yStart += stepY;
    double xEnd=initX, yEnd=initY;
    while (xEnd < img.x && yEnd < img.y && xEnd >=0 && yEnd >=0) xEnd -= stepX, yEnd+= stepY;
    xEnd += stepX, yEnd-= stepY;
    ret[0].x= floor(xStart), ret[0].y= floor(yStart), ret[1].x= ceil(xEnd), ret[1].y= ceil(yEnd);
    if (hypot(abs(xEnd - xStart), abs(yEnd - yStart)) < sqrt(img.x*img.y)) return nullptr;
    return ret;
}
    Image* detectLines(Image* img){
        Image cpy(*img);
        img=Preprocessing::edge(img);
        int range= getRange(img);
        auto * hough=houghLine(img);
        for(int r=0; r < range; r++){
            for(int t=0; t < 360; t++){
                if(hough[r * 360 + t] == 255) {
                    auto * points= getLinePoints(r,t,*img);
                    if(points== nullptr) continue;
                    cv::line(cv::Mat(img->y, img->x, CV_8UC3, cpy.data),points[0],points[1],{255,0,0},1);
                }
            }
        }
        cpy.channels=3;
        if (SHOW_STEPS){
            cpy.fileName.fileBaseName.append("_lines");
            cpy.save(Common::OUTPUT_DIR);
        }
        return img;
    }
void fullEdgeTest(Image* image){
    Preprocessing::edge(image)->save(Common::OUTPUT_DIR);
}
void thrustTests(){
//    Kernel::edge::gradient gradients[10]={{26.4,0},{1.26,0},{19.5,1},{-3.5,3},{-5.634,3},{-65.3,1}
//            ,{100.3,1},{15.4,1},{10.5,1},{0.5,1}};
//    std::vector<Kernel::edge::gradient>::iterator first(gradients);
//    std::vector<Kernel::edge::gradient>::iterator last(&gradients[10]);
//    thrust::host_vector<Kernel::edge::gradient> host(first,last);
//    for(auto& g: host){
//        std::cout << g.value <<std::endl;
//    }
//    auto max=thrust::max_element(first,last);
//    thrust::device_vector<Kernel::edge::gradient> dev(10);
//    thrust::copy(host.begin(), host.end(), dev.begin());
//    thrust::sort(thrust::device.on(thisStream),dev.begin(), dev.end());
//    thrust::copy(dev.begin(), dev.end(), host.begin());
//    std::cout << std::endl;
//    for(auto& g:host){
//        std::cout << g.value <<std::endl;
//    }
//    std::cout << std::endl;
//    std::cout << host[4].value;
}
int __host__  main(){
    init();
    auto * image=new Image("/home/jonossar/proj/nutdetector1/images/edge/dama.png");
//    fullEdgeTest(image);
    detectLines(image);
//    matrixTests();
    clean();
}