//
// Created by jonossar on 3/19/24.
//
#include <cstring>
#include <stack>
#include "../include/Detector.h"
#include "common.h"
#define THRESHOLDING_MODE 1
enum Direction{
    N,NE,E,NW,W,SE,SW,S,NONE
};
/** this function goes into every direction from starting point and paints every pixel with the starting color if they are marked as an containing an object in the binary input image
 *
 * @param initial
 */
    void Detector::_detect(unsigned int initial){
        using pairTp=std::pair<int, int>;
    auto& stack= *new std::stack<pairTp>;
    stack.emplace(initial, 0);
    do{
        auto& idx=stack.top();stack.pop();
        int y_pos= idx.first /(int) x;
        int x_pos= idx.first % (int)x;
        if(y_pos >=y || y_pos <0 ||x_pos >=x || x_pos <0) continue;
        if(data[idx.first] == (unsigned char) 0 && Common::isWhite(&out[idx.first * 3])){
            if(y_pos<(int)currentPos.y_start) {
                currentPos.y_start=y_pos;
            } else if(y_pos>(int)currentPos.y_end)
                currentPos.y_end=(unsigned int)y_pos;
            if(x_pos<(int)currentPos.x_start){
                currentPos.x_start=x_pos;
            }else if (x_pos>(int)currentPos.x_end)
                currentPos.x_end=(unsigned int)x_pos;
            Common::writeColor(&out[idx.first*3], currentColor);
            for(int i=0; i< kernelDim; i++){
                for( int j=0; j< kernelDim  ;j++){
                    int y_new=y_pos+ i -((int)kernelDim/2);
                    int x_new=x_pos+ j -(int)kernelDim/2;
                    stack.emplace(y_new * x + x_new,i * kernelDim + j);
                }
            }
        }
    } while (!stack.empty());
    delete& stack;
}
//    void Detector::_detect(unsigned int i){
//        for (unsigned int j = 0; j < kernelDim * kernelDim; j++) {
//            int y_pos = (int)i / x + (j / kernelDim - kernelDim / 2);
//            int x_pos = (int)i % x + (j % kernelDim - kernelDim / 2);
//            if(x_pos<0 || y_pos<0 ||x_pos>=(int)x || y_pos>=(int)y) continue;
//            unsigned int idx=y_pos*x + x_pos;
//            if(data[idx]==(unsigned char)0 && Common::isWhite(&(out[idx * 3]))){
//                if(y_pos<(int)currentPos.y_start) {
//                    currentPos.y_start=y_pos;
//                } else if(y_pos>(int)currentPos.y_end)
//                    currentPos.y_end=(unsigned int)y_pos;
//                if(x_pos<(int)currentPos.x_start){
//                    currentPos.x_start=x_pos;
//                }else if (x_pos>(int)currentPos.x_end)
//                    currentPos.x_end=(unsigned int)x_pos;
//                Common::writeColor(&out[idx*3], currentColor);
//                _detect(idx);
//            }
//        }
//    }
/** performs detection on the image
 * it traverses every pixel and calls @p _detect to paint every pixel connecting to the pixel @p data[i] with the same color. It then performs a filtering on the found object positions depending on their size
 * @param labelLength
 * @return array containing found object positions
 */
    Common::ObjectPosition * Detector::detect(unsigned int* labelLength){
        printf("--Started detecting--\n");
//        boost::thread::attributes attr;
//        boost::thread::id id= boost::this_thread::get_id();
        std::vector<Common::ObjectPosition> positions{};
        for (unsigned int i = 0; i < x * y; i++) {
            if (data[i] == (unsigned char) 0) {
                if (!Common::isWhite(&out[i * 3])) continue;
                currentColor = Common::randomColor();
                currentPos= Common::ObjectPosition(i % x, i / x);
                _detect(i);
                currentPos.x_len=currentPos.x_end-currentPos.x_start+1;
                currentPos.y_len=currentPos.y_end-currentPos.y_start+1;
                positions.push_back(currentPos);
            }
        }
//        {
//            boost::unique_lock<boost::mutex> lock(mutex);
//            ++finishedThreads;
//        }
//        cond.notify_all();
        *labelLength=positions.size();
        printf("Initial object count: %d\n", *labelLength);
        double threshold=0;
        if(THRESHOLDING_MODE==0) { //max thresholding
            for (unsigned int i = 0; i < *labelLength; i++) {
                double s = positions[i].y_len * positions[i].x_len * 1.0;
                if (s > threshold) threshold = s;
            }
            threshold /= 5;
        } else if(THRESHOLDING_MODE==1){ //averaging
            double sum=0.0;
            unsigned int count=0;
            for (unsigned int i = 0; i < *labelLength; i++) {
                sum += positions[i].y_len * positions[i].x_len * 1.0;
                count++;
            }
            threshold=sum/(count*3);
        }
//        for (int i=0; i<*labelLength; i++){
//            if(abs(positions[i].y_end*1.0-positions[i].y_start)*abs(positions[i].x_end*1.0 - positions[i].x_start)<threshold/3){
//                ObjectPosition small=positions[i];
//                ObjectPosition * closest=  findClosestLabel(positions.data(),&small, *labelLength, threshold);
//                closest->x_start=small.x_start<closest->x_start?small.x_start:closest->x_start;
//                closest->x_end=small.x_end>closest->x_end?small.x_end:closest->x_end;
//                closest->y_start=small.y_start<closest->y_start?small.y_start:closest->y_start;
//                closest->y_end=small.y_end>closest->y_end?small.y_end:closest->y_end;
//            }
//        }
        std::vector<Common::ObjectPosition> filteredLabels{};
        for(unsigned int i=0;i<*labelLength; i++){
            if(positions[i].y_len * positions[i].x_len * 1.0 > threshold){
                filteredLabels.push_back(positions[i]);
            }
        }
        *labelLength=filteredLabels.size();
        printf("filtered object count: %d\n", *labelLength);
        delete[] image->data;
        image->data=out;
        image->channels=3;
        if(SHOW_STEPS) prntstps(image, "cluster")
        auto * ret= (Common::ObjectPosition*) malloc(*labelLength * sizeof(Common::ObjectPosition));
        memcpy(ret, filteredLabels.data(), *labelLength* sizeof(Common::ObjectPosition));
        positions.clear();
        filteredLabels.clear();
        return ret;
    }
    Detector::Detector(Image *image1) {
    this->x=image1->x;
    this->y=image1->y;
    this->image=image1;
    this->data=image1->data;
    out=Common::initializeArray<unsigned char>((unsigned char) 255, x*y*3);
}
//    static const unsigned int kernelDim=3;
////    Common::Color currentColor;
//    static void _detect(unsigned int i, int x, int y,const unsigned char* in, unsigned char * out,Common::ObjectPosition currentPos, Common::Color currentColor){
//        for (int j = 0; j < kernelDim * kernelDim; j++) {
//            int y_pos = i / x + (j / kernelDim - kernelDim / 2);
//            int x_pos = i % x + (j % kernelDim - kernelDim / 2);
//            if(x_pos<0 || y_pos<0 ||x_pos>=x || y_pos>=y) continue;
//            unsigned int lastX=y_pos*x + x_pos;
//            if(in[lastX]==(unsigned char)0 && !Common::isWhite(&(out[lastX * 3]))){
//                if(y_pos<currentPos.y_start) {
//                    currentPos.y_start=y_pos;
//                } else if(y_pos>currentPos.y_end) currentPos.y_end=(unsigned int)y_pos+1;
//                if(x_pos<currentPos.x_start){
//                    currentPos.x_start=x_pos;
//                }else if (x_pos>currentPos.x_end) currentPos.x_end=(unsigned int)x_pos+1;
//                Common::writeColor(&out[lastX*3], currentColor);
//                _detect(lastX, x,y, in,out, currentPos,currentColor);
//            }
//        }
//    }
//    ObjectPosition * findClosestLabel(ObjectPosition * labels,ObjectPosition * _detect, int length, double threshold){
//        auto closestValue=DBL_MAX;
//        ObjectPosition * closest;
//        for(int i=0; i<length; i++){
//            double x_center=abs(labels[i].x_start*1.0+labels[i].x_end)/2;
//            double y_center=abs(labels[i].y_start*1.0 + labels[i].y_end)/2;
//            double x_center1=abs(label->y_end*1.0+_detect->y_start)/2;
//            double y_center1=abs(label->x_end*1.0+ _detect->x_start)/2;
//            double dist=sqrt(pow(x_center-x_center1,2.0) + pow(y_center-y_center1,2.0) );
//            if (dist<closestValue && abs(labels[i].y_end*1.0-labels[i].y_start)*abs(labels[i].x_end*1.0 - labels[i].x_start)>threshold ){
//                closest=&labels[i];
//                closestValue=dist;
//            }
//        }
//        return closest;
//    }
//    static Common::ObjectPosition * Detect(Image * image, unsigned int* labelLength){
//        printf("--Started detecting--\n");
//        std::vector<Common::ObjectPosition> labels{};
//        const unsigned  char * in=image->data;
//        auto * out=Common::initializeArray((unsigned char) 255, image->x*image->y*3);
//        Common::Color currentColor;
//        for (int i = 0; i < image->x * image->y; i++) {
//            if (in[i] == (unsigned char) 0) {
//                if (Common::isWhite(&out[i * 3])) continue;
//                currentColor = Common::randomColor();
//                Common::ObjectPosition objectLabel = Common::ObjectPosition(i % image->x, i / image->x);
//                _detect(i, image->x, image->y, in, out, objectLabel, currentColor);
//                labels.push_back(objectLabel);
//            }
//        }
//        *labelLength=labels.size();
//        printf("Initial object count: %d\n", *labelLength);
//        double threshold=0;
//        for(int i=0; i< *labelLength; i++){
//             double s=abs(labels[i].y_end*1.0-labels[i].y_start)*abs(labels[i].x_end*1.0 - labels[i].x_start);
//             if(s>threshold) threshold=s;
//        }
//        threshold/=6;
////        for (int i=0; i<*labelLength; i++){
////            if(abs(labels[i].y_end*1.0-labels[i].y_start)*abs(labels[i].x_end*1.0 - labels[i].x_start)<threshold/3){
////                ObjectPosition small=labels[i];
////                ObjectPosition * closest=  findClosestLabel(labels.data(),&small, *labelLength, threshold);
////                closest->x_start=small.x_start<closest->x_start?small.x_start:closest->x_start;
////                closest->x_end=small.x_end>closest->x_end?small.x_end:closest->x_end;
////                closest->y_start=small.y_start<closest->y_start?small.y_start:closest->y_start;
////                closest->y_end=small.y_end>closest->y_end?small.y_end:closest->y_end;
////            }
////        }
//        std::vector<Common::ObjectPosition> filteredLabels{};
//        for(int i=0;i<*labelLength; i++){
//            if(abs(labels[i].y_end*1.0-labels[i].y_start)*abs(labels[i].x_end*1.0 - labels[i].x_start) > threshold){
//                filteredLabels.push_back(labels[i]);
//            }
//        }
//        *labelLength=filteredLabels.size();
//        printf("filtered object count: %d\n", *labelLength);
//        free(image->data);
//        image->data=out;
//        image->channels=3;
//        if(SHOW_STEPS) image->fileName.fileBaseName.append("_detect");
//        auto * ret= (Common::ObjectPosition*) malloc(*labelLength * sizeof(Common::ObjectPosition));
//        memcpy(ret, filteredLabels.data(), *labelLength* sizeof(Common::ObjectPosition));
//        labels.clear();
//        filteredLabels.clear();
//         return ret;
//    }
// 
// Detector