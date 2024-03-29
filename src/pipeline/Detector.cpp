//
// Created by jonossar on 3/19/24.
//
#include <cstring>
#include "../include/Detector.h"
#include "common.h"
//not threadSafe
    void Detector::label(unsigned int i){
        for (unsigned int j = 0; j < kernelDim * kernelDim; j++) {
            int y_pos = (int)i / x + (j / kernelDim - kernelDim / 2);
            int x_pos = (int)i % x + (j % kernelDim - kernelDim / 2);
            if(x_pos<0 || y_pos<0 ||x_pos>=(int)x || y_pos>=(int)y) continue;
            unsigned int idx=y_pos*x + x_pos;
            if(data[idx]==(unsigned char)0 && !Common::isNotWhite(&(out[idx * 3]))){
                if(y_pos<(int)currentPos->y_start) {
                    currentPos->y_start=y_pos;
                } else if(y_pos>(int)currentPos->y_end)
                    currentPos->y_end=(unsigned int)y_pos;
                if(x_pos<(int)currentPos->x_start){
                    currentPos->x_start=x_pos;
                }else if (x_pos>(int)currentPos->x_end)
                    currentPos->x_end=(unsigned int)x_pos;
                Common::writeColor(&out[idx*3], currentColor);
                label(idx);
            }
        }
    }
    Common::ObjectLabel * Detector::detect(unsigned int* labelLength){
        printf("--Started detecting--\n");
//        boost::thread::attributes attr;
//        boost::thread::id id= boost::this_thread::get_id();
        std::vector<Common::ObjectLabel> labels{};

        printf("picked up execution\n");
        for (unsigned int i = 0; i < x * y; i++) {
            if (data[i] == (unsigned char) 0) {
                if (Common::isNotWhite(&out[i * 3])) continue;
                currentColor = Common::randomColor();
                currentPos = new Common::ObjectLabel(i % x, i / x);
                label(i);
                delete currentColor;
                currentPos->x_len=currentPos->x_end-currentPos->x_start+1;
                currentPos->y_len=currentPos->y_end-currentPos->y_start+1;
                labels.push_back(*currentPos);
                delete currentPos;
            }
        }
//        {
//            boost::unique_lock<boost::mutex> lock(mutex);
//            ++finishedThreads;
//        }
//        cond.notify_all();
        *labelLength=labels.size();
        printf("Initial object count: %d\n", *labelLength);
        double threshold=0;
        for(unsigned int i=0; i< *labelLength; i++){
            double s=labels[i].y_len*labels[i].x_len*1.0;
            if(s>threshold) threshold=s;
        }
        threshold/=5;
//        for (int i=0; i<*labelLength; i++){
//            if(abs(labels[i].y_end*1.0-labels[i].y_start)*abs(labels[i].x_end*1.0 - labels[i].x_start)<threshold/3){
//                ObjectLabel small=labels[i];
//                ObjectLabel * closest=  findClosestLabel(labels.data(),&small, *labelLength, threshold);
//                closest->x_start=small.x_start<closest->x_start?small.x_start:closest->x_start;
//                closest->x_end=small.x_end>closest->x_end?small.x_end:closest->x_end;
//                closest->y_start=small.y_start<closest->y_start?small.y_start:closest->y_start;
//                closest->y_end=small.y_end>closest->y_end?small.y_end:closest->y_end;
//            }
//        }
        std::vector<Common::ObjectLabel> filteredLabels{};
        for(unsigned int i=0;i<*labelLength; i++){
            if(labels[i].y_len*labels[i].x_len*1.0 > threshold){
                filteredLabels.push_back(labels[i]);
            }
        }
        *labelLength=filteredLabels.size();
        printf("filtered object count: %d\n", *labelLength);
        delete[] image->data;
        image->data=out;
        image->channels=3;
        if(RENAME_IMAGE) image->fileName.fileBaseName.append("_detect");
        auto * ret= (Common::ObjectLabel*) malloc(*labelLength * sizeof(Common::ObjectLabel));
        memcpy(ret, filteredLabels.data(), *labelLength* sizeof(Common::ObjectLabel));
        labels.clear();
        filteredLabels.clear();
        return ret;
    }
    Detector::Detector(Image *image1) {
    this->x=image1->x;
    this->y=image1->y;
    this->image=image1;
    this->data=image1->data;
    out=Common::initializeArray((unsigned char) 255, x*y*3);
}
//    static const unsigned int kernelDim=3;
////    Common::Color currentColor;
//    static void label(unsigned int i, int x, int y,const unsigned char* in, unsigned char * out,Common::ObjectLabel currentPos, Common::Color currentColor){
//        for (int j = 0; j < kernelDim * kernelDim; j++) {
//            int y_pos = i / x + (j / kernelDim - kernelDim / 2);
//            int x_pos = i % x + (j % kernelDim - kernelDim / 2);
//            if(x_pos<0 || y_pos<0 ||x_pos>=x || y_pos>=y) continue;
//            unsigned int idx=y_pos*x + x_pos;
//            if(in[idx]==(unsigned char)0 && !Common::isNotWhite(&(out[idx * 3]))){
//                if(y_pos<currentPos.y_start) {
//                    currentPos.y_start=y_pos;
//                } else if(y_pos>currentPos.y_end) currentPos.y_end=(unsigned int)y_pos+1;
//                if(x_pos<currentPos.x_start){
//                    currentPos.x_start=x_pos;
//                }else if (x_pos>currentPos.x_end) currentPos.x_end=(unsigned int)x_pos+1;
//                Common::writeColor(&out[idx*3], currentColor);
//                label(idx, x,y, in,out, currentPos,currentColor);
//            }
//        }
//    }
//    ObjectLabel * findClosestLabel(ObjectLabel * labels,ObjectLabel * label, int length, double threshold){
//        auto closestValue=DBL_MAX;
//        ObjectLabel * closest;
//        for(int i=0; i<length; i++){
//            double x_center=abs(labels[i].x_start*1.0+labels[i].x_end)/2;
//            double y_center=abs(labels[i].y_start*1.0 + labels[i].y_end)/2;
//            double x_center1=abs(label->y_end*1.0+label->y_start)/2;
//            double y_center1=abs(label->x_end*1.0+ label->x_start)/2;
//            double dist=sqrt(pow(x_center-x_center1,2.0) + pow(y_center-y_center1,2.0) );
//            if (dist<closestValue && abs(labels[i].y_end*1.0-labels[i].y_start)*abs(labels[i].x_end*1.0 - labels[i].x_start)>threshold ){
//                closest=&labels[i];
//                closestValue=dist;
//            }
//        }
//        return closest;
//    }
//    static Common::ObjectLabel * Detect(Image * image, unsigned int* labelLength){
//        printf("--Started detecting--\n");
//        std::vector<Common::ObjectLabel> labels{};
//        const unsigned  char * in=image->data;
//        auto * out=Common::initializeArray((unsigned char) 255, image->x*image->y*3);
//        Common::Color currentColor;
//        for (int i = 0; i < image->x * image->y; i++) {
//            if (in[i] == (unsigned char) 0) {
//                if (Common::isNotWhite(&out[i * 3])) continue;
//                currentColor = Common::randomColor();
//                Common::ObjectLabel objectLabel = Common::ObjectLabel(i % image->x, i / image->x);
//                label(i, image->x, image->y, in, out, objectLabel, currentColor);
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
////                ObjectLabel small=labels[i];
////                ObjectLabel * closest=  findClosestLabel(labels.data(),&small, *labelLength, threshold);
////                closest->x_start=small.x_start<closest->x_start?small.x_start:closest->x_start;
////                closest->x_end=small.x_end>closest->x_end?small.x_end:closest->x_end;
////                closest->y_start=small.y_start<closest->y_start?small.y_start:closest->y_start;
////                closest->y_end=small.y_end>closest->y_end?small.y_end:closest->y_end;
////            }
////        }
//        std::vector<Common::ObjectLabel> filteredLabels{};
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
//        if(RENAME_IMAGE) image->fileName.fileBaseName.append("_detect");
//        auto * ret= (Common::ObjectLabel*) malloc(*labelLength * sizeof(Common::ObjectLabel));
//        memcpy(ret, filteredLabels.data(), *labelLength* sizeof(Common::ObjectLabel));
//        labels.clear();
//        filteredLabels.clear();
//         return ret;
//    }
// 
// Detector