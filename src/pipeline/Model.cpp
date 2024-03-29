//
// Created by jonossar on 3/21/24.
//

#include <filesystem>
#include "../include/Model.h"
#include "common.h"
#include <boost/thread.hpp>

namespace Model{
    using json=nlohmann::json;
    Normalization normalization= Normalization::MINMAX;
    Preprocessing::Preprocessor * preprocessor=new Preprocessing::Preprocessor(2,256);
    long double * trainOne(Image * sample){
        preprocessor->polarize(sample, Preprocessing::GAUSSIAN);
        unsigned int numObjects;
        auto * detector=new Detector(sample);
        Common::ObjectLabel * objects= detector->detect( &numObjects);
        auto * featuresSum=Common::initializeArray((long double) 0, FEATURES_LENGTH);
//        cv::Mat cv_image(sample->y, sample->x, CV_8UC3, sample->data);
        long double ** featuresAll= FeatureExtractor::extractFeatures(objects, sample, numObjects);
        for(unsigned int i=0; i<numObjects; i++) {
            for (int j = 0; j < FEATURES_LENGTH; j++) {
                featuresSum[j] += featuresAll[i][j];
            }
            delete[] featuresAll[i];
//            cv::rectangle(cv_image, cv::Point(objects[i].x_start - 2, objects[i].y_start - 2), cv::Point(objects[i].x_end + 2, objects[i].y_end + 2), cv::Scalar_<unsigned char>(255, 0, 0), 5);
        }
        //TODO DISABLED AVERAGING
        for(unsigned int i= 0; i<FEATURES_LENGTH; i++)
            featuresSum[i]/=numObjects*1.0;
        delete[] featuresAll;
        free(objects);
        delete detector;
        return featuresSum;
//        out->insert_or_assign(sample->fileName.fileBaseName.c_str(), featuresSum);
    }
    void standardize(std::map<std::string,long double *>& results, json * data){
        auto * means=Common::initializeArray((long double )0.0, FEATURES_LENGTH);
        for(int i=0; i<FEATURES_LENGTH; i++){
            for(auto& pair: results) {
                means[i] += pair.second[i];
            }
            means[i]/=results.size();
            data->operator[]("metadata")["mean"].push_back(means[i]);
        }
        auto * std_dev=Common::initializeArray((long double)0, FEATURES_LENGTH);
        for(int i=0; i < FEATURES_LENGTH; i++){
            for(auto& pair: results) {
                std_dev[i]+=pow(pair.second[i]-means[i],2);
            }
            std_dev[i]= sqrt(std_dev[i]/results.size());
            data->operator[]("metadata")["deviation"].push_back(std_dev[i]);
        }
        for(auto& pair:results){
            for(int i=0; i<FEATURES_LENGTH; i++)
                pair.second[i]=(pair.second[i]-means[i])/(std_dev[i]);
        }
        delete[] means;
        delete[] std_dev;
    }
    void minmax(std::map<std::string, long double *>& results,json * data){
        auto * featuresMax=Common::initializeArray((long double ) 0.0, FEATURES_LENGTH);
        auto * featuresMin=Common::initializeArray((long double) DBL_MAX, FEATURES_LENGTH);
        for(auto& pair:results){
            for (int j=0;j<FEATURES_LENGTH; j++){
                 long double f=pair.second[j];
                featuresMax[j]=featuresMax[j]<f?f:featuresMax[j];
                featuresMin[j]=featuresMin[j]>f?f:featuresMin[j];
            }
        }
        for(auto &pair:results){
            for(int j=0; j<FEATURES_LENGTH; j++)
                pair.second[j]= (pair.second[j]-featuresMin[j])/(featuresMax[j]-featuresMin[j]);
        }
        for(int i=0; i<FEATURES_LENGTH; i++){
            data->operator[]("metadata")["min"].push_back(featuresMin[i]);
            data->operator[]("metadata")["max"].push_back(featuresMax[i]);
        }
        delete[] featuresMin;
        delete[] featuresMax;
    }
    void normalize(std::map<std::string,long double *>& results, json* data){
        switch(normalization){
            case STANDARD:
                standardize(results, data);
                break;
            case MINMAX:
                minmax(results,data);
                break;
            case NONE:
                break;
        }
    }
    void train(const char * trainDir){
        json data;
        std:: map<std::string,long double *> results{};
        int idx=1;
        for(const auto &file: std::filesystem::directory_iterator(trainDir)){
            auto * image = new Image(file.path().c_str());
            printf("\n---%d. Learning: %s---\n", idx++, image->fileName.fileBaseName.c_str());
            auto * res=trainOne(image);
           results.insert_or_assign(image->fileName.fileBaseName, res);
           image->save(OUTPUT_DIR);
            delete[] image->data;
            delete image;
        }
        normalize(results, &data);
        for(auto& pair:results){
            std::cout<< pair.first << std::endl;
            for(int j=0;j<FEATURES_LENGTH; j++){
                std::cout << std::setprecision(15) << std::scientific << pair.second[j] << std::endl;
                data["labels"][pair.first].push_back(pair.second[j]);
            }
            delete[] pair.second;
        }
        std::ofstream out(MODEL_DATA_DIR.append(MODEL_DATA), std::ios::trunc);
        out << data.dump(2,' ');
        out.close();
        data.clear();
    }
    std::string findClosest( long double * features, nlohmann::json& data){
        long double closestValue=DBL_MAX;
        std::string label;
        json labels=data["labels"];
        for(auto i=labels.begin(); i!=labels.end(); ++i){
            json f=i.value();
            long double sum=0.0;
            unsigned int idx=0;
            for(auto j=f.begin(); j!=f.end(); ++j, idx++){
                double v=j->get<double>();
                if(std::isnan(features[idx])) printf("NaN: %Lf\n",features[idx]);
                long double val=pow(features[idx] - v, 2);
                sum+=val;
            }
            sum= sqrt(sum);
            if(sum<closestValue){
                closestValue=sum;
                label=i.key();
            }
        }
        return label;
    }
    void minmax(long double * features, json& data){
        json min=data["metadata"]["min"];
        json max=data["metadata"]["max"];
        unsigned int k=0;
        for(auto i=min.begin(), j=max.begin(); i!=min.end()&& j!=max.end(); ++i, ++j, k++){
            double m=i->get<double>();
            double ma=j->get<double>();
            features[k]=(features[k]- m)/(ma-m);
        }
    }
    void standardize(long double * features, json& data){
        json means=data["metadata"]["means"];
        json deviation=data["metadata"]["deviation"];
        unsigned int k=0;
        for(auto i=means.begin(), j=deviation.begin(); i!=means.end()&&j!=deviation.end(); ++i, ++j, k++){
            long double mean=i->get<long double>();
            long double std_dev=j->get<long double>();
            features[k]=(features[k]-mean)/std_dev;
        }
    }
    void normalize(long double * features, json& data){
        switch (normalization) {
            case MINMAX:
                minmax(features, data);
                break;
            case STANDARD:
                standardize(features, data);
                break;
            case NONE:
                break;
        }
    }
    Image * infer(Image * image){
        std::ifstream in1(MODEL_DATA_DIR.append(MODEL_DATA));
        nlohmann::json data=nlohmann::json::parse(in1);
        printf("%s\n", image->fileName.fileBaseName.c_str());
        preprocessor->polarize(image, Preprocessing::GAUSSIAN);
        unsigned int numObjects;
        auto* detector=new Detector(image);
        Common::ObjectLabel * objects= detector->detect( &numObjects);
        printf("--Started extracting--\n");
        long double ** featuresAll= FeatureExtractor::extractFeatures(objects, image, numObjects);
        auto * labels=new std::string[numObjects];
        for(unsigned int i=0; i<numObjects; i++){
            normalize(featuresAll[i], data);
            labels[i]= findClosest(featuresAll[i],data);
            delete[] featuresAll[i];
            //                        drawRect(objects[i], image,2);
        }
        delete[] featuresAll;
        cv::Mat cv_image(image->y, image->x, CV_8UC3, image->data);
        for(unsigned int i=0; i<numObjects; i++){
            cv::rectangle(cv_image, cv::Point(objects[i].x_start - 2, objects[i].y_start - 2), cv::Point(objects[i].x_end + 2, objects[i].y_end + 2), cv::Scalar_<unsigned char>(255, 0, 0), 5);
            cv::putText(cv_image, cv::String(labels[i]), cv::Point(objects[i].x_start + 15, objects[i].y_end + -10), cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar_<unsigned char>(255, 0, 0));
        }
        image->data=cv_image.data;
        printf("\n");
        delete[] labels;
        free(objects);
        delete detector;
        in1.close();
        data.clear();
        return image;
    }
}