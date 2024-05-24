//
// Created by jonossar on 3/21/24.
//

#ifndef IMG1_MODEL_H
#define IMG1_MODEL_H

#include "json.hpp"
#include <fstream>
#include "opencv2/opencv.hpp"
#include "common.h"
#include "Preprocessor.h"
#include "Detector.h"
#include "Extractor.cuh"

#define MODEL_DATA_DIR std::string("./model/")
#define MODEL_DATA std::string("model.json")
#define MODEL_DATA_DUMP std::string("dump.json")
//#define OUTPUT_DIR Common::OUTPUT_DIR
//#define TRAINING_DIR Common::TRAINING_DIR
//#define TESTING_DIR Common::TESTING_DIR
namespace Model {
    using dataType=double;
    enum DistanceMethod {
        Covariance,
        Mean,
        Dump
    };
    inline std::map<DistanceMethod, const char*> storageStrings({{Covariance, "covariance"}, {Mean, "mean"}, {Dump, "dump"}});
    inline std::map <std::string, DistanceMethod> storageStringsRev{{"covariance", Covariance}, {"mean", Mean}, {"dump", Dump}};
    struct DataManager{
        using json=nlohmann::json;
    protected:
        static inline nlohmann::json data;
        template <int n>
        struct Iterator{
            struct labelMatPair{
                std::string label;
                Matrix<Model::dataType> mats[n]{};
            };
            json::iterator current;
            const json::iterator endIt;
            std::string labels[n];
            Iterator(json& dat, std::string*&& dataLocations);
            inline labelMatPair operator*();
            inline void operator++(){current ++;}
            inline Iterator begin(){return *this;}
            inline Iterator end(){return *this;}
            inline bool operator!=(const Iterator& thiss)const{return this->current!=thiss.endIt;}
        };
    public:
        DataManager()=default;
        static inline void init(){
            if(auto p=std::filesystem::path{MODEL_DATA_DIR.append(MODEL_DATA)};std::filesystem::exists(p)){
                DataManager::data=json::parse(std::ifstream(p));
            }
        }
        static inline void clear(){
            DataManager::data.clear();
        }
        static inline void flush(){
            std::ofstream out(MODEL_DATA_DIR.append(MODEL_DATA));
            out << data.dump(2, ' ');
            out.close();
        }
        virtual inline DistanceMethod type() const=0;
        virtual inline void storeMetadata(const Matrix<dataType>** all, const int len){
            int count=0;
            for(int i=0; i< len; i++){
                count+=all[i]->y;
            }
            store("count",data["metadata"],count);
        };
        virtual inline void store(std::string label,const Matrix<dataType>& that)=0;

        virtual inline std::string findClosest(const Matrix<Model::dataType> &that)=0;
        template <typename T>
        static inline typename std::enable_if<std::is_same<T,Matrix<Model::dataType>>::value, void>::type store(const std::string&& key,json& node,const T& that ){
            if (!node.contains(key)) {
                for (int i = 0; i < that.y; i++) {
                    auto ar = json::array();
                    for (int j = 0; j < that.x; j++) {
                        ar.push_back(that[i * that.x + j]);
                    }
                    node[key].push_back(ar);
                }
            }
        }
        template <typename T>
        static inline typename std::enable_if<std::is_arithmetic<T>::value,void>::type store(const std::string&& key,json& node, const T& that ){
            node[key]=that;
        }
        template<typename T>
        static inline void storeMetadata(const std::string&& key, T& that){
            store(std::move(key),data["metadata"],that);
        }
        template <typename T>
        static inline typename std::enable_if<std::is_same<T,Matrix<Model::dataType>>::value,Matrix<Model::dataType>>::type retrieveMetadata(const std::string&& key){
                auto vec=data["metadata"][key].get<std::vector<std::vector<dataType>>>();
                Matrix<dataType> ret(vec[0].size(), vec.size());
                for(int i=0; i<vec.size(); i++){
                    for(int j=0; j< vec[0].size();j++){
                        ret.operator[](i*vec[0].size()+ j)=vec[i][j];
                    }
                }
                return ret;
        }

        template <typename T>
        static inline typename std::enable_if<std::is_same<T,dataType>::value,T&>::type retrieveMetadata(const std::string&& key){
            auto * v=new T (data["metadata"][key].get<dataType>());
            return std::forward<T&>(*v);
        }
    };
    class Model {
    public:
        Model(bool PCA, int closestTest, int s, const DistanceMethod* storageMethods, bool standardize=true);
        Model(bool PCA, int storageMethodsLen, const DistanceMethod* storageMethods, bool standardize=true): Model(PCA, 3, storageMethodsLen, storageMethods, standardize){}
        static Image *detectLines(Image *img);
        void train(const char *trainDir, double threshold=0.8);
        Image ** infer(Image *image)const;
        const int storageMethodsCount;
    private:
        const bool PCA;
        bool standardize;
        static Matrix<dataType>
        doPCA(Matrix<dataType> &total, std::map<std::string, Matrix<dataType>> &results, double threshold,
              bool standardize);
        static Matrix<dataType>& trainOne(Image *sample);
    private:
        DataManager** storageProviders;
    };
}


#endif //IMG1_MODEL_H
