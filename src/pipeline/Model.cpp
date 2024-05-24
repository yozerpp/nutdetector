//
// Created by jonossar on 3/21/24.
//

#include <filesystem>
#include <future>
#include "../include/Model.h"
#include "common.h"
#include "../include/Kernel.cuh"
namespace Model {
    using json=nlohmann::json;
    static inline int getRange(Image * img){
        return (int)ceil(hypot(img->x-1,img->y-1) ) +1;
    }
    /** this is a wrapper function for kernel call to houghLine (see @p HoughLine for implementation details)
     * @param img
     * @return
     */
    static inline unsigned int* getHough(Image* img){
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
    /** extract hough matrix from image
     * this function first acquires the raw hough matrix (see @p getHough ) then it filters top %1 percent of the hough matrix and converts matrix into a binary.
     * @param img a binary image with white edges
     * @return hough matrix
     */
    static inline unsigned char* houghLine(Image* img){
#define numMeans 3
        int range= getRange(img);
        auto* hough=getHough(img);
        unsigned int max=hough[0];
        for(int i=0; i<range*360; i++){
            if(hough[i] > max) max=hough[i];
        }
#define maxLines 30
        auto * sorted=Kernel::sort(hough,range*360);
        auto  threshold=sorted[ (int)(range*360 * (99/100.0))];
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
    /** this function attemps to map line given with polar coordinates to x-y plane but currently it doesn't appear to work correctly
     *
     * @param r
     * @param t
     * @param img
     * @return
     */
   static inline cv::Point * getLinePoints(const int r,const  int t, const Image& img){
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
    /** detect lines
     * this function first performs and edge detection on the image (see @p Preprocessing::edge ), then it extracts the hough matrix from the image (see @p houghLine ), then it draws every line in the hough matrix (h[rho][thta]) by first converting polar coordinates to x-y coordinates and extending the line to span the whole image (see @p getLinePoints )
     * @param img
     * @return image with lines drawn
     */
    Image* Model::detectLines(Image* img){
        Image& cpy=*new Image(*img);
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
        img->fileName=cpy.fileName;
        img->fileName.fileBaseName.append("_edge");
        return& cpy;
    }
    //TODO IMPLEMENT INCREMENTAL COVARIANCE CALCULATION
    //TODO NORMALIZATION ACROSS CLASSES NEEDED?
    /** extract features of one image containing only one class of images and return the results.
     * this function first performs a binary conversion on the image (see @p Preprocessing::binary ), then it detects the objects in the image (see @p Detector) and draws the bounding box encapsulating the image with opencv::rectangle function, then it extracts the features from the detected objects (see @p Extractor).
     * @param sample
     * @return matrix containing features of the objects in the @p sample
     */
    Matrix<dataType>& Model::trainOne(Image * sample){
        auto * original =new unsigned char[ sample->size()];
        memcpy(original,sample->data, sizeof(unsigned char)*sample->size());
        sample= Preprocessing::binary(sample);
        unsigned int count;
        Common::ObjectPosition* objects= Detector(sample).detect(&count);
        for(unsigned int i=0; i<count; i++) {
            cv::rectangle(cv::Mat(sample->y, sample->x, CV_8UC3, original), cv::Point(objects[i].x_start - 2, objects[i].y_start - 2), cv::Point(objects[i].x_end + 2, objects[i].y_end + 2), cv::Scalar_<unsigned char>(255, 0, 0), 5);
        }
        auto * ret= new Matrix<dataType>(Extractor::extractFeatures(objects, sample, count));

        delete[]sample->data;
        sample->data=original;
        free(objects);
        return *ret;
    }
    /** Perform principal component analysis and dimensionality reduction to input data
     * this function first acquires the projection matrix from the input matrix containing all the input data (see @p Matrix::projection ), then it multiplies (after centering) every matrix in the @p results by the projection matrix
     * @param total matrix containing all the training data
     * @param results map object containing the features grouped by classes
     * @param threshold threshold for eigenvalue filtering
     * @param standardize whether or not standardization will be performed
     * @return projection matrix
     */
    Matrix<dataType> Model::Model::doPCA(Matrix<dataType> &total, std::map<std::string, Matrix<dataType>> &results,
                                           double threshold,
                                           bool standardize) {
        auto projection= Matrix<dataType>::projection(total, threshold, standardize);
        for(auto& pair:results) {
            auto centered=pair.second.center();
            if(standardize) {
                auto stddevv = pair.second.stddev(true);
                centered /= stddevv;
            }
            auto projected = centered * projection;
            auto *dtCpy = new dataType[projected.x * projected.y];
            for (int i = 0; i < projected.x * projected.y; i++) dtCpy[i] = projected[i];
            delete[] pair.second.data;
            pair.second.x = projected.x;
            pair.second.data = dtCpy;
        }
        return projection;
    }
    template <int n>
        DataManager::Iterator<n>::Iterator(json& dat, std::string*&& dataLocations):  current(dat.begin()), endIt(dat.end()){
            for (int i=0; i<n; i++)
                labels[i]=std::move(dataLocations[i]);
        }

    template <int n>
        typename DataManager::Iterator<n>::labelMatPair DataManager::Iterator<n>::operator*(){
            labelMatPair ret;
            int k=0;
            for(auto& label:labels){
                auto tmp= current->at(label).template get<std::vector<std::vector<dataType>>>();
                int x=tmp[0].size();
                int y=tmp.size();
                ret.mats[k]=Matrix<dataType>(x, y);
                for(int i=0; i<y; i++) {
                    for (int j = 0; j < x; j++)
                        ret.mats[k].operator[](i * x + j) = tmp[i][j];
                }
                k++;
            }
            ret.label=current.key();
            return ret;
        }
        /** MeanDistanceCalculator
         * this class labels the image by measuring the euclidiean distance between the features of the input object and means of the classes in the training data.
         */
        struct MeanDistanceCalculater : public DataManager{
        Matrix<MinMax<dataType>> minmax = Matrix<MinMax<dataType>>(7,1);
        static inline Matrix<MinMax<dataType>> storeMean(const Matrix<dataType>** all, const int len){
            Matrix<dataType> means(all[0]->x,0);
            for(int i=0; i< len; i++){
                means.merge(all[i]->mean());
            }
            auto minmax=means.minMax(Matrix<dataType>::axis::Y);
            auto mm=Matrix<dataType>::mins(minmax);
            DataManager::store("min", data["metadata"], mm);
            auto mam=Matrix<dataType>::maxes(minmax);
            DataManager::store("max", data["metadata"], mam);
            return minmax;
        }
        inline void storeMetadata(const Matrix<dataType>** all, const int len)override{
            DataManager::storeMetadata(all, len);
            minmax= storeMean(all,len);
        }
        static inline Matrix<dataType> normalizeMinMax(const Matrix<dataType>& that,const Matrix<MinMax<dataType>>& minmax){
            auto mean =that.mean();
            auto mins=Matrix<dataType>::mins(minmax);
            auto maxes=Matrix<dataType>::maxes(minmax);
            return (mean - maxes)/(maxes- mins);
        }
        inline void store(std::string label,const Matrix<dataType>& that) override{
            auto normalized= normalizeMinMax(that,minmax);
            DataManager::store("meanNormalized", data["classes"][label], normalized);
        }
        inline DistanceMethod type()const override{
            return Mean;
        }
        MeanDistanceCalculater(): DataManager(){}
        inline std::string findClosest(const Matrix<dataType> &that)override {
            dataType  closestDist=FLT_MAX;
            auto min=DataManager::retrieveMetadata<Matrix<dataType>>("min");
            auto max=DataManager::retrieveMetadata<Matrix<dataType>>("max");
            minmax=Matrix<MinMax<dataType>>(min.x,1);
            for(int i=0; i<min.x; i++) minmax[i]=MinMax<dataType>{min[i],max[i]};
            auto normalized= normalizeMinMax(that,minmax);
            std::string label;
            for(auto pair:Iterator<1> (data["classes"], new std::string[1]{"meanNormalized"})){
                auto dist=Vector<dataType>(normalized - pair.mats[0]).length();
                if(dist< closestDist){
                    label=pair.label;
                    closestDist=dist;
                }
            }
            return label;
        }
    };
    /** CovarianceDistanceCalculator
     * this class labels the image by measuring the mahalanobis distance between the features of the input object and means of the classes in the training data.
     */
    struct CovarianceDistanceCalculater : public DataManager{
        CovarianceDistanceCalculater(): DataManager(){}
        inline void storeMetadata(const Matrix<dataType>** all, const int len)override{
            DataManager::storeMetadata(all, len);
        }
        inline void store(std::string label, const Matrix<dataType>& that) override{
            auto mean= that.mean();
            auto covariance= that.covariance(true);
            DataManager::store("mean", data["classes"][label], mean);
            DataManager::store("covariance", data["classes"][label], covariance);
        }
        inline DistanceMethod type()const override{
            return Covariance;
        }

        inline std::string findClosest(const Matrix<dataType> &that)override {
            dataType  closestDist=FLT_MAX;
            std::string label;
            auto min=DataManager::retrieveMetadata<Matrix<dataType>>("min");
            auto max=DataManager::retrieveMetadata<Matrix<dataType>>("max");
            for(auto pair:Iterator<2> (data["classes"], new std::string[2]{"mean", "covariance"})){ // TODO DO I NEED TO DELETE PAIR AFTER?
                auto& mean=pair.mats[0];
                auto& cov= pair.mats[1];
                auto dist=Matrix<dataType>::mahalanobis(that, mean ,cov);
                if(dist<closestDist) {
                    label=pair.label;
                    closestDist=dist;
                }
            }
            return label;
        }
    };
    /** DumpDistance calculator
     * this class is a performs labeling by finding @p n number of closest objects from the training data and assigning the _detect which is prominent among the found objects.
     */
    struct DumpStorageProvider: public DataManager{
    public:
        int count;
        DumpStorageProvider(): DataManager(), count(3){}
        inline void store(std::string label,const Matrix<dataType>& that)override{
            DataManager::store("raw", data["classes"][label], that);
        }
        inline DistanceMethod type()const override{
            return Dump;
        }
        inline void storeMetadata(const Matrix<dataType>** all,const int len)override{
            DataManager::storeMetadata(all, len);
        }
        inline std::string findClosest(const Matrix<dataType> &that)override{
            std::multimap<std::string, nullptr_t> counts{};
            for(int i=0;i < count; i++){
                std::string closest;
                dataType closestVal=FLT_MAX;
                for(auto pair: Iterator<1>(data["classes"], new std::string[1]{"raw"})){
                    auto tmp= Vector(that - pair.mats[0]);
                    auto dist= (dynamic_cast<Vector<dataType>*>(&tmp))->length();
                    if(dist < closestVal) closest = pair.label;
                }
                counts.insert(std::pair(closest,nullptr));
            }
            std::string closest;
            int matchCount=0;
            for(auto it=counts.begin(); it!=counts.end(); ++it){
                int c=counts.count(it->first);
                if(matchCount < c) {
                    matchCount=c;
                    closest=it->first;
                }
            }
            return closest;
        }
    };
    /** constructor
     *
     * @param PCA whether PCA will be performed or not (it currently gives incorrect results)
     * @param closestTest if Dump distance calculation (see @p DumpDistanceCalculator ) is chosen as a distance calculation method, program will find @p closestTest number of objects and assign the count of the whichever _detect is highest.
     * @param s number of distance calculation methods
     * @param storageMethods distance calculation methods
     * @param standardize whether or not standardization should be performed on during the principal component analysis
     */
    Model::Model(bool PCA, int closestTest, int s, const DistanceMethod* storageMethods, bool standardize): PCA(PCA), storageMethodsCount(s), standardize(standardize){
        storageProviders=new DataManager*[storageMethodsCount];
        for(int i=0; i<s; i++){
            switch (storageMethods[i]) {
                case DistanceMethod::Covariance:
                    storageProviders[i]=(DataManager*) new CovarianceDistanceCalculater;
                    break;
                case DistanceMethod::Mean:
                    storageProviders[i]=(DataManager*) new MeanDistanceCalculater();
                    break;
                case DistanceMethod::Dump:
                    auto* ss= new DumpStorageProvider();
                    ss->count= closestTest;
                    storageProviders[i]=(DataManager*)ss;
                    break;
            }
        }
    }
    /** trains with the dataset present in the directory specified.
     * this function performs following actions to every individual image in the input directory:
     * binary conversion (see @p Preprocessing::binary)
     * object detection (see @p Detector::detect)
     * feature extraction (see @p Extractor)
     * (if specified) then it joins every data in the dataset together(including different labels) and performs principal component analysis on them.
     * then it stores different variations of the data for different distance calculators (see @p Model::MeanDistanceCalculator, Model::CovarianceDistanceCalculator ) for example it stores mean if mean is specified as a distance calculation methods in the program arguments. It also stores the covariance if the covariance is chosen as a distance calculation method.
     * @param trainDir
     * @param threshold threshold to be used for dimension reduction during principal component analysis
     */
    void Model::train(const char * trainDir, double threshold){
        std::map<std::string, Matrix<dataType>>results{};
        Matrix<dataType> total(FEATURES_LENGTH, 0, nullptr);
        int idx=1;
        for(const auto &file: std::filesystem::directory_iterator(trainDir)){
            Image image(file.path().c_str());
            std::string name=image.fileName.fileBaseName;
            printf("\n---%d. Learning: %s---\n", idx++, image.fileName.fileBaseName.c_str());
            auto res=trainOne(&image);
            results.insert_or_assign(name,res);
            total.merge(res,false);
            image.fileName.fileBaseName=name;
            image.fileName.fileBaseName.append("_trained");
            image.save(Common::OUTPUT_DIR);
        }
        if(PCA){
            auto projection= doPCA(total, results, threshold, standardize);
            DataManager::storeMetadata("projection", projection);
        }
        Matrix<dataType>*ar [results.size()];
        int q=0;
        for (auto& pair:results){
            ar[q++]=&pair.second;
        }
        for(auto& pair:results){
            for(int i=0; i < storageMethodsCount; i++){
                storageProviders[i]->storeMetadata((const Matrix<dataType>**)(ar), results.size());
                storageProviders[i]->store(pair.first,pair.second);
            }
        }
        DataManager::flush();
    }
    /** performs labeling on input image
     * this function first converts the input image to binary (see @p Preprocessing::binary ) then it gathers positions of the objects (see @p Detector ),
     * then it extracts the features from the detected objects (see @p Extractor), then it (if specified) performs principal component analysis on the results, then it labels the object with various methods (see @p Model::MeanDistanceCalculator, Model::CovarianceDistanceCalculator ), and writes the _detect text onto the image using @p opencv::rectangle
     * @param image
     * @return
     */
    Image ** Model::infer(Image * image)const{
        printf("%s\n", image->fileName.fileBaseName.c_str());
        std::map<std::string, std::vector<std::string>> providerResults{};
        auto ** ret=new Image*[storageMethodsCount];
        for(int i=0; i < storageMethodsCount; i++) providerResults.insert_or_assign(storageStrings.at(storageProviders[i]->type()), std::vector<std::string>());
        auto name=image->fileName.fileBaseName;
        auto * original = new unsigned char[image->x*image->y* image->channels];
        for(int i=0; i<image->size(); i++) original[i]=image->data[i];
        Preprocessing::binary(image);
        unsigned int numObjects;
        auto* objects= Detector(image).detect(&numObjects);
        Matrix<dataType> all= Extractor::extractFeatures(objects, image, numObjects);
        if(PCA) {
            auto projectionMat = DataManager::retrieveMetadata<Matrix<dataType>>("projection");
            Matrix<dataType> tmp = all* projectionMat;
            all=std::move(tmp);
        }
        cv::Mat cv_images[storageMethodsCount];
        for(int i=0; i < storageMethodsCount; i++) {
            auto * cp= new unsigned char[image->x*image->y*image->channels];
            for(int j=0; j< image->x*image->y*image->channels; j++) cp[j]=original[j];
            cv_images[i]=cv::Mat(image->y,image->x, CV_8UC3, cp);
        }
        for(int j=0; j < storageMethodsCount; j++) {
            for(int i=0; i<numObjects; i++){
                std::string label=storageProviders[j]->findClosest(all.row(i).detach());
                cv::rectangle(cv_images[j], cv::Point(objects[i].x_start - 2, objects[i].y_start - 2),
                              cv::Point(objects[i].x_end + 2, objects[i].y_end + 2),
                              cv::Scalar_<unsigned char>(255, 0, 0), 5);
                cv::putText(cv_images[j], cv::String(label), cv::Point(objects[i].x_start + 15, objects[i].y_end + -10),
                            cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar_<unsigned char>(255, 0, 0));
                }
            auto * cpy=new unsigned char[cv_images[j].cols*cv_images[j].rows*cv_images[j].channels()];
            for(int k=0; k<cv_images[j].cols*cv_images[j].rows*cv_images[j].channels(); k++) cpy[k] =cv_images[j].data[k];
            ret[j]=new Image(cpy,cv_images[j].cols,cv_images[j].rows, 3, FileName((name + "_" + storageStrings.at(storageProviders[j]->type())),std::string(image->fileName.fileExt)));
        }
        printf("\n");
        delete[] original;
        free(objects);
        return ret;
    }
}