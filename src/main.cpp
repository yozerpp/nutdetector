 //
// Created by jonossar on 3/8/24.
//

#include <fstream>
#include "./include/main.h"


#include "json.hpp"
#include "boost/program_options.hpp"
 void test(){
     for(const auto &file: std::filesystem::directory_iterator(TESTING_DIR)){
         auto * image=new Image(file.path().c_str());
         printf("\n\n--testing: %s--\n\n", image->fileName.fileBaseName.c_str());
          image=Model::infer(image);
          image->fileName.fileBaseName.append("_results");
          image->save(OUTPUT_DIR);
          delete[] image->data;
          delete image;
     }
 }

 bool contains(std::vector<int> v, int val){
    for(int i=0; i<v.size(); i++)
        if(val==v[i]) return true;
     return false;
}
 void testt(){
     auto * preprocessor=new Preprocessing::Preprocessor(2,256);
    for(auto &file: std::filesystem::directory_iterator(TRAINING_DIR)){
        auto * image=new Image(file.path().c_str());
        preprocessor->polarize(image, Preprocessing::GAUSSIAN);
//        image->save(OUTPUT_DIR);
        delete[] image->data;
        delete image;
    }
    delete preprocessor;
 }
 namespace opts=boost::program_options;
int main(int argc, const char * argv[]){
    const rlim_t stackSize=10000000000000000;
    struct rlimit l;
    getrlimit(RLIMIT_STACK, &l);
    l.rlim_cur=stackSize;
    if(setrlimit(RLIMIT_STACK, &l)!=0)
        return 1;
    opts::options_description options;
    options.add_options()("test", "run tests")("train", "run training");
    opts::variables_map args;
    opts::store(opts::parse_command_line(argc, argv, options), args);
    if(args.count("test")) test();
    if(args.count("train")) Model::train(TRAINING_DIR);
//    Model::train(TRAINING_DIR);
//    freopen("log", "w", stdout);
//testt();
//    return a;
//    auto * preProcessor=new Preprocessing::Preprocessor(2, 256);
//    for (const auto &file : std::filesystem::directory_iterator(INPUT_DIR)) {
//        auto * img=new Image(file.path().c_str());
//        img=preProcessor->polarize(img, Preprocessing::GAUSSIAN);
//        img->save(OUTPUT_DIR);
//        auto * labelLength=new unsigned int;
//        ObjectLabel* labels=Detector::Detect(img, labelLength);
//        for(int i=0; i<*labelLength; i++)
//            Extractor::extractFeatures(labels[i], img);
//        img->save(OUTPUT_DIR);
//    }
// void train(){
//     for (const auto &file : std::filesystem::directory_iterator(TRAINING_DIR)){
//         auto * sample=new Image(file.path().c_str());
//         printf("\n\n--training on: %s--\n\n", sample->fileName.fileBaseName.c_str());
//         Model::train(sample, sample->fileName.fileBaseName.substr(0, sample->fileName.fileBaseName.size()).c_str());
//         sample->fileName.fileBaseName.append("_trained");
//         sample->save(OUTPUT_DIR);
//         free(sample->data);
//         delete sample;
//     }
// }
////    auto * preprocessor=new Preprocessing::Preprocessor(2,256);
////     for (const auto &file : std::filesystem::directory_iterator(INPUT_DIR)) {
////     preprocessor->polarize(new Image(file.path().c_str()), Preprocessing::GAUSSIAN)->save(OUTPUT_DIR);
////     }
//YAML::Node a;
//nlohmann::json a;
//for(std::string str: {"a","b","c"})
//    for(int i=0; i<5; i+=16)
//    a[str].push_back(i);
}