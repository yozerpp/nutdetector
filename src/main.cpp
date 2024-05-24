 //
// Created by jonossar on 3/8/24.
//

#include <fstream>
#include "./include/main.h"
#include "./include/tclap/CmdLine.h"
#include "json.hpp"
#include "boost/program_options.hpp"
#include "common.h"

 static inline Model::Model* model;
 void signalHandler(int sig){
    Common::waitChildren();
    abort();
 }
 using namespace TCLAP;
 static bool isTraining=false;
 static bool isTesting=false;
 static double threshold=0.8;

static bool isLines=false;
bool init(const vector<Arg*>& args){
     signal(SIGSEGV,signalHandler);
     signal(SIGABRT,signalHandler);
     bool pca=false;
     bool clear=true;
     bool standardize=true;
     int dmpCount=3;

     vector<Model::DistanceMethod> storageMethods{};
     for(auto arg:args){
         auto& name=arg->getName();
         try {
             if (name == "show steps") {
                 SHOW_STEPS = ((ValueArg<bool> *) arg)->getValue();
             } else if(name=="lines"){
                 isLines=arg->isSet();
             } else if(name=="test"){
                 isTesting=arg->isSet();
             } else if(name=="train"){
                 isTraining=arg->isSet();
             }else if (name == "lines-dir") {
                 auto dir = dynamic_cast<ValueArg<string> *>(arg)->getValue();
                 if (!dir.empty())
                     Common::EDGE_DIR = dir;
             } else if (name == "train-dir") {
                 auto dir = dynamic_cast<ValueArg<string> *>(arg)->getValue();
                 if (!dir.empty())
                     Common::TRAINING_DIR = dir;
             } else if (name == "test-dir") {
                 auto dir = dynamic_cast<ValueArg<string> *>(arg)->getValue();
                 if (!dir.empty())
                     Common::TESTING_DIR = dir;
             } else if (name == "clear" && isTraining) {
                 clear = dynamic_cast<ValueArg<bool> *>(arg)->getValue();
             } else if (arg->getFlag() == "pca" && (isTesting || isTraining)) {
                 pca = dynamic_cast<ValueArg<bool> *>(arg)->getValue();
             } else if (arg->getFlag() == "-distance" && (isTesting || isTraining)) {
                 for (auto &s: *dynamic_cast<MultiArg<string> *>(arg)) {
                     storageMethods.push_back(Model::storageStringsRev.at(s));
                 }
             } else if (arg->getFlag() == "-dump_count") {
                 dmpCount = dynamic_cast<ValueArg<int> *>(arg)->getValue();
             } else if (arg->getFlag() == "-standardize") {
                 standardize = dynamic_cast<ValueArg<bool> *>(arg)->getValue();
             } else if (arg->getFlag() == "-threshold") {
                 threshold = dynamic_cast<ValueArg<double> *>(arg)->getValue();
             }
         } catch (ArgException& e){
             cerr  << e.error() << " for " << e.argId() << endl;
         }
     }
     if(!isTraining && !isTesting && !isLines){
         cout << "Usage: " <<endl;
         for(auto& arg: args)
         cout << arg->toString() << ": " << arg->getDescription(arg->isRequired())<<endl << endl;
         cout << "-h (--help)" << ": " << "read help"<<endl << endl;
     }

     if(clear) Model::DataManager::clear();
     model=new Model::Model(pca,dmpCount,storageMethods.size(),storageMethods.data(),standardize);
     return true;

 }
 vector<Arg*> getArgs(int argc, const char* argv[]){
     auto wd= std::filesystem::current_path().string();
     auto& cmd=*new CmdLine("Usage: ",' ');
     vector<Arg *> ret{};
     auto* trainSw=new SwitchArg("r","train","perform training. See other options for training parameters",false);
     auto* testSwitch=new SwitchArg("t", "test", "perform testing. See other options for testing parameters",false);
     auto* lineSwitch=new SwitchArg("l", "lines", "perform edge detection", false);
     auto* train=new ValueArg<string>("j", "train-dir", "the directory that contains training images.",false,Common::TRAINING_DIR,"string");
     auto* test=new ValueArg<string>("k", "testing-dir","run tests, you can cascade --train and --test arguments to perform both in one execution. You can specify the directory that contains test images.",false,Common::TESTING_DIR,"string");
     auto* clear=new ValueArg<bool>("c", "clear", "overwrite existing training data", false,false,"bool");
     auto* standardize=new ValueArg<bool>("s", "standardize", "standardize the data. Standardization gives better results if samples of the same class are very similar",false,true,"bool");
     auto* pca=new ValueArg<bool>("p", "pca", "perform principal analysis with training and testing, currently giving incorrect results.", false,
                        true,"bool");
     auto* thresholdArg=new ValueArg<double>("b", "threshold", "threshold for eigenvalues during pca",false,.8,"0.0:1.0");
     auto storageMethodArgs=vector<string>{"Covariance", "Mean", "Dump"};
     auto *constraints=new ValuesConstraint(storageMethodArgs);
     auto* output=new ValueArg<string>("o", "output-dir","location to save final result",false,Common::OUTPUT_DIR,"string");
     auto* step=new ValueArg<string>("z", "steps-output-dir","where the outputs of intermediate steps will be saved to",false,Common::STEP_DIR,"string");
     auto* distanceMethods=new MultiArg<string>("d", "distance-methods", "methods to determine closest class to a object",false,dynamic_cast<Constraint<string>*>(constraints));
     auto* showSteps=new ValueArg<bool>("v", "show-steps","specify if intermediary steps will be outputted.",false,true,"bool");
     auto* lines=new ValueArg<string>("i", "lines-dir", "detect lines, you can specify directory that contains images to be used.",false,Common::EDGE_DIR,"string");
     auto* closestTestInstances=new ValueArg<int>("x","test-case-count", "Parameter for dump classification, number of training objects to match before evaluating which class is closest.",false,3,"int" );
     cmd.add(train).add(test).add(showSteps).add(clear).add(pca).add(lines).add(distanceMethods).add(closestTestInstances).add(standardize).add(thresholdArg).add(output).add(step).add(trainSw).add(testSwitch).add(lineSwitch);
     ret.push_back(train);ret.push_back(test);ret.push_back(clear);ret.push_back(pca);ret.push_back(showSteps);ret.push_back(lines);ret.push_back(closestTestInstances); ret.push_back(distanceMethods);ret.push_back(standardize);ret.push_back(thresholdArg);ret.push_back(output);ret.push_back(step);ret.push_back(lineSwitch);ret.push_back(testSwitch);ret.push_back(trainSw);
     cmd.parse(argc, argv);
     return ret;
 }
 /** entry point
  * handles thread management and command line options
  * @param argc
  * @param argv
  * @return
  */

 void test(){
     for(const auto &file: std::filesystem::directory_iterator(Common::TESTING_DIR)){
         auto * image=new Image(file.path().c_str());
         printf("\n\n--testing: %s--\n\n", image->fileName.fileBaseName.c_str());
         auto** results=model->infer(image);
         for(int i=0; i< model->storageMethodsCount; i++){
             results[i]->save(Common::OUTPUT_DIR);
             delete[] results[i]->data;
             delete results[i];
         }
         delete[] results;
         delete[] image->data;
         delete image;
     }
 }

 void lines(){
     for(auto& f:std::filesystem::directory_iterator("./images/edge")){
         Image image(f.path().c_str());
         auto* out= Model::Model::detectLines(&image);
         image.save(Common::OUTPUT_DIR); out->save(Common::OUTPUT_DIR);
     }
 }
 static void train(){
     model->train(Common::TRAINING_DIR.c_str(),threshold);
 }
 int main(int argc, const char * argv[]){
     auto args= getArgs(argc, argv);
     init(args);
     if(isLines){
        lines();
     }else {
         if(isTraining) train();
         if(isTesting)test();
     }
     Common::waitChildren();
}