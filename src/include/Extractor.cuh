//
// Created by jonossar on 3/23/24.
//

#ifndef IMG1_FEATUREEXTRACTOR_CUH
#define IMG1_FEATUREEXTRACTOR_CUH
#ifndef IMG1_COMMON_H
#include "common.h"
#endif
#define MOMENT_MAX 3
namespace Extractor{
    class Moment123{
    private:
        cudaStream_t stream;
        double xNormal;
        double yNormal;
        double totalPixel;
        dim3 blocks;
        dim3 threads;
        __host__ void calculateNormals();
        __host__ void calculateCentrals();
    public:

        Common::ObjectLabel *label;
        Image *image;
        std::map<std::string, double > moments{};
        __host__ void calculate();
        __host__ Moment123(Common::ObjectLabel *label, Image *image);
        __host__ inline double getMoment(int p, int q) const{
            return moments.at(std::to_string(p)+ "," + std::to_string(q));
        }
    };
    __host__ long double calculateFeature(Moment123 * m, unsigned int i);
    __host__ long double * extractOne(Common::ObjectLabel * label, Image *image);
    __host__ long double ** extractFeatures(Common::ObjectLabel * objects, Image * image, unsigned int labelLength);
}
#endif //IMG1_MOMENT_CUH