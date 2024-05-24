//
// Created by jonossar on 5/12/24.
//
#include "../include/Matrix.h"
#include "opencv2/opencv.hpp"
#include "../include/common.h"
#include "../include/Model.h"
#include "Eigen/Dense"
#include "thread"
#define IMG1_MATRIX_IMPLEMENTATION
/** acquires projection matrix from data @p in with @p threshold
 * this function first calculates the covariance matrix of the input matrix (see @p Matrix::covariance ) then acquires the eigenvectors and eigenvalues through the Eigen library. then it finds the eigen values that make up @p threshold amount of variance and filter out the rest.
 * @tparam T type of matrix
 * @param in input data
 * @param threshold threshold
 * @param standardize
 * @return projection matrix
 */
template <typename T>
Matrix<T> Matrix<T>::projection(const Matrix<T> &in, double threshold, bool standardize) {
    using Vector=Vector<T>;
    using Matrix=Matrix<T>;
    typedef Eigen::Matrix<T,FEATURES_LENGTH,FEATURES_LENGTH,Eigen::RowMajorBit> cov_t;
    int CVTYPE;
    if(typeid(T)== typeid(double)) CVTYPE=CV_64F;
    else if(typeid(T)== typeid(float))CVTYPE=CV_32F;
    auto covariance= in.covariance(standardize, true);
    cov_t cov (covariance.data);
    Eigen::SelfAdjointEigenSolver<cov_t> solver(cov);
    auto eval=solver.eigenvalues().real();
    auto * evalcpy=new T[covariance.x];
    for(int i=0; i<covariance.x; i++)
        evalcpy[i]=eval[covariance.x-i-1];
    Vector eigenVal(covariance.x,evalcpy);
    auto evec=solver.eigenvectors().real();
    auto* eveccpy=new T[covariance.x*covariance.y];
    for(int i=0; i< covariance.y; i++)
        for(int j=0; j<covariance.x; j++) eveccpy[i*covariance.x + j]=evec(i,covariance.x-1-j);
     Matrix eigenVec(covariance.x,covariance.y,eveccpy);
    int i=0;
    T sum=eigenVal.sum();
    T cumsum=(T)0;
    while(i< eigenVal.x && cumsum/sum < threshold)
        cumsum += eigenVal[i++];
    std::cout << "selected features count: " << i <<std::endl;
    eigenVec.crop(i);
    eigenVec.template normalize<NORMALIZATION_STRENGTH, axis::Y>();
    return eigenVec;
}
template Matrix<Model::dataType > Matrix<Model::dataType>::projection(const Matrix<Model::dataType> &in, double threshold, bool standardize);