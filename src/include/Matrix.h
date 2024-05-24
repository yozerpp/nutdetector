//
// Created by jonossar on 5/11/24.
//

#ifndef IMG1_MATRIX_H
#define IMG1_MATRIX_H

#include <utility>
#include <cmath>
#include <exception>
#include <iostream>
#include <thread>
#include <atomic>
struct IncompatibleDimensions : std::exception{
    const char* msg;
    IncompatibleDimensions() : msg("Matrices do not have matching dimensions."){}
    explicit IncompatibleDimensions(const char* msg):msg(msg){}
    [[nodiscard]] const char * what() const noexcept override{
        return msg;
    }
};
struct NotASquareMatrix : IncompatibleDimensions{
    NotASquareMatrix(): IncompatibleDimensions("Not a square matrix"){}
};
using namespace std;
template <typename T>
struct MatrixLike{
public:
    int x;
    int y;
    inline virtual T& operator[](int i)const=0;
   friend inline std::ostream& operator <<(std::ostream& out, const MatrixLike<T>& mat){
        for(int i=0; i< mat.y;i++){
            for(int j=0; j<mat.x; j++){
                out << mat.operator[](i*mat.x + j) <<(j!=mat.x?", ":"");
            }
            out << std::endl;
        }
        return out;
    }
    MatrixLike(int x,int y):x(x),y(y){}
    MatrixLike()=default;
};
template <typename T>
struct MinMax{
    T min;
    T max;
    MinMax():min(MAXFLOAT), max(-MAXFLOAT){}
    explicit MinMax(T min, T max):min(min),max(max){}
    explicit MinMax(int i):MinMax(){}
};
template <typename T>
struct Matrix :public MatrixLike<T>{
    enum axis{
        X,Y
    };
    Matrix() :MatrixLike<T>(0,0),data(nullptr){}

    T * data;
public:
    static inline bool SHOW_MATRIX_RESULTS=false;
    inline T & operator [](int i) const override{
        return std::forward<T&>(data[i]);
    }
    Matrix<T>& merge(Matrix<T>& other, bool discard=true){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(x!=other.x) throw IncompatibleDimensions();
        auto * ret= new T[x * (y+other.y)];
        int i;
        for(i=0; i<x*y;i++ )
            ret[i]=this->operator[](i);
        for(int j=0; j<other.x*other.y; j++)
            ret[i + j]=other[j];
        delete[] this->data;
        this->data=ret;
        this->y+=other.y;
        if(discard) delete &other;
        return std::forward<Matrix<T>&>(*this);
    }
    Matrix<T>& merge(Matrix<T>&& other){
       return merge(other,false);
    }
    /** calculate mahalanoibs distance from the following formula (x - m)t * C^-1 * (x-m)
     *
     * @param val
     * @param mean
     * @param cov
     * @return
     */
    static inline T mahalanobis(Matrix<T> val, Matrix<T> mean, const Matrix<T>& cov){
        if(mean.y==1) mean=mean.transpose();
        if(val.y==1) val=val.transpose();
        auto tmp= val-mean;
        auto singularMat=(tmp.transposeNew() *cov.inverse()*tmp);
        assert(singularMat.x==1 && singularMat.y==1);
        auto dist= sqrt (singularMat[0]);
        return dist;
    }
    static inline ostream& printOperator(ostream& s, const MatrixLike<T>& m1,const MatrixLike<T>& m2, const char* op){
        return cout << m1 << "      " << op << "       "<<endl << m2 << "      =       "  << endl;
    }
    static inline ostream& printOperator(ostream& s, const MatrixLike<T>& m1,const T& m2, const char* op){
        return cout << m1 << "      " << op << "       "<<endl << m2 << "      =       "  << endl;
    }
    inline Matrix<T> operator -(const T& val) const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        Matrix<T> ret(x,y);
        for(int i=0; i<x*y; i++){
            ret.operator[](i)=this->operator[](i)-val;
        }
        if(SHOW_MATRIX_RESULTS) {
            printOperator(cout, *this, val, "-")<<ret;
        }
        return ret;
    }
    inline Matrix<T> operator -(const MatrixLike<T>&other)const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        Matrix ret(x,y);
        if((other.y==1 && this->x==other.x) || (other.x==1 && this->y==other.y)){
            bool axis=other.x==1;
            for(int i=0; i<this->y; i++)
                for(int j=0; j<this->x; j++)
                    ret.operator[](i*x+j)=this->operator[](i*x+j)-other[axis?i:j];
        }
        else if (other.x==this->x && other.y==this->y) {
            for (int i = 0; i < x * y; i++)
                ret.operator[](i) = this->operator[](i)-other[i];
        } else throw IncompatibleDimensions();
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "-") << ret;
        return ret;
    }
    inline Matrix<T> operator +(const MatrixLike<T>&other)const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        Matrix ret(x,y);
        if((other.y==1 && this->x==other.x) || (other.x==1 && this->y==other.y)){
            bool axis=other.x==1;
            for(int i=0; i<this->y; i++)
                for(int j=0; j<this->x; j++)
                    ret.operator[](i*x+j)=this->operator[](i*x+j)+other[axis?(i%other.y):(j%other.x)];
        }
        else if (other.x==this->x && other.y==this->y) {

            for (int i = 0; i < x * y; i++)
                ret.operator[](i) = this->operator[](i)+other[i];
        } else throw IncompatibleDimensions();
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "+") << ret;
        return ret;
    }
    inline Matrix<T> operator +(const T& val) const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
         Matrix<T> ret(x,y);
        for(int i=0; i<x*y; i++){
            ret.operator[](i)=this->operator[](i)+val;
        }
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "+") << ret;
        return ret;
    }
    inline Matrix<T> valMul(const MatrixLike<T>& other)const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
       Matrix ret(x,y);
        if((other.y==1 && this->x==other.x) || (other.x==1 && this->y==other.y)){
            bool axis=other.x==1;
            for(int i=0; i<this->y; i++)
                for(int j=0; j<this->x; j++)
                    ret.operator[](i*x+j)=this->operator[](i*x+j)*other[axis?(i%other.y):(j%other.x)];
        }
        else if (other.x==this->x && other.y==this->y) {
            for (int i = 0; i < x * y; i++)
                ret.operator[](i) = this->operator[](i)*other[i];
        } else throw IncompatibleDimensions();
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "*(non-algebric)") << ret;
        return ret;
    }
    inline Matrix<T> operator *(const MatrixLike<T>&other)const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if (other.y!=this->x) throw IncompatibleDimensions();
         Matrix<T> ret(other.x,y);
        for(int i=0; i<y; i++){
            for(int j=0; j<other.x; j++){
                for(int k=0; k<x; k++)
                    ret.operator[](i*other.x+j)+= operator[](i*x+ k)*other[k*other.x+ j];
            }
        }
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "*") << ret;
        return ret;
    }
    inline Matrix<T> operator *(const T& val) const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
         Matrix<T> ret(x,y);
        for(int i=0; i<x*y; i++){
            ret.operator[](i)=this->operator[](i)*val;
        }
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "*(non-algebric)") << ret;
        return ret;
    }
    inline Matrix<T> operator /(const MatrixLike<T>&other)const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
         Matrix ret(x,y);
        if((other.y==1 && this->x==other.x) || (other.x==1 && this->y==other.y)){
            bool axis=other.x==1;
            for(int i=0; i<this->y; i++)
                for(int j=0; j<this->x; j++)
                    ret.operator[](i*x+j)=this->operator[](i*x+j)/other[axis?(i%other.y):(j%other.x)];
        }
        else if (other.x==this->x && other.y==this->y) {
            for (int i = 0; i < x * y; i++)
                ret.operator[](i) = this->operator[](i)/other[i];
        } else throw IncompatibleDimensions();
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "/") << ret;
        return ret;
    }
    inline Matrix<T> operator /(const T& val)const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        Matrix<T> ret(x,y);
        for(int i=0; i<x*y; i++){
            ret.operator[](i)=this->operator[](i)/val;
        }
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "/") << ret;
        return ret;
    }
    inline Matrix<T> operator ^(const T& val)const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        Matrix<T> ret(x,y);
        for(int i=0; i<x*y; i++){
            ret.operator[](i)= pow(this->operator[](i),val);
        }
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "^") << ret;
        return ret;
    }
    inline Matrix<T>& operator -=(const MatrixLike<T>&other){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "-=");
        if((other.y==1 && this->x==other.x) || (other.x==1 && this->y==other.y)){
            bool axis=other.x==1;
            for(int i=0; i<this->y; i++)
                for(int j=0; j<this->x; j++)
                    this->operator[](i*x+j)-=other[axis?i:j];
        }
        else if (other.x==this->x && other.y==this->y) {
            for (int i = 0; i < x * y; i++)
                this->operator[](i) -= other[i];
        } else throw IncompatibleDimensions();
        if (SHOW_MATRIX_RESULTS) cout  << *this;
        return *this;
    }
    inline Matrix<T>& operator -=(const T& val){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "-=");
        for(int i=0; i<x*y; i++){
            this->operator[](i)-=val;
        }
        if(SHOW_MATRIX_RESULTS) cout <<*this;
        return *this;
    }
    inline Matrix<T>& operator +=(const MatrixLike<T>&other){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "+=");
        if((other.y==1 && this->x==other.x) || (other.x==1 && this->y==other.y)){
            bool axis=other.x==1;
            for(int i=0; i<this->y; i++)
                for(int j=0; j<this->x; j++)
                    this->operator[](i*x+j)+=other[axis?(i):(j)];
        }
        else if (other.x==this->x && other.y==this->y) {
            for (int i = 0; i < x * y; i++)
                this->operator[](i) += other[i];
        } else throw IncompatibleDimensions();
        if(SHOW_MATRIX_RESULTS) cout << *this;
        return *this;
    }
    inline Matrix<T>& operator +=(const T& val){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "+=");
        for(int i=0; i<x*y; i++){
            this->operator[](i)+=val;
        }
        if (SHOW_MATRIX_RESULTS)  cout << *this;
        return *this;
    }
    inline Matrix<T>& operator *=(const T& val){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "-=");
        for(int i=0; i<x*y; i++){
            this->operator[](i)*=val;
        }
        if (SHOW_MATRIX_RESULTS)  cout<< *this;
        return *this;
    }
    inline Matrix<T>& operator /=(const MatrixLike<T>&other){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, other, "/=");
        if((other.y==1 && this->x==other.x) || (other.x==1 && this->y==other.y)){
            bool axis=other.x==1;
            for(int i=0; i<this->y; i++)
                for(int j=0; j<this->x; j++)
                    this->operator[](i*x+j)/=other[axis?(i):(j)];
        }
        else if (other.x==this->x && other.y==this->y) {
            for (int i = 0; i < x * y; i++)
                this->operator[](i) /= other[i];
        } else throw IncompatibleDimensions();
        if (SHOW_MATRIX_RESULTS) cout <<*this;
        return *this;
    }
    inline Matrix<T>& operator /=(const T& val){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        if(SHOW_MATRIX_RESULTS) printOperator(cout, *this, val, "/=");
        for(int i=0; i<x*y; i++){
            this->operator[](i)/=val;
        }
        if(SHOW_MATRIX_RESULTS) cout << *this;
        return *this;
    }

#define NORMALIZATION_STRENGTH 0
#define NORMALIZATION_MINMAX 1
template <int method=NORMALIZATION_STRENGTH, axis axis=Y>
    inline Matrix<T> &normalize() {
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        for(int i=0; i<(axis?x:y); i++){
            if constexpr (method==NORMALIZATION_STRENGTH) {
                T norm = (T) 0;
                for (int j = 0; j < (axis ? y : x); j++) {
                    norm += pow(this->operator[]((axis) ? (j * x + i) : (i * x + j)), 2);
                }
                norm = sqrt(norm);
                for (int j = 0; j < (axis ? y : x); j++) {
                    this->operator[]((axis) ? (j * x + i) : (i * x + j)) /= norm;
                }
            } else {
                T min=(T)MAXFLOAT;
                T max=(T)-MAXFLOAT;
                for (int j = 0; j < (axis ? y : x); j++) {
                    auto val=this->operator[]((axis) ? (j * x + i) : (i * x + j));
                    if(val >max) max=val;
                    else if(val < min) min=val;
                }
                for (int j = 0; j < (axis ? y : x); j++) {
                    auto& val=this->operator[]((axis) ? (j * x + i) : (i * x + j));
                    val= (val - min)/(max-min);
                }
            }
        }
        return std::forward<Matrix<T>&>(*this);
    }

    class RowReference :public MatrixLike<T>{
    public:
        int row;
        bool detached=false;
        RowReference(const Matrix<T>& target,int r): MatrixLike<T>(target.x,1),target(target),row(r){}
        inline T& operator [](int i) const override{
            return std::forward<T&>(target[(detached?0:row)*this->x + i]);
        }
        inline void operator ++(){
            ++row;
        }
        inline Matrix<T> detach(){
            Matrix<T>dat(target.x,1);
            for(int i=0; i<target.x; i++) dat.operator[](i)=target.data[row*target.x + i];
            target=dat;
            detached=true;
            return dat;
        }
        ~RowReference()= default;
        inline bool operator !=(const RowReference& other)const{
            return row!=other.row;
        }
    private:
        Matrix<T> target;
    };
    /** access a row of the matrix
     *
     * @param i
     * @return
     */
    RowReference row(int i)const;
    /** calculate mean of this matrix on the y axis.
     *
     * @return
     */
    inline Matrix<T> mean()const;
    /** calculate standard deviation of this matrix on the y axis
     *
     * @param population whether matrix represents the whole population or only the portion of it.
     * @return
     */
    inline Matrix<T> stddev(bool population) const;
    /** standardize the matrix.
     * it first centers the matrix and
     * @param population
     * @return
     */
     // TODO STRANDARDIZATION IS WRONG
//    inline Matrix<T> &standardize(bool population) {
//        auto center=this->center();
//        this->operator-=(center);
//        auto pow2=this->operator^(2);
//        auto stddevv= this->stddev(population);
//        this->operator/=(stddevv);
//        return std::forward<Matrix<T>&>(*this);
//    }

#define DET_DIR_POS 0
#define DET_DIR_NEG 1
private:
    /**
     * calculate determinant with a loop
     * @tparam direction
     * @param out
     */
template <int direction>
     inline void _determinant(T * out)const{
    static_assert(direction==1 || direction==0, "direction should be zero or one");
        if(this->x!=this->x) throw IncompatibleDimensions("Not a 2x2 matrix");
        *out=0;
        std::thread ts[this->y];
        volatile T total;
        for(int i=0; i<this->y; i++){
            ts[i]=std::thread([this,i,&total](){
                T subTotal=1;
                for(int j=0;j< this->x; j++){
                    int idx_x;
                    if constexpr (direction==DET_DIR_POS) idx_x=j;
                    else idx_x=this->x-j;
                    subTotal*=this->operator[](((i+j)%this->y)*this->x + idx_x);
                }
                if constexpr (direction==DET_DIR_POS) total+=subTotal;
                else total -=subTotal;
            });
        }
        for(auto& t:ts){
            if(t.joinable())t.join();
        }
        *out=total;
    }
    /** calculates determinant of the matrix on two directions.
     *
     * @return
     */
    inline T determinant()const{
        T outs[2]={0,0};
        for(int i=0; i<2; i++){
            if(i==0)
            this->_determinant<DET_DIR_POS>((T*)&outs[i]);
            else this->_determinant<DET_DIR_NEG>((T*)&outs[i]);
        }
        T total=0;
        for(int i=0; i<2; i++) if(i==DET_DIR_POS) total+=outs[i]; else total-=outs[i];
        return total;
    }
public:
    /** calculates minors of the matrix.
     *
     * @return
     */
    Matrix<T> minors()const{
        if(this->y!=this->x) throw NotASquareMatrix();
        Matrix ret(this->x, this->y);
        int instances=0;
        unsigned int maxThreads=std::thread::hardware_concurrency()>1?std::thread::hardware_concurrency()-1:1;
        std::thread crunchers[maxThreads];
        for(int i=0; i<this->y; i++){
            for(int j=0; j<this->x; j++){
                if(instances==maxThreads) {
                    for(auto& t:crunchers) if(t.joinable())t.join();
                    instances=0;
                }
                crunchers[instances++]=std::thread([this,i=i,j=j,&ret]()->void {
                    Matrix<T> mat(this->x - 1, this->y - 1);
                    int idx = 0;
                    for (int k = 0; k < this->y; k++) {
                        if (k == i) continue;
                        for (int q = 0; q < this->x; q++) {
                            if (q == j) continue;
                            mat[idx++] = this->operator[](k * (this->x) + q);
                        }
                    }
                    ret.operator[](i * this->x + j) = mat.determinant();
                });
            }
        }
        for(auto& t:crunchers)if(t.joinable()) t.join();
        return ret;
    }
    inline Matrix<T> cofactors()const{
        if(this->y!=this->x) throw NotASquareMatrix();
        auto minors=this->minors();
        for(int i=0; i< minors.y; i++){
            for(int j=0; j< minors.x; j++){
                minors[i*minors.x + j]*= pow<T>(-1, i+j);
            }
        }
        return minors;
    }
    Matrix<MinMax<T>> minMax(axis axis)const {
        Matrix<MinMax<T>> ret(axis==Y?this->x:1,axis==Y?1:this->y);
        for (int i = 0; i < (axis==Y ? this->x : this->y); i++) {
            MinMax<T> m;
                for (int j = 0; j < (axis==Y ? this->y : this->x); j++) {
                   auto val= this->operator[]((axis) ? (j * this->x + i) : (i * this->x + j));
                   if(val>m.max) m.max=val;
                   else if(val<m.min) m.min=val;
                }
                ret.operator[](i)=m;
        }
        return ret;
    }
//    inline T determinant()const{
//        if (this->x!=this->y) throw NotASquareMatrix();
//        if(this->x==2) return this->operator[](0) * this->operator[](3) - (this->operator[](1) *this->operator[](2));
//        auto& cofactors=this->cofactors();
//        T ret=(T)0;
//        for(int i=0; i<this->x; i++){
//            ret+=this->operator[](i)*cofactors[i];
//        }
//        return ret;
//    }
/**
 * calulates the inverse of the matrix from the following formula: @p |A|^-1 * @p Adj(A)
 * @return
 */
    inline Matrix<T>inverse()const{
        auto det=this->determinant();
        auto cofactors=this->cofactors();
        return cofactors.transpose() / det;
    }
    /**
     * centers the matrix by substracting by the following formula @p In - @p Jn/ @p(n-1) where In is an identity matrix and Jn is a matrix of all 1s.
     * @return
     */
    inline Matrix<T> center()const {
        auto identity=Matrix<T>::identity(this->y);
        auto ones=Matrix<T>::ones(this->y);
        auto centering=identity - (ones/this->y);
        return centering*(*this);
    }
    /** calculate covariance of the matrix from the follwing formula @p XcT* @p Xc / @p (n-1) where @p Xc is centered(and possibly standardized) matrix and @p XcT is it's transpose
     *
     * @param standardize
     * @param population
     * @return
     */
    inline Matrix<T>covariance(bool standardize = true, bool population=true) const{
//        Matrix<T>& mean=this->mean();
//        Matrix<T>& subtracted=this->operator-(mean);
//        delete&mean;
//        auto& t=subtracted.transposeNew();
//        auto& ret=t*subtracted;
//        delete &t;
//        delete &subtracted;
        auto centered=this->center();
        if(standardize){
            auto stddevv= stddev(population);
            centered/=stddevv;
        }
        auto t=centered.transposeNew();
        return (t*centered)/(this->y-1);
//        auto * ret=new Matrix<T>(x,x);
//        for(int i=0; i<y; i++){
//            Matrix<T>& row=this->row(i);
//            for(int j=0; j<x; j++)
//                for(int k=0; k<x; k++)
//                    ret->operator[](j*x+k)+=(row[j]-mean[j])*(row[k]-mean[k]);
//        }
//        ret->operator/=((T)y);
//        return std::forward<Matrix<T>&>(*ret);
    }
    /** transpose the matrix so that any point x,y is y,x is after transformation.
     *
     * @return
     */
    inline Matrix<T>& transpose(){
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
        auto * t=new T[y*x];
        for (int i = 0; i < y; i++)
            for (int j = 0; j < x; j++) {
                t[j * y + i] = this->operator[](i * x + j);
            }
        delete[] this->data;
        this->data=t;
        int tmp=this->y; this->y=this->x; this->x=tmp;
        return std::forward<Matrix<T>&>(*this);
    }
    inline Matrix<T> transposeNew()const{
       const int& x=MatrixLike<T>::x;
       const int& y=MatrixLike<T>::y;
            Matrix<T> ret(y, x);
            for (int i = 0; i < y; i++)
                for (int j = 0; j < x; j++) {
                    ret.operator[](j * y + i) = operator[](i * x + j);
                }
            if(SHOW_MATRIX_RESULTS) cout << *this << "    t     " << endl << ret;
            return ret;
    }
    /** projection, see the documentation in @p Matrix.cpp file
     *
     * @param in
     * @param threshold
     * @param standardize
     * @return
     */
    static Matrix<T>projection(const Matrix<T> &in, double threshold, bool standardize);
    Matrix<T>& crop( int s1=0, int s2=0){
        s1=s1>0?s1:this->x;s2=s2>0?s2:this->y;
        auto * newData=new T[s1*s2];
        for(int i=0; i<s2; i++)
            for(int j=0; j<s1; j++)
                newData[i*s1 + j]= operator[](i*this->x + j);
        this->x=s1;this->y=s2;
        delete[] this->data;this->data=newData;
        return std::forward<Matrix<T>&>(*this);
    }
    Matrix(const Matrix&& other) noexcept : MatrixLike<T>(other.x, other.y), data(other.data){
    }
    Matrix& operator=(Matrix&& other) noexcept{
        this->x=other.x;
        this->y=other.y;
        if(this->data!=nullptr)
            delete[]this->data;
        this->data=other.data;
        other.data=nullptr;
        return *this;
    }
    bool operator !=(const Matrix& other) const {
        if( this->x!=other.x || this->y!=other.y) return true;
        if (this-> data==other.data) return false;
        for(int i=0; i<this->x*this->y; i++) if(this->operator[](i) != other[i]) return true;
        return false;
    }
    bool operator ==(const Matrix& other) const{
        return !this->operator!=(other);
    }
    Matrix& operator=(const MatrixLike<T>& other){
        if(this->operator==(other)) return *this;
        this->x=other.x;
        this->y=other.y;
        delete [] this->data;
        this->data=new T[this->y*this->x];
        for(int i=0; i< other.x*other.y; i++) data[i]= other[i];
        return *this;
    }
    Matrix& operator=(const Matrix<T>& other){
       return operator=(std::forward<const MatrixLike<T>>(other));
    }
    Matrix(const MatrixLike<T>& other):MatrixLike<T>(other.x,other.y){
        data=new T[other.x*other.y];
        for(int i=0; i<other.x*other.y; i++) data[i]=other[i];
    }
    Matrix(const Matrix<T>& other):Matrix(std::forward<const MatrixLike<T>&>(other)){}
    Matrix(Matrix&& other)noexcept: MatrixLike<T>(other.x, other.y), data(other.data){
        other.data= nullptr;
    }
    Matrix(int x,int y): MatrixLike<T>(x,y){
        data=new T[x*y];
        for(int i=0; i< x*y; i++) data[i]=(T)0;
    }
    Matrix(int x, int y, const std::initializer_list<T>&& data): MatrixLike<T>(x,y){
        this->data=new T[x*y];
        for(int i=0; i< x*y; i++) this->data[i]=std::data(data)[i];
    }
    ~Matrix(){
        delete[]data;
    }
    static inline Matrix<T> mins(const MatrixLike<MinMax<T>>& mins){
        Matrix<T> ret(mins.x,mins.y);
        for(int i=0; i<mins.x*mins.y; i++) ret.operator[](i)=mins[i].min;
        return ret;
    }
    static inline Matrix<T> maxes(const MatrixLike<MinMax<T>>& mins){
        Matrix<T> ret(mins.x,mins.y);
        for(int i=0; i<mins.x*mins.y; i++) ret.operator[](i)=mins[i].max;
        return ret;
    }
    static inline Matrix<T> identity(int n){
         Matrix<T> ret(n,n);
        for(int i=0; i< n*n; i++){
            if(i%(n-1)==0)
                ret.operator[](i)=1;
            else ret.operator[](i)=0;
        }
        return ret;
    }
    static inline Matrix<T> ones(int n){
        Matrix<T> ret(n,n);
        for(int i=0; i< n*n; i++) ret.operator[](i)=1;
        return ret;
    }
    explicit Matrix(int x,int y,T * data): MatrixLike<T>(x,y),data(data){}
};
template <typename T>
struct Vector:public Matrix<T>{
    using axis=enum Matrix<T>::axis;
    axis ax;
    Vector(const MatrixLike<T>& other): ax(other.x==1?axis::Y:axis::X), Matrix<T>(std::forward<const Matrix<T>&>(other)){
    }
    Vector(const Matrix<T>& other): ax(other.x==1?axis::Y:axis::X), Matrix<T>(std::forward<const MatrixLike<T>&>(other)){
    }
    Vector(Matrix<T>&& other) noexcept:ax(other.x==1?axis::Y:axis::X), Matrix<T>(std::forward<Matrix<T>&&>(other)){}
    Vector(int x, axis ax=axis::X): Matrix<T>(ax ? 1 : x, ax ? x : 1), ax(ax){}
    Vector(int x, T* data, axis ax=axis::X): Matrix<T>(ax==axis::Y ? 1 : x, ax==axis::Y ? x : 1, data), ax(ax){}
    Vector(int x, const std::initializer_list<T>&& data, axis ax=axis::X): Matrix<T>(ax==axis::Y ? 1 : x, ax==axis::Y ? x : 1), ax(ax){
        this->data=new T[x*1];
        for(int i=0; i< x*1; i++) this->data[i]=std::data(data)[i];
    }

    inline T& operator[](int i) const override{return Matrix<T>::operator[](i);}
    Vector& cumsum()const{
        auto* ret=new Vector(ax ? this->y : this->x, this->y != 1);
        for(int i=0; i<(ax ? this->y : this->x); i++){
            ret->operator[](i)= this->operator[](i)+(i>0?ret->operator[](i-1):0.0);
        }
        return std::forward<Vector&>(*ret);
    }
    T sum()const{
        T ret=(T)0;
        for(int i=0; i<this->x; i++)
            ret+= this->operator[](i);
        return ret;
     }
     /**
      * calculate euclidian length of this matrix
      * @return
      */
    inline T length() const{
        const int& x=MatrixLike<T>::x;
        const int& y=MatrixLike<T>::y;
        T len=0;
        for(int i=0; i<(this->ax ? y : x); i++)
            len += pow(this->operator[](i),2);
        len= sqrt(len);
        return len;
    }
};
template <typename T>
typename Matrix<T>::RowReference Matrix<T>::row(int i)const{
    return RowReference(*this,i);;
}
template <typename T>
inline Matrix<T> Matrix<T>:: mean()const{
    const int& x=MatrixLike<T>::x;
    const int& y=MatrixLike<T>::y;
    Vector<T> ret(x);
    for(int i=0; i<x;i++){
        for(int j=0; j<y;j++)
            ret.operator[](i)+=this->operator[](j*x+i);
        ret.operator[](i)/=(T)y;
    }
    return ret;
}
template <typename T>
inline Matrix<T> Matrix<T>::stddev(bool population) const{
    const int& x=MatrixLike<T>::x;
    const int& y=MatrixLike<T>::y;
    auto centered=this->center();
    auto ret=centered^2.0;
    ret/=(this->y + (population?0:-1));
    return ret;
}
#endif //MODEL_JSON_MATRIX_H
