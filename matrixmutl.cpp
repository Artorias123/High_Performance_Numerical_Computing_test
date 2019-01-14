#include <iostream>
#include <cstdlib>
#include <emmintrin.h>
#include <pmmintrin.h>
using namespace std;
class Matrix{
    private:
    double **_Matrix;
    size_t _Row,_Column;
    public:
    Matrix():_Matrix(nullptr),_Row(0),_Column(0){}
    Matrix(size_t r,size_t c):_Row(r),_Column(c){
        if(!_Column||!_Row) return;
        _Matrix=(double**)malloc(_Column*sizeof(double*));
        double **p=_Matrix,**end=_Matrix+_Column;
        do{
            *(p++)=(double*)malloc(_Row*sizeof(double));
        }while(p!=end);
    }
    Matrix(size_t r,size_t c,const double init):_Row(r),_Column(c){
        if(!_Column||!_Row) return;
        _Matrix=(double**)malloc(_Column*sizeof(double*));
        double **pr=_Matrix,**endr=_Matrix+_Column,*p,*end;
        do{
            p=*(pr++)=(double*)malloc(_Row*sizeof(double));
            end=p+_Row;
            do{
                *(p++)=init;
            }while(p!=end);
        }while(pr!=endr);
    }
    Matrix(const Matrix& B){
        _Row=B._Row;
        _Column=B._Column;
        _Matrix=(double**)malloc(_Column*sizeof(double*));
        double **pbr=B._Matrix,**endbr=B._Matrix+_Column,*pb,*endb,
               **par=_Matrix,**endar=_Matrix+_Column,*pa,*enda;
        do{
            pa=*(par++)=(double*)malloc(_Row*sizeof(double));
            enda=pa+_Row;
            pb=*(pbr++);
            endb=pb+_Row;
            do{
                *(pa++)=*(pb++);
            }while(pa!=enda);
        }while(par!=endar);
    }
    ~Matrix(){
        if(!_Matrix) return;
        double **p=_Matrix,**end=_Matrix+_Column;
        do{
            free(*(p++));
        }while(p!=end);
        _Column=_Row=0;
        free(_Matrix);
    }
    double& operator()(size_t i,size_t j){return _Matrix[j][i];}
    const double operator()(size_t i,size_t j)const{return _Matrix[j][i];}
    Matrix& operator=(Matrix&& B){
        if(_Matrix){
            double **p=_Matrix,**end=_Matrix+_Row;
            do{
                free(*(p++));
            }while(p!=end);
            free(_Matrix);
        }
        _Row=B._Row;
        _Column=B._Column;
        _Matrix=B._Matrix;
        B._Matrix=nullptr;
        return *this;
    }
//------------------------------------------------------------------------------------    
//按列存储的矩阵，按列求解乘积，最慢
    Matrix multi1(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        int i,j(0),k;
        do{
            i=0;
            do{
                k=0;
                do{
                    tmp(i,j)+=(*this)(i,k)*B(k,j);
                    k++;
                }while(k<_Column);
                i++;
            }while(i<_Row);
            j++;
        }while(j<B._Column);
        return tmp;
    }
//------------------------------------------------------------------------------------    
//按列存储的矩阵，按行求解乘积，每一行都可以通过cache复用，减少按行访问的次数
    Matrix multi2(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        int i(0),j,k;
        do{
            j=0;
            do{
                k=0;
                do{
                    tmp(i,j)+=(*this)(i,k)*B(k,j);
                    k++;
                }while(k<_Column);
                j++;
            }while(j<B._Column);
            i++;
        }while(i<_Row);
        return tmp;
    }
//------------------------------------------------------------------------------------
//用指针代替下标访问行
    Matrix multi3(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        int i(0),j(0),k;
        double **pr,*p;
        do{
            j=0;
            pr=B._Matrix;
            do{
                k=0;p=*(pr++);
                do{
                    tmp(i,j)+=(*this)(i,k)**(p++);
                    k++;
                }while(k<_Column);
                j++;
            }while(j<B._Column);
            i++;
        }while(i<_Row);
        return tmp;
    }
//------------------------------------------------------------------------------------
//设置kernel函数，使用寄存器变量，一次求解4*4的元素，可以在一定程度上复用行变量
//同时在kernel内部则按行优先的原则计算乘积，获得好的流水性能
    void multi4kernel(double **c,double **a,double **b,int row,int col){
        register double t0(0),t1(0),t2(0),t3(0),t4(0),t5(0),t6(0),t7(0),
                        t8(0),t9(0),t10(0),t11(0),t12(0),t13(0),t14(0),t15(0);
        double *a0(a[0]),*a1(a[1]),*a2(a[2]),*a3(a[3]),
               *b0(b[col]),*b1(b[col+1]),*b2(b[col+2]),*b3(b[col+3]),*end=b0+_Row;
        do{
            t0+=*(a0)**(b0);
            t1+=*(a0)**(b1);
            t2+=*(a0)**(b2);
            t3+=*(a0++)**(b3);
            t4+=*(a1)**(b0);
            t5+=*(a1)**(b1);
            t6+=*(a1)**(b2);
            t7+=*(a1++)**(b3);
            t8+=*(a2)**(b0);
            t9+=*(a2)**(b1);
            t10+=*(a2)**(b2);
            t11+=*(a2++)**(b3);
            t12+=*(a3)**(b0++);
            t13+=*(a3)**(b1++);
            t14+=*(a3)**(b2++);
            t15+=*(a3++)**(b3++);
        }while(b0!=end);
        c[col][row]=t0;
        c[col+1][row]=t1;
        c[col+2][row]=t2;
        c[col+3][row]=t3;
        c[col][row+1]=t4;
        c[col+1][row+1]=t5;
        c[col+2][row+1]=t6;
        c[col+3][row+1]=t7;
        c[col][row+2]=t8;
        c[col+1][row+2]=t9;
        c[col+2][row+2]=t10;
        c[col+3][row+2]=t11;
        c[col][row+3]=t12;
        c[col+1][row+3]=t13;
        c[col+2][row+3]=t14;
        c[col+3][row+3]=t15;
    }
    Matrix multi4(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        double *tr[4];
        int i(0),j(0);
        do{
            tr[i++]=(double*)malloc(sizeof(double)*_Column);
        }while(i<4);
        do{
            i=0;
            do{
                tr[0][i]=_Matrix[i][j];
                tr[1][i]=_Matrix[i][j+1];
                tr[2][i]=_Matrix[i][j+2];
                tr[3][i]=_Matrix[i][j+3];
            }while((++i)<_Column);
            i=0;
            do{
                multi4kernel(tmp._Matrix,tr,B._Matrix,j,i);
                i+=4;
            }while(i<B._Column);
            j+=4;
        }while(j<_Row);
        return tmp;
    }
//------------------------------------------------------------------------------------   
//设置kernel函数，一次求解4*4的元素，并使用SIMD矢量加速
//对矩阵A、B分别进行packing，其中B矩阵在外层循环，A矩阵在内层循环
void multi5kernel(double **c,double **a,double *b,int row,int col){
        __m128d t01_0,t01_1,t01_2,t01_3,t23_0,t23_1,t23_2,t23_3,
                a0,a1,b0,b1,b2,b3;
        t01_0=t01_1=t01_2=t01_3=t23_0=t23_1=t23_2=t23_3=_mm_set1_pd(0);
        double *pb(b),*end=pb+4*_Column,*pa0(a[0]),*pa1(a[1]);
        do{
            a0=_mm_load_pd(pa0);
            a1=_mm_load_pd(pa1);

            b0=_mm_set1_pd(*(pb++));
            b1=_mm_set1_pd(*(pb++));
            b2=_mm_set1_pd(*(pb++));
            b3=_mm_set1_pd(*(pb++));
            t01_0+=a0*b0;
            t01_1+=a0*b1;
            t01_2+=a0*b2;
            t01_3+=a0*b3;
            t23_0+=a1*b0;
            t23_1+=a1*b1;
            t23_2+=a1*b2;
            t23_3+=a1*b3;
            pa0+=2;
            pa1+=2;
        }while(pb!=end);
        _mm_store_pd(&c[col][row],t01_0);
        _mm_store_pd(&c[col+1][row],t01_1);
        _mm_store_pd(&c[col+2][row],t01_2);
        _mm_store_pd(&c[col+3][row],t01_3);
        _mm_store_pd(&c[col][row+2],t23_0);
        _mm_store_pd(&c[col+1][row+2],t23_1);
        _mm_store_pd(&c[col+2][row+2],t23_2);
        _mm_store_pd(&c[col+3][row+2],t23_3);
    }
    Matrix multi5(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        double **ta,*tb;
        ta=(double**)malloc(sizeof(double*)*2);
        ta[0]=(double*)malloc(sizeof(double)*2*_Column);
        ta[1]=(double*)malloc(sizeof(double)*2*_Column);
        tb=(double*)malloc(sizeof(double)*4*B._Row);
        int i(0),j(0),k,t;
        do{
            i=0;k=0;
            do{
                tb[k++]=B._Matrix[j][i];
                tb[k++]=B._Matrix[j+1][i];
                tb[k++]=B._Matrix[j+2][i];
                tb[k++]=B._Matrix[j+3][i++];
            }while(i<B._Row);
            i=0;
            do{
                k=0;t=0;
                do{
                    ta[0][k]=_Matrix[t][i];
                    ta[1][k++]=_Matrix[t][i+2];
                    ta[0][k]=_Matrix[t][i+1];
                    ta[1][k++]=_Matrix[t++][i+3];
                }while(t<_Column);
                multi5kernel(tmp._Matrix,ta,tb,i,j);
                i+=4;
            }while(i<_Row);
            j+=4;
        }while(j<B._Column);
        free(ta[0]);
        free(ta[1]);
        free(ta);
        free(tb);
        return tmp;
    }
//------------------------------------------------------------------------------------
//设置kernel函数，一次求解4*4的元素，并使用SIMD矢量加速
//只对B矩阵进行packing，在循环外部进行
void multi6kernel(double **c,double **a,double *b,int row,int col){
        __m128d t01_0,t01_1,t01_2,t01_3,t23_0,t23_1,t23_2,t23_3,
                a0,a1,b0,b1,b2,b3;
        t01_0=t01_1=t01_2=t01_3=t23_0=t23_1=t23_2=t23_3=_mm_set1_pd(0);
        double *pb(b),**pa(a);
        int i(0);
        do{
            a0=_mm_load_pd(&pa[i][row]);
            a1=_mm_load_pd(&pa[i][row+2]);
            b0=_mm_set1_pd(*(pb++));
            b1=_mm_set1_pd(*(pb++));
            b2=_mm_set1_pd(*(pb++));
            b3=_mm_set1_pd(*(pb++));
            t01_0+=a0*b0;
            t01_1+=a0*b1;
            t01_2+=a0*b2;
            t01_3+=a0*b3;
            t23_0+=a1*b0;
            t23_1+=a1*b1;
            t23_2+=a1*b2;
            t23_3+=a1*b3;
            i++;
        }while(i<_Column);
        _mm_store_pd(&c[col][row],t01_0);
        _mm_store_pd(&c[col+1][row],t01_1);
        _mm_store_pd(&c[col+2][row],t01_2);
        _mm_store_pd(&c[col+3][row],t01_3);
        _mm_store_pd(&c[col][row+2],t23_0);
        _mm_store_pd(&c[col+1][row+2],t23_1);
        _mm_store_pd(&c[col+2][row+2],t23_2);
        _mm_store_pd(&c[col+3][row+2],t23_3);
    }
    Matrix multi6(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        double *tb;
        tb=(double*)malloc(sizeof(double)*4*B._Row);
        int i(0),j(0),k,t;
        do{
            i=0;k=0;
            do{
                tb[k++]=B._Matrix[j][i];
                tb[k++]=B._Matrix[j+1][i];
                tb[k++]=B._Matrix[j+2][i];
                tb[k++]=B._Matrix[j+3][i++];
            }while(i<B._Row);
            i=0;
            do{
                multi6kernel(tmp._Matrix,_Matrix,tb,i,j);
                i+=4;
            }while(i<_Row);
            j+=4;
        }while(j<B._Column);
        free(tb);
        return tmp;
    }
//------------------------------------------------------------------------------------
//设置kernel函数，一次求解4*4的元素，并使用SIMD矢量加速
//对A、B矩阵分别packing，其中A矩阵在外层循环，B矩阵在内层循环
void multi7kernel(double **c,double **a,double *b,int row,int col){
        __m128d t01_0,t01_1,t01_2,t01_3,t23_0,t23_1,t23_2,t23_3,
                a0,a1,b0,b1,b2,b3;
        t01_0=t01_1=t01_2=t01_3=t23_0=t23_1=t23_2=t23_3=_mm_set1_pd(0);
        double *pb(b),*pa0(a[0]),*pa1(a[1]),*endb=b+4*_Column;
        do{
            a0=_mm_load_pd(pa0);
            a1=_mm_load_pd(pa1);
            b0=_mm_set1_pd(*(pb++));
            b1=_mm_set1_pd(*(pb++));
            b2=_mm_set1_pd(*(pb++));
            b3=_mm_set1_pd(*(pb++));
            t01_0+=a0*b0;
            t01_1+=a0*b1;
            t01_2+=a0*b2;
            t01_3+=a0*b3;
            t23_0+=a1*b0;
            t23_1+=a1*b1;
            t23_2+=a1*b2;
            t23_3+=a1*b3;
            pa0+=2;
            pa1+=2;
        }while(pb!=endb);
        _mm_store_pd(&c[col][row],t01_0);
        _mm_store_pd(&c[col+1][row],t01_1);
        _mm_store_pd(&c[col+2][row],t01_2);
        _mm_store_pd(&c[col+3][row],t01_3);
        _mm_store_pd(&c[col][row+2],t23_0);
        _mm_store_pd(&c[col+1][row+2],t23_1);
        _mm_store_pd(&c[col+2][row+2],t23_2);
        _mm_store_pd(&c[col+3][row+2],t23_3);
    }
    Matrix multi7(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        double *tb,*ta[2],*pb,*pb0,*pb1,*pb2,*pb3,*end;
        ta[0]=(double*)malloc(sizeof(double)*2*_Column);
        ta[1]=(double*)malloc(sizeof(double)*2*_Column);
        tb=(double*)malloc(sizeof(double)*4*B._Row);
        end=tb+4*B._Row;
        int i(0),j(0),k,t;
        do{
            k=0;i=0;
            do{
                ta[0][k]=_Matrix[i][j];
                ta[1][k++]=_Matrix[i][j+2];
                ta[0][k]=_Matrix[i][j+1];
                ta[1][k++]=_Matrix[i++][j+3];
            }while(i<_Column);
            i=0;
            do{
                pb=tb;pb0=B._Matrix[i];pb1=B._Matrix[i+1];pb2=B._Matrix[i+2];pb3=B._Matrix[i+3];
                do{
                    *(pb++)=*(pb0++);
                    *(pb++)=*(pb1++);
                    *(pb++)=*(pb2++);
                    *(pb++)=*(pb3++);
                }while(pb!=end);
                multi7kernel(tmp._Matrix,ta,tb,j,i);
                i+=4;
            }while(i<B._Column);
            j+=4;
        }while(j<_Row);
        free(tb);
        free(ta[0]);
        free(ta[1]);
        return tmp;
    }
//------------------------------------------------------------------------------------
//设置kernel函数，一次求解4*4的元素，并使用SIMD矢量加速
//对A矩阵packing
void multi8kernel(double **c,double **a,double **b,int row,int col){
        __m128d t01_0,t01_1,t01_2,t01_3,t23_0,t23_1,t23_2,t23_3,
                a0,a1,b0,b1,b2,b3;
        t01_0=t01_1=t01_2=t01_3=t23_0=t23_1=t23_2=t23_3=_mm_set1_pd(0);
        double *pb0(b[col]),*pb1(b[col+1]),*pb2(b[col+2]),*pb3(b[col+3]),*pa0(a[0]),*pa1(a[1]),*endb0=pb0+_Column;
        do{
            a0=_mm_load_pd(pa0);
            a1=_mm_load_pd(pa1);
            b0=_mm_set1_pd(*(pb0++));
            b1=_mm_set1_pd(*(pb1++));
            b2=_mm_set1_pd(*(pb2++));
            b3=_mm_set1_pd(*(pb3++));
            t01_0+=a0*b0;
            t01_1+=a0*b1;
            t01_2+=a0*b2;
            t01_3+=a0*b3;
            t23_0+=a1*b0;
            t23_1+=a1*b1;
            t23_2+=a1*b2;
            t23_3+=a1*b3;
            pa0+=2;
            pa1+=2;
        }while(pb0!=endb0);
        _mm_store_pd(&c[col][row],t01_0);
        _mm_store_pd(&c[col+1][row],t01_1);
        _mm_store_pd(&c[col+2][row],t01_2);
        _mm_store_pd(&c[col+3][row],t01_3);
        _mm_store_pd(&c[col][row+2],t23_0);
        _mm_store_pd(&c[col+1][row+2],t23_1);
        _mm_store_pd(&c[col+2][row+2],t23_2);
        _mm_store_pd(&c[col+3][row+2],t23_3);
    }
    Matrix multi8(const Matrix &B){
        if(_Column!=B._Row) return *this;
        Matrix tmp(_Row,B._Column,0);
        double *tb,*ta[2],*pb,*pb0,*pb1,*pb2,*pb3,*end;
        ta[0]=(double*)malloc(sizeof(double)*2*_Column);
        ta[1]=(double*)malloc(sizeof(double)*2*_Column);
        tb=(double*)malloc(sizeof(double)*4*B._Row);
        end=tb+4*B._Row;
        int i(0),j(0),k,t;
        do{
            k=0;i=0;
            do{
                ta[0][k]=_Matrix[i][j];
                ta[1][k++]=_Matrix[i][j+2];
                ta[0][k]=_Matrix[i][j+1];
                ta[1][k++]=_Matrix[i++][j+3];
            }while(i<_Column);
            i=0;
            do{
                multi8kernel(tmp._Matrix,ta,B._Matrix,j,i);
                i+=4;
            }while(i<B._Column);
            j+=4;
        }while(j<_Row);
        free(tb);
        free(ta[0]);
        free(ta[1]);
        return tmp;
    }
//------------------------------------------------------------------------------------
    void prt(){
        for(int i=0;i<_Row;i++){
            for(int j=0;j<_Column;j++) cout<<(*this)(i,j)<<'\t';
            cout<<'\n';
        }
    }
};
//------------------------------------------------------------------------------------
#define N 2000
#define M 2000
int main(int argc,char* argv[]){
    int pd=1;
    if(argc>1) pd=atoi(argv[1]);
    if(argc==1||pd==1){
         Matrix A(N,M,1),B(M,M,2);
        double dt;
        clock_t start=clock();
        A.multi4(B);
        dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
        cout<<"简单乘积用时：\n"<<dt<<"s"<<endl;
        cout<<"----------------------------"<<endl;
        start=clock();
        A.multi8(B);
        dt=static_cast<double>(clock()-start)/CLOCKS_PER_SEC;
        cout<<"优化后用时：\n"<<dt<<"s"<<endl;
    }
    else if(pd==2){
        Matrix A(N,M),B(M,N),C,D;
        srand(clock());
        for(int i=0;i<N;i++){
            for(int j=0;j<M;j++) {
                A(i,j)=double(rand())/RAND_MAX;
                B(j,i)=double(rand())/RAND_MAX;
            }
        }
        C=A.multi2(B);
        D=A.multi8(B);
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++) {
                if(abs(C(i,j)-D(i,j))>0.00001) {
                    cout<<"error"<<endl;
                    return 1;
                }
            }
        }
        cout<<"correctly"<<endl;
        
    }
    else if(pd==3){
        Matrix A(N,N),B(N,N),C,D;
        srand(clock());
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++) {
                A(i,j)=(rand())%10;
                B(i,j)=(rand())%10;
            }
        }
        cout<<"A:\n";
        A.prt();
        cout<<"B:\n";
        B.prt();
        C=A.multi2(B);
        D=A.multi8(B);
        cout<<"C:\n";
        C.prt();
        cout<<"D:\n";
        D.prt();
    }
    else{
        Matrix A(N,N);
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++) A(i,j)=i*N+j;
        }
        A.prt();
    }
    return 0;
};

/*
int main(){
    Matrix A(N,N),B(N,N),C,D;
    srand(clock());
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) {
            A(i,j)=double(rand())/RAND_MAX;
            B(i,j)=double(rand())/RAND_MAX;
        }
    }
    C=A.multi1(B);
    D=A.multi2(B);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) {
            if(abs(C(i,j)-D(i,j))>0.00001) cout<<"error"<<endl;
        }
    }
    return 1;
}
*/