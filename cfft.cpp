#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>
using namespace std;
constexpr double pi(bool k=0){
    return atan(1)*4;
}
void prt(complex<double>* x,int n){
    for(int i=0;i<n;i++) cout<<x[i]<<'\t';
    cout<<endl;
}
void dotmul(complex<double> *x,complex<double> *y,int n){
    double res(0);
    for(int i=0;i<n;i++) res+=norm(x[i]-y[i]);
    cout<<res<<endl;
}
template <typename T>
void Swap(T *x,unsigned i,unsigned j){
    T tmp=x[i];
    x[i]=x[j];
    x[j]=tmp;
}
template <typename T>
void BitReverse(T *x,unsigned n){
    if(n<4) return;
    bool *record=(bool*)malloc(n);
    for(int i=0;i<n;i++) record[i]=0;
    unsigned count=0,k=n-1;
    while(k){
        k>>=1;
        count++;
    }
    for(unsigned i=0;i<n;i++){
        if(record[i]) continue;
        k=0;
        for(unsigned j=0;j<count;j++) {
            k|=(1<<j&i)>>j;
            k<<=1;
        }
        k>>=1;
        Swap(x,i,k);
        record[k]=!record[k];
    }
    free(record);
}
template <int n,typename T>
struct BitReverse4{
    bool judg[n];
    int record[n];
    BitReverse4(){
        for(int i=0;i<n;i++) judg[i]=0;
        unsigned count=0,k=n-1;
        while(k){
            k>>=1;
            count++;
        }
        for(unsigned i=0;i<n;i++){
            if(judg[i]) continue;
            k=0;
            for(unsigned j=0;j<count;j+=2) {
                k|=(3<<j&i)>>j;
                k<<=2;
            }
            record[i]=k>>2;
            judg[k>>2]=!judg[k>>2];
        }
    }
    void operator()(T *x){
        if(n<4) return;
        for(int i=0;i<n;i++){
            if(!judg[i]) Swap(x,i,record[i]);
        }
    }
};

template <int n,typename T>
void cdft(complex<T> *in,complex<T> *out){
    complex<T> *W;
    W=new complex<T>[n];
    for(int i=0;i<n;i++) W[i]=complex<double>(cos(2*i*pi()/n),-sin(2*i*pi()/n));
    for(int i=0;i<n;i++){
        out[i]=0;
        for(int j=0;j<n;j++){
            out[i]+=in[j]*W[i*j%n];
        }
    }
    delete W;
}
template <int n,typename T>
void cfft(complex<T> *in,complex<T> *out){
    int tt=n,td=1,i,j;
    complex<T> *W,t1,t2;
    W=new complex<T>[n];    
    for(i=0;i<n;i++) {
        if(i<(n>>1))W[i]=complex<double>(cos(2*i*pi()/n),-sin(2*i*pi()/n));
        out[i]=in[i];
    }
    BitReverse(out,n);
    do{
        i=0;
        tt=tt>>1;
        do{
            j=0;
            do{
                t1=out[i];
                t2=out[i+td]*W[j*tt];
                out[i]=t1+t2;
                out[i+td]=t1-t2;
                i++;j++;
            }while(j<td);
            i+=td;
        }while(i<n);
        td*=2;
    }while(!(tt&1));
    delete W;
}
template <int n,typename T>
struct cfft2
{
    complex<T> *W;
    cfft2(){
        W=new complex<T>[n]; 
        for(int i=0;i<n;i++) {
            W[i]=complex<double>(cos(2*i*pi()/n),-sin(2*i*pi()/n));
        }
    }
    ~cfft2(){
        delete []W;
    }
    void operator()(complex<double> *in,complex<double> *out){
        int tt=n,td=1,i,j;
        complex<T> t1,t2;
        for(i=0;i<n;i++) out[i]=in[i];
        BitReverse(out,n);
        do{
            i=0;
            tt=tt>>1;
            do{
                j=0;
                do{
                    t1=out[i];
                    t2=out[i+td]*W[j*tt];
                    out[i]=t1+t2;
                    out[i+td]=t1-t2;
                    i++;j++;
                }while(j<td);
                i+=td;
            }while(i<n);
            td*=2;
        }while(!(tt&1));
    }
    void operator()(complex<double> *out){
        int tt=n,td=1,i,j;
        complex<T> t1,t2;
        //T r1,r2,i1,i2,rw,iw;
        BitReverse(out,n);
        do{
            i=0;
            tt=tt>>1;
            do{
                j=0;
                do{
                    
                    t1=out[i];
                    t2=out[i+td]*W[j*tt];
                    out[i]=t1+t2;
                    out[i+td]=t1-t2;
                    
                   /*
                   r1=out[i].real();i1=out[i].imag();
                   r2=out[i+td].real();i2=out[i+td].imag();
                   rw=W[j*tt].real();iw=W[j*tt].imag();
                   out[i].real(r1+r2);out[i].imag(i1+i2);
                   out[i+td].real(r1-r2);out[i+td].imag(i1-i2);
                    */
                    i++;j++;
                }while(j<td);
                i+=td;
            }while(i<n);
            td*=2;
        }while(!(tt&1));
        i=0;
        do{
            out[i++]/=n;
        }while(i!=n);
    }
};

template <int n,typename T>
struct cfft4
{
    BitReverse4<n,complex<T>> br;
    inline complex<T> muli(complex<T> &x){
        return complex<T>(-x.imag(),x.real());
    }
    complex<T> *W;
    cfft4(){
        W=new complex<T>[n]; 
        for(int i=0;i<n;i++) {
            W[i]=complex<T>(cos(2*i*pi()/n),-sin(2*i*pi()/n));
        }
    }
    ~cfft4(){
        delete []W;
    }
    void operator()(complex<T> *in,complex<T> *out){
        int tt=n,td=1,i,j;
        complex<T> t1,t2,t3,t4;
        for(i=0;i<n;i++) out[i]=in[i];
        br(out);
        do{
            i=0;
            tt>>=2;
            do{
                j=0;
                do{
                    t1=out[i];
                    t2=out[i+td]*W[j*tt];
                    t3=out[i+td*2]*W[j*tt*2];
                    t4=out[i+td*3]*W[j*tt*3];
                    out[i]=t1+t2+t3+t4;
                    out[i+td]=t1-muli(t2)-t3+muli(t4);
                    out[i+td*2]=t1-t2+t3-t4;
                    out[i+td*3]=t1+muli(t2)-t3-muli(t4);
                    i++;j++;
                }while(j<td);
                i+=td*3;
            }while(i<n);
            td<<=2;
        }while(tt!=1);
    }
    void operator()(complex<T> *out){
        int tt=n,td=1,i,j;
        complex<T> t1,t2,t3,t4;
        //for(i=0;i<n;i++) out[i]=in[i];
        br(out);
        do{
            i=0;
            tt>>=2;
            do{
                j=0;
                do{
                    t1=out[i];
                    t2=out[i+td]*W[j*tt];
                    t3=out[i+td*2]*W[j*tt*2];
                    t4=out[i+td*3]*W[j*tt*3];
                    out[i]=t1+t2+t3+t4;
                    out[i+td]=t1-muli(t2)-t3+muli(t4);
                    out[i+td*2]=t1-t2+t3-t4;
                    out[i+td*3]=t1+muli(t2)-t3-muli(t4);
                    i++;j++;
                }while(j<td);
                i+=td*3;
            }while(i<n);
            td<<=2;
        }while(tt!=1);
    }
};
#define N 4
int main(){
    srand(clock());
    complex<double> x[N],y[N],z[N];
    cfft2<N,double> fft1;
    cfft4<N,double> fft2;
    for(int i=0;i<N;i++) x[i]=rand()%10;
    //prt(x,N);
    //for(int i=0;i<N;i++) cfft<N,double>(x,y);
    //for(int i=0;i<100*N;i++) fft1(x,y);
    for(int i=0;i<4;i++) 
    fft2(x);
    //cdft<N,double>(x,z);
    //fft2(x);
    //fft2(x,y);
    //prt(y,N);
    //for(int i=0;i<N;i++) cdft<N,double>(x,z);
    //cdft<N,double>(x,z);
    //fft1(x,z);
    //prt(z,N);
    //dotmul(x,z,N);
}