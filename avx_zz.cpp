#include <immintrin.h>
#include <iostream>
#include <ctime>
using namespace std;
#define N 4096
#define S1
int main()
{
    double* x=(double*)_mm_malloc(sizeof(double)*N*N,64);
    double* y=(double*)_mm_malloc(sizeof(double)*N*N,64);
    int k=0;    
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            x[k++]=i;
        }
    }
    double *p1,*p2,*p3,*p4;
    double *d1,*d2,*d3,*d4,*t=y;
    d1=y;d2=y+N;d3=y+2*N;d4=y+3*N;
    p1=x;p2=x+N;p3=x+2*N;p4=x+3*N;
    t+=4;
    clock_t s=clock();
    for(int i=0;i<N/4;i++)
    {
        for(int j=0;j<N/4;j++)
        {
            __m256d s1,s2,s3,s4,t1,t2,t3,t4,t5,t6,t7,t8;
            s1=_mm256_load_pd(p1);
            s2=_mm256_load_pd(p2);
            s3=_mm256_load_pd(p3);
            s4=_mm256_load_pd(p4);
            t1=_mm256_permute4x64_pd(s1,0b01001110);
            t2=_mm256_permute4x64_pd(s2,0b01001110);
            t3=_mm256_permute4x64_pd(s3,0b01001110);
            t4=_mm256_permute4x64_pd(s4,0b01001110);
            t5=_mm256_blend_pd(s1,t3,0b1100);
            t6=_mm256_blend_pd(s2,t4,0b1100);
            t7=_mm256_blend_pd(t1,s3,0b1100);
            t8=_mm256_blend_pd(t2,s4,0b1100);
            #ifdef S1
            _mm256_store_pd(d1,_mm256_unpacklo_pd(t5,t6));
            _mm256_store_pd(d2,_mm256_unpackhi_pd(t5,t6));
            _mm256_store_pd(d3,_mm256_unpacklo_pd(t7,t8));
            _mm256_store_pd(d4,_mm256_unpackhi_pd(t7,t8));
            #else
            _mm256_stream_pd(d1,_mm256_unpacklo_pd(t5,t6));
            _mm256_stream_pd(d2,_mm256_unpackhi_pd(t5,t6));
            _mm256_stream_pd(d3,_mm256_unpacklo_pd(t7,t8));
            _mm256_stream_pd(d4,_mm256_unpackhi_pd(t7,t8));      
            #endif
            p1+=4;p2+=4;p3+=4;p4+=4;
            d1+=4*N;d2+=4*N;d3+=4*N;d4+=4*N;
        }
        p1+=3*N;p2+=3*N;p3+=3*N;p4+=3*N;
        d1=t;d2=t+N;d3=t+2*N;d4=t+3*N;
        t+=4;
    } 
cout<<y[101]<<'\n';
cout<<double(clock()-s)/CLOCKS_PER_SEC<<'\n';
/*
    k=0;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            cout<<y[k++]<<'\t';
        }
        cout<<'\n';
    }
*/   
    return 0;
}