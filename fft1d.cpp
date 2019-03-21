#ifndef FFT1D_HPP
#define FFT1D_HPP
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <unordered_map>
#include <vector>
#include <iostream>
constexpr double const pi = atan(double(1)) * 4;
constexpr double const c4 = cos(pi/4);
__m256d const mc4=_mm256_set1_pd(c4);
//-------1---------2---------3---------4---------5---------
template <int n>
struct BitReverse2
{
    int rec[n],k;
    std::vector<int> sp;
    BitReverse2()
    {
        std::unordered_map<int,int> p;
        int x[n];
        for(int i=0;i<n;i++) {
            p[i]=i;
            x[i]=i;
        }
        size_t count = 0;
        k = n - 1;
        while (k)
        {
            k >>= 1;
            count++;
        }
        for (unsigned i = 0; i < n; i++)
        {
            k = 0;
			k |= (7 & i);
			k <<= 2;
            for (unsigned j = 3; j < count; j += 2)
            {
                k |= (3 << j & i) >> j;
                k <<= 2;
            }
			k>>=2;
            rec[i]=k;
        }
        k=0;
        for (int i = 1; i < n-1; i++)
        {
            if(i==p[rec[i]]) continue;
            int t1=x[i],t2=x[p[rec[i]]];
            std::swap(x[i],x[p[rec[i]]]);
            p[t1]=p[rec[i]];
            p[t2]=i;
            k++;
        }
        sp.resize(2*k);
        for(int i=0;i<n;i++) {
            p[i]=i;
            x[i]=i;
        }
        int tk=0;
        for (int i = 1; i < n-1; i++)
        {
            if(i==p[rec[i]]) continue;
            int t1=x[i],t2=x[p[rec[i]]];
            std::swap(x[i],x[p[rec[i]]]);
            sp[2*tk]=i;sp[2*tk+1]=p[rec[i]];
            tk++;
            p[t1]=p[rec[i]];
            p[t2]=i;
        }
    }
    void operator()(double *x)
    {
        for(int i=0;i<2*k;i+=2){
            std::swap(x[sp[i]],x[sp[i+1]]);
        }
    }
};
template <int n>
struct BitReverse4
{
    bool judg[n];
    int record[n],rec[n];
    BitReverse4()
    {
        for (int i = 0; i < n; i++)
            judg[i] = 0;
        unsigned count = 0, k = n - 1;
        while (k)
        {
            k >>= 1;
            count++;
        }
        for (unsigned i = 0; i < n; i++)
        {
            if (judg[i])
                continue;
            k = 0;
            for (unsigned j = 0; j < count; j += 2)
            {
                k |= (3 << j & i) >> j;
                k <<= 2;
            }
			k>>=2;
            record[i] = k;
            rec[i]=record[i];
            rec[rec[i]]=i;
            judg[k] = !judg[k];
        }
    }
    void operator()(double *x)
    {
        for (int i = 0; i < n; i++)
        {
            if (!judg[i])
                std::swap(x[i], x[record[i]]);
        }
    }
};
size_t log4(size_t x)
{
    size_t k = 0;
    do
    {
        x >>= 2;
        k++;
    } while (x != 1);
    return k;
}
size_t log2(size_t x)
{
	size_t k=0;
	do
	{
		x>>=2;
		k++;
	} while (x!=2);
	return k;
	
}
//-------1---------2---------3---------4---------5---------
template <size_t n>
struct fft2_row
{
    double *W[3];
    BitReverse2<n> BR;
    fft2_row(BitReverse2<n>& r)
    {
        BR=r;
        int k=log2(n);
		W[0] = (double*)_mm_malloc((n-8)/3*sizeof(double)*2,32);
		W[1] = (double*)_mm_malloc((n-8)/3*sizeof(double)*2,32);
		W[2] = (double*)_mm_malloc((n-8)/3*sizeof(double)*2,32);
		double *r0=W[0],*r1=W[1],*r2=W[2],*i0=W[0]+(n-8)/3,*i1=W[1]+(n-8)/3,*i2=W[2]+(n-8)/3;
		for (int i = 1; i < k; i++){
			for(int j=0;j<(1 << (i * 2+1));j++){
				*(r0++)=cos(2 * j * pi / (1 << (i * 2+3)));
				*(r1++)=cos(2 * 2*j * pi / (1 << (i * 2+3)));
				*(r2++)=cos(2 * 3*j * pi / (1 << (i * 2+3)));
				*(i0++)=-sin(2 * j * pi / (1 << (i * 2+3)));
				*(i1++)=-sin(2 * 2*j * pi / (1 << (i * 2+3)));
				*(i2++)=-sin(2 * 3*j * pi / (1 << (i * 2+3)));
			}
		}
    }
    ~fft2_row()
    {
        _mm_free(W[0]);
        _mm_free(W[1]);
        _mm_free(W[2]);
    }
	void operator()(double *x)
	{
        BR(x);
		__m256d mr1,mr2,mr3,mr4,mi1,mi2,mi3,mi4;
        double *wr1,*wr2,*wr3,*wi1,*wi2,*wi3,*twr1,*twr2,*twr3,*twi1,*twi2,*twi3;
        double *rx1,*rx2,*rx3,*rx4,*ix1,*ix2,*ix3,*ix4,*end=x+2*n;
        int i(0),j,td=2;
        wr1=W[0];wr2=W[1];wr3=W[2];wi1=W[0]+(n-8)/3;wi2=W[1]+(n-8)/3;wi3=W[2]+(n-8)/3;
        rx1=x;rx2=x+1;rx3=x+2;rx4=x+3;
        ix1=n+x;ix2=n+1+x;ix3=n+2+x;ix4=n+3+x;
        double r1, r2, r3, r4, r5, r6, r7, r8, 
			   i1, i2, i3, i4, i5, i6, i7, i8,
			   rt2, rt4, rt6, rt8, it2, it4, it6, it8;
		do
    	{
			r1=x[i];r2=x[i+1];r3=x[i+2];r4=x[i+3];r5=x[i+4];r6=x[i+5];r7=x[i+6];r8=x[i+7];
			i1=x[i+n];i2=x[i+1+n];i3=x[i+2+n];i4=x[i+3+n];i5=x[i+4+n];i6=x[i+5+n];i7=x[i+6+n];i8=x[i+7+n];
			rt2=c4*(r2-i2);rt4=c4*(r4-i4);rt6=c4*(r6-i6);rt8=c4*(r8-i8);
			it2=c4*(r2+i2);it4=c4*(r4+i4);it6=c4*(r6+i6);it8=c4*(r8+i8);
			x[i]=r1+r2+r3+r4+r5+r6+r7+r8;
			x[i+n]=i1+i2+i3+i4+i5+i6+i7+i8;
			x[i+1]=r1+it2+i3-rt4-r5-it6-i7+rt8;
			x[i+1+n]=i1-rt2-r3-it4-i5+rt6+r7+it8;
			x[i+2]=r1+i2-r3-i4+r5+i6-r7-i8;
			x[i+2+n]=i1-r2-i3+r4+i5-r6-i7+r8;
			x[i+3]=r1-rt2-i3+it4-r5+rt6+i7-it8;
			x[i+3+n]=i1-it2+r3-rt4-i5+it6-r7+rt8;
			x[i+4]=r1-r2+r3-r4+r5-r6+r7-r8;
			x[i+4+n]=i1-i2+i3-i4+i5-i6+i7-i8;
			x[i+5]=r1-it2+i3+rt4-r5+it6-i7-rt8;
			x[i+5+n]=i1+rt2-r3+it4-i5-rt6+r7-it8;
			x[i+6]=r1-i2-r3+i4+r5-i6-r7+i8;
			x[i+6+n]=i1+r2-i3-r4+i5+r6-i7-r8;
			x[i+7]=r1+rt2-i3-it4-r5-rt6+i7+it8;
			x[i+7+n]=i1+it2+r3+rt4-i5-it6-r7-rt8;
			i+=8;
    	} while (i < n);
	    td <<= 2;
    	do
		{
	    	i = 0;
			twr1=wr1;twr2=wr2;twr3=wr3;twi1=wi1;twi2=wi2;twi3=wi3;
            rx1=x;rx2=x+td;rx3=td+td+x;rx4=td+td+td+x;
            ix1=x+n;ix2=n+td+x;ix3=n+td+td+x;ix4=n+td+td+td+x;
    		do
    		{
		    	wr1=twr1;wr2=twr2;wr3=twr3;wi1=twi1;wi2=twi2;wi3=twi3;
				for (j=0; j < td; j+=4)
			    {
				    mr1 = _mm256_load_pd(rx1);
					mi1 = _mm256_load_pd(ix1);
	    			mr2 = _mm256_fmsub_pd(_mm256_load_pd(rx2) , _mm256_load_pd(wr1) , _mm256_load_pd(ix2) * _mm256_load_pd(wi1));
		    		mi2 = _mm256_fmadd_pd(_mm256_load_pd(rx2) , _mm256_load_pd(wi1) , _mm256_load_pd(ix2) * _mm256_load_pd(wr1));
					mr3 = _mm256_fmsub_pd(_mm256_load_pd(rx3) , _mm256_load_pd(wr2) , _mm256_load_pd(ix3) * _mm256_load_pd(wi2));
			    	mi3 = _mm256_fmadd_pd(_mm256_load_pd(rx3) , _mm256_load_pd(wi2) , _mm256_load_pd(ix3) * _mm256_load_pd(wr2));
				    mr4 = _mm256_fmsub_pd(_mm256_load_pd(rx4) , _mm256_load_pd(wr3) , _mm256_load_pd(ix4) * _mm256_load_pd(wi3));
					mi4 = _mm256_fmadd_pd(_mm256_load_pd(rx4) , _mm256_load_pd(wi3) , _mm256_load_pd(ix4) * _mm256_load_pd(wr3));
                    _mm256_store_pd(rx1,mr1 + mr2 + mr3 + mr4);
		    		_mm256_store_pd(ix1,mi1 + mi2 + mi3 + mi4);
			    	_mm256_store_pd(rx2,mr1 + mi2 - mr3 - mi4);
			    	_mm256_store_pd(ix2,mi1 - mr2 - mi3 + mr4);
				    _mm256_store_pd(rx3,mr1 - mr2 + mr3 - mr4);
    				_mm256_store_pd(ix3,mi1 - mi2 + mi3 - mi4);
    				_mm256_store_pd(rx4,mr1 - mi2 - mr3 + mi4);
		    		_mm256_store_pd(ix4,mi1 + mr2 - mi3 - mr4);
			    	rx1+=4;rx2+=4;rx3+=4;rx4+=4;
                    ix1+=4;ix2+=4;ix3+=4;ix4+=4;
                    wr1+=4;wr2+=4;wr3+=4;
    				wi1+=4;wi2+=4;wi3+=4;
			    }
				rx1+=3*td;rx2+=3*td;rx3+=3*td;rx4+=3*td;
                ix1+=3*td;ix2+=3*td;ix3+=3*td;ix4+=3*td;
		    } while (ix4 < end);
		    td <<= 2;
	    } while (td != n);
	}
};
template <size_t n>
struct fft2_col
{
	double *W[3];
	BitReverse2<n> BR;
    fft2_col(BitReverse2<n>& r)
    {
        BR=r;
        int k=log2(n);
		W[0] = (double*)_mm_malloc(((n-5)/3-k)*sizeof(double)*2,32);
		W[1] = (double*)_mm_malloc(((n-5)/3-k)*sizeof(double)*2,32);
		W[2] = (double*)_mm_malloc(((n-5)/3-k)*sizeof(double)*2,32);
		double *r0=W[0],*r1=W[1],*r2=W[2],*i0=W[0]+(n-5)/3-k,*i1=W[1]+(n-5)/3-k,*i2=W[2]+(n-5)/3-k;
		for (int i = 1; i < k; i++){
			for(int j=1;j<(1 << (i * 2+1));j++){
				*(r0++)=cos(2 * j * pi / (1 << (i * 2+3)));
				*(r1++)=cos(2 * 2*j * pi / (1 << (i * 2+3)));
				*(r2++)=cos(2 * 3*j * pi / (1 << (i * 2+3)));
				*(i0++)=-sin(2 * j * pi / (1 << (i * 2+3)));
				*(i1++)=-sin(2 * 2*j * pi / (1 << (i * 2+3)));
				*(i2++)=-sin(2 * 3*j * pi / (1 << (i * 2+3)));
			}
		}
    }
    ~fft2_col()
    {
        _mm_free(W[0]);
        _mm_free(W[1]);
        _mm_free(W[2]);
    }
	void operator()(double **x,int k)
	{
		int tt, td , i, j,col=4;
		__m256d r1, r2, r3, r4, r5, r6, r7, r8, 
				i1, i2, i3, i4, i5, i6, i7, i8,
				rt2, rt4, rt6, rt8, it2, it4, it6, it8;
		double *rx[n],*ix[n];
		for(i=0;i<n;i++) 
		{
			rx[i]=&x[i][k];
			ix[i]=&x[i][k+col];
		}
		double *tu=(double*)_mm_malloc(4*n*sizeof(double)*2,32);
		double *wr1,*wr2,*wr3,*wi1,*wi2,*wi3,*twr1,*twr2,*twr3,*twi1,*twi2,*twi3;
        double *rx1,*rx2,*rx3,*rx4,*ix1,*ix2,*ix3,*ix4,*end=tu+8*n,
			   *rx5,*rx6,*rx7,*rx8,*ix5,*ix6,*ix7,*ix8;
		wr1=W[0];wr2=W[1];wr3=W[2];
		wi1=W[0]+((n-5)/3-log2(n));wi2=W[1]+((n-5)/3-log2(n));wi3=W[2]+((n-5)/3-log2(n));
        rx1=tu;rx2=tu+4;rx3=tu+8;rx4=tu+12;rx5=tu+16;rx6=tu+20;rx7=tu+24;rx8=tu+28;
        ix1=4*n+tu;ix2=4*(n+1)+tu;ix3=4*(n+2)+tu;ix4=4*(n+3)+tu;
		ix5=4*(n+4)+tu;ix6=4*(n+5)+tu;ix7=4*(n+6)+tu;ix8=4*(n+7)+tu;
        td=2;
		i=0;
		do
		{
			r1 = _mm256_load_pd(rx[BR.rec[i]]);
			i1 = _mm256_load_pd(ix[BR.rec[i]]);
			r2 = _mm256_load_pd(rx[BR.rec[i+1]]);
			i2 = _mm256_load_pd(ix[BR.rec[i+1]]);
			r3 = _mm256_load_pd(rx[BR.rec[i+2]]);
			i3 = _mm256_load_pd(ix[BR.rec[i+2]]);
			r4 = _mm256_load_pd(rx[BR.rec[i+3]]);
			i4 = _mm256_load_pd(ix[BR.rec[i+3]]);
			r5 = _mm256_load_pd(rx[BR.rec[i+4]]);
			i5 = _mm256_load_pd(ix[BR.rec[i+4]]);
			r6 = _mm256_load_pd(rx[BR.rec[i+5]]);
			i6 = _mm256_load_pd(ix[BR.rec[i+5]]);
			r7 = _mm256_load_pd(rx[BR.rec[i+6]]);
			i7 = _mm256_load_pd(ix[BR.rec[i+6]]);
			r8 = _mm256_load_pd(rx[BR.rec[i+7]]);
			i8 = _mm256_load_pd(ix[BR.rec[i+7]]);
			rt2=mc4*(r2-i2);rt4=mc4*(r4-i4);rt6=mc4*(r6-i6);rt8=mc4*(r8-i8);
			it2=mc4*(r2+i2);it4=mc4*(r4+i4);it6=mc4*(r6+i6);it8=mc4*(r8+i8);
			_mm256_store_pd(rx1,r1+r2+r3+r4+r5+r6+r7+r8);
			_mm256_store_pd(ix1,i1+i2+i3+i4+i5+i6+i7+i8);
			_mm256_store_pd(rx2,r1+it2+i3-rt4-r5-it6-i7+rt8);
			_mm256_store_pd(ix2,i1-rt2-r3-it4-i5+rt6+r7+it8);
			_mm256_store_pd(rx3,r1+i2-r3-i4+r5+i6-r7-i8);
			_mm256_store_pd(ix3,i1-r2-i3+r4+i5-r6-i7+r8);
			_mm256_store_pd(rx4,r1-rt2-i3+it4-r5+rt6+i7-it8);
			_mm256_store_pd(ix4,i1-it2+r3-rt4-i5+it6-r7+rt8);
			_mm256_store_pd(rx5,r1-r2+r3-r4+r5-r6+r7-r8);
			_mm256_store_pd(ix5,i1-i2+i3-i4+i5-i6+i7-i8);
			_mm256_store_pd(rx6,r1-it2+i3+rt4-r5+it6-i7-rt8);
			_mm256_store_pd(ix6,i1+rt2-r3+it4-i5-rt6+r7-it8);
			_mm256_store_pd(rx7,r1-i2-r3+i4+r5-i6-r7+i8);
			_mm256_store_pd(ix7,i1+r2-i3-r4+i5+r6-i7-r8);
			_mm256_store_pd(rx8,r1+rt2-i3-it4-r5-rt6+i7+it8);
			_mm256_store_pd(ix8,i1+it2+r3+rt4-i5-it6-r7-rt8);
			rx1+=32;rx2+=32;rx3+=32;rx4+=32;rx5+=32;rx6+=32;rx7+=32;rx8+=32;
            ix1+=32;ix2+=32;ix3+=32;ix4+=32;ix5+=32;ix6+=32;ix7+=32;ix8+=32;
			i+=8;
		} while (ix4 < end);
		td <<= 2;
		do
		{
			rx1=tu;rx2=tu+4*td;rx3=tu+8*td;rx4=tu+12*td;
        	ix1=4*n+tu;ix2=4*(n+td)+tu;ix3=4*(n+td+td)+tu;ix4=4*(n+td+td+td)+tu;
			twr1=wr1;twr2=wr2;twr3=wr3;twi1=wi1;twi2=wi2;twi3=wi3;
			do
			{
				r1 = _mm256_load_pd(rx1);
				i1 = _mm256_load_pd(ix1);
				r2 = _mm256_load_pd(rx2);
				i2 = _mm256_load_pd(ix2);
				r3 = _mm256_load_pd(rx3);
				i3 = _mm256_load_pd(ix3);
				r4 = _mm256_load_pd(rx4);
				i4 = _mm256_load_pd(ix4);
				_mm256_store_pd(rx1,r1 + r2 + r3 + r4);
				_mm256_store_pd(ix1,i1 + i2 + i3 + i4);
				_mm256_store_pd(rx2,r1 + i2 - r3 - i4);
				_mm256_store_pd(ix2,i1 - r2 - i3 + r4);
				_mm256_store_pd(rx3,r1 - r2 + r3 - r4);
				_mm256_store_pd(ix3,i1 - i2 + i3 - i4);
				_mm256_store_pd(rx4,r1 - i2 - r3 + i4);
				_mm256_store_pd(ix4,i1 + r2 - i3 - r4);
				rx1+=4;rx2+=4;rx3+=4;rx4+=4;
            	ix1+=4;ix2+=4;ix3+=4;ix4+=4;
				j=1;
				wr1=twr1;wr2=twr2;wr3=twr3;wi1=twi1;wi2=twi2;wi3=twi3;
				for (; j < td; j++)
				{
					r1 = _mm256_load_pd(rx1);
					i1 = _mm256_load_pd(ix1);
					r2 = _mm256_fmsub_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wr1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wi1));
					i2 = _mm256_fmadd_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wi1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wr1));
					r3 = _mm256_fmsub_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wr2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wi2));
					i3 = _mm256_fmadd_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wi2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wr2));
					r4 = _mm256_fmsub_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wr3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wi3));
					i4 = _mm256_fmadd_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wi3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wr3));
					_mm256_store_pd(rx1,r1 + r2 + r3 + r4);
					_mm256_store_pd(ix1,i1 + i2 + i3 + i4);
					_mm256_store_pd(rx2,r1 + i2 - r3 - i4);
					_mm256_store_pd(ix2,i1 - r2 - i3 + r4);
					_mm256_store_pd(rx3,r1 - r2 + r3 - r4);
					_mm256_store_pd(ix3,i1 - i2 + i3 - i4);
					_mm256_store_pd(rx4,r1 - i2 - r3 + i4);
					_mm256_store_pd(ix4,i1 + r2 - i3 - r4);
					wr1++;wr2++;wr3++;
					wi1++;wi2++;wi3++;
					rx1+=4;rx2+=4;rx3+=4;rx4+=4;
					ix1+=4;ix2+=4;ix3+=4;ix4+=4;
				}
				rx1+=12*td;rx2+=12*td;rx3+=12*td;rx4+=12*td;
                ix1+=12*td;ix2+=12*td;ix3+=12*td;ix4+=12*td;
			} while (ix4 < end);
			td <<= 2;
		} while (td != n>>2);
		rx1=tu;rx2=tu+4*td;rx3=tu+8*td;rx4=tu+12*td;
		ix1=4*n+tu;ix2=4*(n+td)+tu;ix3=4*(n+td+td)+tu;ix4=4*(n+td+td+td)+tu;
		r1 = _mm256_load_pd(rx1);
		i1 = _mm256_load_pd(ix1);
		r2 = _mm256_load_pd(rx2);
		i2 = _mm256_load_pd(ix2);
		r3 = _mm256_load_pd(rx3);
		i3 = _mm256_load_pd(ix3);
		r4 = _mm256_load_pd(rx4);
		i4 = _mm256_load_pd(ix4);
		i=0;
		_mm256_store_pd(rx[i],r1+r2+r3+r4);
		_mm256_store_pd(ix[i],i1+i2+i3+i4);
		_mm256_store_pd(rx[i+td],r1+i2-r3-i4);
		_mm256_store_pd(ix[i+td],i1-r2-i3+r4);
		_mm256_store_pd(rx[i+td+td],r1-r2+r3-r4);
		_mm256_store_pd(ix[i+td+td],i1-i2+i3-i4);
		_mm256_store_pd(rx[i+td+td+td],r1-i2-r3+i4);
		_mm256_store_pd(ix[i+td+td+td],i1+r2-i3-r4);
		rx1+=4;rx2+=4;rx3+=4;rx4+=4;
		ix1+=4;ix2+=4;ix3+=4;ix4+=4;
		j=1;
		i=1;
		for (; j < td; j++)
		{
			r1 = _mm256_load_pd(rx1);
			i1 = _mm256_load_pd(ix1);
			r2 = _mm256_fmsub_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wr1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wi1));
			i2 = _mm256_fmadd_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wi1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wr1));
			r3 = _mm256_fmsub_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wr2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wi2));
			i3 = _mm256_fmadd_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wi2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wr2));
			r4 = _mm256_fmsub_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wr3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wi3));
			i4 = _mm256_fmadd_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wi3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wr3));
			_mm256_store_pd(rx[i],r1+r2+r3+r4);
			_mm256_store_pd(ix[i],i1+i2+i3+i4);
			_mm256_store_pd(rx[i+td],r1+i2-r3-i4);
			_mm256_store_pd(ix[i+td],i1-r2-i3+r4);
			_mm256_store_pd(rx[i+td+td],r1-r2+r3-r4);
			_mm256_store_pd(ix[i+td+td],i1-i2+i3-i4);
			_mm256_store_pd(rx[i+td+td+td],r1-i2-r3+i4);
			_mm256_store_pd(ix[i+td+td+td],i1+r2-i3-r4);
			i++;
			wr1++;wr2++;wr3++;
			wi1++;wi2++;wi3++;
			rx1+=4;rx2+=4;rx3+=4;rx4+=4;
			ix1+=4;ix2+=4;ix3+=4;ix4+=4;
		}
		free(tu);
    }
};
template <size_t n>
struct fft4_row
{
    double *W[3];
    BitReverse4<n> BR;
    fft4_row(BitReverse4<n>& r)
    {
        BR=r;
        int k=log4(n);
		W[0] = (double*)_mm_malloc((n-4)/3*sizeof(double)*2,32);
		W[1] = (double*)_mm_malloc((n-4)/3*sizeof(double)*2,32);
		W[2] = (double*)_mm_malloc((n-4)/3*sizeof(double)*2,32);
		double *r0=W[0],*r1=W[1],*r2=W[2],*i0=W[0]+(n-4)/3,*i1=W[1]+(n-4)/3,*i2=W[2]+(n-4)/3;
		for (int i = 1; i < k; i++){
			for(int j=0;j<(1 << (i * 2));j++){
				*(r0++)=cos(2 * j * pi / (1 << (i * 2+2)));
				*(r1++)=cos(2 * 2*j * pi / (1 << (i * 2+2)));
				*(r2++)=cos(2 * 3*j * pi / (1 << (i * 2+2)));
				*(i0++)=-sin(2 * j * pi / (1 << (i * 2+2)));
				*(i1++)=-sin(2 * 2*j * pi / (1 << (i * 2+2)));
				*(i2++)=-sin(2 * 3*j * pi / (1 << (i * 2+2)));
			}
		}
    }
    ~fft4_row()
    {
        _mm_free(W[0]);
        _mm_free(W[1]);
        _mm_free(W[2]);
    }
    void operator()(double* x)
    {
        BR(x);
        double r1, r2, r3, r4, i1, i2, i3, i4;
        __m256d mr1,mr2,mr3,mr4,mi1,mi2,mi3,mi4;
        double *wr1,*wr2,*wr3,*wi1,*wi2,*wi3,*twr1,*twr2,*twr3,*twi1,*twi2,*twi3;
        double *rx1,*rx2,*rx3,*rx4,*ix1,*ix2,*ix3,*ix4,*end=x+2*n;
        int i,j,td=1;
        wr1=W[0];wr2=W[1];wr3=W[2];wi1=W[0]+(n-4)/3;wi2=W[1]+(n-4)/3;wi3=W[2]+(n-4)/3;
        rx1=x;rx2=x+1;rx3=x+2;rx4=x+3;
        ix1=n+x;ix2=n+1+x;ix3=n+2+x;ix4=n+3+x;
        do
    	{
			r1 = *rx1;r2 = *rx2;r3 = *rx3;r4 = *rx4;
            i1 = *ix1;i2 = *ix2;i3 = *ix3;i4 = *ix4;
    		*rx1=r1 + r2 + r3 + r4;
            *ix1=i1 + i2 + i3 + i4;
		    *rx2=r1 + i2 - r3 - i4;
		    *ix2=i1 - r2 - i3 + r4;
    		*rx3=r1 - r2 + r3 - r4;
            *ix3=i1 - i2 + i3 - i4;
    		*rx4=r1 - i2 - r3 + i4;
		    *ix4=i1 + r2 - i3 - r4;
		    rx1+=4;rx2+=4;rx3+=4;rx4+=4;
            ix1+=4;ix2+=4;ix3+=4;ix4+=4;
    	} while (ix4 < end);
	    td <<= 2;
    	do
		{
	    	i = 0;
			twr1=wr1;twr2=wr2;twr3=wr3;twi1=wi1;twi2=wi2;twi3=wi3;
            rx1=x;rx2=x+td;rx3=td+td+x;rx4=td+td+td+x;
            ix1=x+n;ix2=n+td+x;ix3=n+td+td+x;ix4=n+td+td+td+x;
    		do
    		{
		    	wr1=twr1;wr2=twr2;wr3=twr3;wi1=twi1;wi2=twi2;wi3=twi3;
				for (j=0; j < td; j+=4)
			    {
				    mr1 = _mm256_load_pd(rx1);
					mi1 = _mm256_load_pd(ix1);
	    			mr2 = _mm256_fmsub_pd(_mm256_load_pd(rx2) , _mm256_load_pd(wr1) , _mm256_load_pd(ix2) * _mm256_load_pd(wi1));
		    		mi2 = _mm256_fmadd_pd(_mm256_load_pd(rx2) , _mm256_load_pd(wi1) , _mm256_load_pd(ix2) * _mm256_load_pd(wr1));
					mr3 = _mm256_fmsub_pd(_mm256_load_pd(rx3) , _mm256_load_pd(wr2) , _mm256_load_pd(ix3) * _mm256_load_pd(wi2));
			    	mi3 = _mm256_fmadd_pd(_mm256_load_pd(rx3) , _mm256_load_pd(wi2) , _mm256_load_pd(ix3) * _mm256_load_pd(wr2));
				    mr4 = _mm256_fmsub_pd(_mm256_load_pd(rx4) , _mm256_load_pd(wr3) , _mm256_load_pd(ix4) * _mm256_load_pd(wi3));
					mi4 = _mm256_fmadd_pd(_mm256_load_pd(rx4) , _mm256_load_pd(wi3) , _mm256_load_pd(ix4) * _mm256_load_pd(wr3));
                    _mm256_store_pd(rx1,mr1 + mr2 + mr3 + mr4);
		    		_mm256_store_pd(ix1,mi1 + mi2 + mi3 + mi4);
			    	_mm256_store_pd(rx2,mr1 + mi2 - mr3 - mi4);
			    	_mm256_store_pd(ix2,mi1 - mr2 - mi3 + mr4);
				    _mm256_store_pd(rx3,mr1 - mr2 + mr3 - mr4);
    				_mm256_store_pd(ix3,mi1 - mi2 + mi3 - mi4);
    				_mm256_store_pd(rx4,mr1 - mi2 - mr3 + mi4);
		    		_mm256_store_pd(ix4,mi1 + mr2 - mi3 - mr4);
			    	rx1+=4;rx2+=4;rx3+=4;rx4+=4;
                    ix1+=4;ix2+=4;ix3+=4;ix4+=4;
                    wr1+=4;wr2+=4;wr3+=4;
    				wi1+=4;wi2+=4;wi3+=4;
			    }
				rx1+=3*td;rx2+=3*td;rx3+=3*td;rx4+=3*td;
                ix1+=3*td;ix2+=3*td;ix3+=3*td;ix4+=3*td;
		    } while (ix4 < end);
		    td <<= 2;
	    } while (td != n);
    }
};
template <size_t n>
struct fft4_col
{
    double *W[3];
    BitReverse4<n> BR;
    fft4_col(BitReverse4<n> &r){
        BR=r;
        int k=log4(n);
		W[0] = (double*)_mm_malloc(((n-1)/3-k)*sizeof(double)*2,32);
		W[1] = (double*)_mm_malloc(((n-1)/3-k)*sizeof(double)*2,32);
		W[2] = (double*)_mm_malloc(((n-1)/3-k)*sizeof(double)*2,32);
		double *r0=W[0],*r1=W[1],*r2=W[2],*i0=W[0]+((n-1)/3-k),*i1=W[1]+((n-1)/3-k),*i2=W[2]+((n-1)/3-k);
		for (int i = 1; i < k; i++){
			for(int j=1;j<(1 << (i * 2));j++){
				*(r0++)=cos(2 * j * pi / (1 << (i * 2+2)));
				*(r1++)=cos(2 * 2*j * pi / (1 << (i * 2+2)));
				*(r2++)=cos(2 * 3*j * pi / (1 << (i * 2+2)));
				*(i0++)=-sin(2 * j * pi / (1 << (i * 2+2)));
				*(i1++)=-sin(2 * 2*j * pi / (1 << (i * 2+2)));
				*(i2++)=-sin(2 * 3*j * pi / (1 << (i * 2+2)));
			}
		}
    }
    ~fft4_col(){
        _mm_free(W[0]);
		_mm_free(W[1]);
		_mm_free(W[2]);
    }
    void operator()(double **x,int k){
        int tt, td , i, j,col=4;
		__m256d r1, r2, r3, r4, i1, i2, i3, i4;
		double *rx[n],*ix[n];
		for(i=0;i<n;i++) 
		{
			rx[i]=&x[i][k];
			ix[i]=&x[i][k+col];
		}
		double *tu=(double*)_mm_malloc(4*n*sizeof(double)*2,32);
		double *wr1,*wr2,*wr3,*wi1,*wi2,*wi3,*twr1,*twr2,*twr3,*twi1,*twi2,*twi3;
        double *rx1,*rx2,*rx3,*rx4,*ix1,*ix2,*ix3,*ix4,*end=tu+8*n;
		wr1=W[0];wr2=W[1];wr3=W[2];
		wi1=W[0]+((n-1)/3-log4(n));wi2=W[1]+((n-1)/3-log4(n));wi3=W[2]+((n-1)/3-log4(n));
        rx1=tu;rx2=tu+4;rx3=tu+8;rx4=tu+12;
        ix1=4*n+tu;ix2=4*(n+1)+tu;ix3=4*(n+2)+tu;ix4=4*(n+3)+tu;
        td=1;
		i=0;
		do
		{
			r1 = _mm256_load_pd(rx[BR.rec[i]]);
			i1 = _mm256_load_pd(ix[BR.rec[i]]);
			r2 = _mm256_load_pd(rx[BR.rec[i+1]]);
			i2 = _mm256_load_pd(ix[BR.rec[i+1]]);
			r3 = _mm256_load_pd(rx[BR.rec[i+2]]);
			i3 = _mm256_load_pd(ix[BR.rec[i+2]]);
			r4 = _mm256_load_pd(rx[BR.rec[i+3]]);
			i4 = _mm256_load_pd(ix[BR.rec[i+3]]);
			_mm256_store_pd(rx1,r1 + r2 + r3 + r4);
			_mm256_store_pd(ix1,i1 + i2 + i3 + i4);
			_mm256_store_pd(rx2,r1 + i2 - r3 - i4);
			_mm256_store_pd(ix2,i1 - r2 - i3 + r4);
			_mm256_store_pd(rx3,r1 - r2 + r3 - r4);
			_mm256_store_pd(ix3,i1 - i2 + i3 - i4);
			_mm256_store_pd(rx4,r1 - i2 - r3 + i4);
			_mm256_store_pd(ix4,i1 + r2 - i3 - r4);
			rx1+=16;rx2+=16;rx3+=16;rx4+=16;
            ix1+=16;ix2+=16;ix3+=16;ix4+=16;
			i+=4;
		} while (ix4 < end);
		td <<= 2;
		do
		{
			rx1=tu;rx2=tu+4*td;rx3=tu+8*td;rx4=tu+12*td;
        	ix1=4*n+tu;ix2=4*(n+td)+tu;ix3=4*(n+td+td)+tu;ix4=4*(n+td+td+td)+tu;
			twr1=wr1;twr2=wr2;twr3=wr3;twi1=wi1;twi2=wi2;twi3=wi3;
			do
			{
				r1 = _mm256_load_pd(rx1);
				i1 = _mm256_load_pd(ix1);
				r2 = _mm256_load_pd(rx2);
				i2 = _mm256_load_pd(ix2);
				r3 = _mm256_load_pd(rx3);
				i3 = _mm256_load_pd(ix3);
				r4 = _mm256_load_pd(rx4);
				i4 = _mm256_load_pd(ix4);
				_mm256_store_pd(rx1,r1 + r2 + r3 + r4);
				_mm256_store_pd(ix1,i1 + i2 + i3 + i4);
				_mm256_store_pd(rx2,r1 + i2 - r3 - i4);
				_mm256_store_pd(ix2,i1 - r2 - i3 + r4);
				_mm256_store_pd(rx3,r1 - r2 + r3 - r4);
				_mm256_store_pd(ix3,i1 - i2 + i3 - i4);
				_mm256_store_pd(rx4,r1 - i2 - r3 + i4);
				_mm256_store_pd(ix4,i1 + r2 - i3 - r4);
				rx1+=4;rx2+=4;rx3+=4;rx4+=4;
            	ix1+=4;ix2+=4;ix3+=4;ix4+=4;
				j=1;
				wr1=twr1;wr2=twr2;wr3=twr3;wi1=twi1;wi2=twi2;wi3=twi3;
				for (; j < td; j++)
				{
					r1 = _mm256_load_pd(rx1);
					i1 = _mm256_load_pd(ix1);
					r2 = _mm256_fmsub_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wr1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wi1));
					i2 = _mm256_fmadd_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wi1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wr1));
					r3 = _mm256_fmsub_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wr2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wi2));
					i3 = _mm256_fmadd_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wi2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wr2));
					r4 = _mm256_fmsub_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wr3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wi3));
					i4 = _mm256_fmadd_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wi3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wr3));
					_mm256_store_pd(rx1,r1 + r2 + r3 + r4);
					_mm256_store_pd(ix1,i1 + i2 + i3 + i4);
					_mm256_store_pd(rx2,r1 + i2 - r3 - i4);
					_mm256_store_pd(ix2,i1 - r2 - i3 + r4);
					_mm256_store_pd(rx3,r1 - r2 + r3 - r4);
					_mm256_store_pd(ix3,i1 - i2 + i3 - i4);
					_mm256_store_pd(rx4,r1 - i2 - r3 + i4);
					_mm256_store_pd(ix4,i1 + r2 - i3 - r4);
					wr1++;wr2++;wr3++;
					wi1++;wi2++;wi3++;
					rx1+=4;rx2+=4;rx3+=4;rx4+=4;
					ix1+=4;ix2+=4;ix3+=4;ix4+=4;
				}
				rx1+=12*td;rx2+=12*td;rx3+=12*td;rx4+=12*td;
                ix1+=12*td;ix2+=12*td;ix3+=12*td;ix4+=12*td;
			} while (ix4 < end);
			td <<= 2;
		} while (td != n>>2);
		rx1=tu;rx2=tu+4*td;rx3=tu+8*td;rx4=tu+12*td;
		ix1=4*n+tu;ix2=4*(n+td)+tu;ix3=4*(n+td+td)+tu;ix4=4*(n+td+td+td)+tu;
		r1 = _mm256_load_pd(rx1);
		i1 = _mm256_load_pd(ix1);
		r2 = _mm256_load_pd(rx2);
		i2 = _mm256_load_pd(ix2);
		r3 = _mm256_load_pd(rx3);
		i3 = _mm256_load_pd(ix3);
		r4 = _mm256_load_pd(rx4);
		i4 = _mm256_load_pd(ix4);
		i=0;
		_mm256_store_pd(rx[i],r1+r2+r3+r4);
		_mm256_store_pd(ix[i],i1+i2+i3+i4);
		_mm256_store_pd(rx[i+td],r1+i2-r3-i4);
		_mm256_store_pd(ix[i+td],i1-r2-i3+r4);
		_mm256_store_pd(rx[i+td+td],r1-r2+r3-r4);
		_mm256_store_pd(ix[i+td+td],i1-i2+i3-i4);
		_mm256_store_pd(rx[i+td+td+td],r1-i2-r3+i4);
		_mm256_store_pd(ix[i+td+td+td],i1+r2-i3-r4);
		rx1+=4;rx2+=4;rx3+=4;rx4+=4;
		ix1+=4;ix2+=4;ix3+=4;ix4+=4;
		j=1;
		i=1;
		for (; j < td; j++)
		{
			r1 = _mm256_load_pd(rx1);
			i1 = _mm256_load_pd(ix1);
			r2 = _mm256_fmsub_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wr1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wi1));
			i2 = _mm256_fmadd_pd(_mm256_load_pd(rx2) , _mm256_set1_pd(*wi1) , _mm256_load_pd(ix2) * _mm256_set1_pd(*wr1));
			r3 = _mm256_fmsub_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wr2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wi2));
			i3 = _mm256_fmadd_pd(_mm256_load_pd(rx3) , _mm256_set1_pd(*wi2) , _mm256_load_pd(ix3) * _mm256_set1_pd(*wr2));
			r4 = _mm256_fmsub_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wr3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wi3));
			i4 = _mm256_fmadd_pd(_mm256_load_pd(rx4) , _mm256_set1_pd(*wi3) , _mm256_load_pd(ix4) * _mm256_set1_pd(*wr3));
			_mm256_store_pd(rx[i],r1+r2+r3+r4);
			_mm256_store_pd(ix[i],i1+i2+i3+i4);
			_mm256_store_pd(rx[i+td],r1+i2-r3-i4);
			_mm256_store_pd(ix[i+td],i1-r2-i3+r4);
			_mm256_store_pd(rx[i+td+td],r1-r2+r3-r4);
			_mm256_store_pd(ix[i+td+td],i1-i2+i3-i4);
			_mm256_store_pd(rx[i+td+td+td],r1-i2-r3+i4);
			_mm256_store_pd(ix[i+td+td+td],i1+r2-i3-r4);
			i++;
			wr1++;wr2++;wr3++;
			wi1++;wi2++;wi3++;
			rx1+=4;rx2+=4;rx3+=4;rx4+=4;
			ix1+=4;ix2+=4;ix3+=4;ix4+=4;
		}
		free(tu);
    }
};
#include <complex>
#define N 1024
#include <iostream>
using namespace std;
template <int n, typename T>
void cdft(complex<T> *in, complex<T> *out)
{
	complex<T> *W;
	W = new complex<T>[n];
	for (int i = 0; i < n; i++)
		W[i] = complex<double>(cos(2 * i * pi / n), -sin(2 * i * pi / n));
	for (int i = 0; i < n; i++)
	{
		out[i] = 0;
		for (int j = 0; j < n; j++)
		{
			out[i] += in[j] * W[i * j % n];
		}
	}
	delete W;
}
void dotmul(double *x, complex<double> *y, int n)
{
	double res(0);
	for (int i = 0; i < n; i++)
		res += abs(x[i]-y[i].real())+abs(x[i+n]-y[i].imag());
	cout << res << endl;
}
int main(){
    BitReverse4<N> BR;
	//BitReverse2<N> BR;
	//fft2_row<N> fft(BR);
	//fft2_col<N> fft(BR);
    //fft4_row<N> fft(BR);
	fft4_col<N> fft(BR);
    complex<double> y[N],z[N];
    double *x=(double*)_mm_malloc(4*N*sizeof(double)*2,32),*r[N],t[2*N];
    for(int i=0;i<N;i++) {
        r[i]=x+i*8;
		r[i][0]=r[i][1]=r[i][2]=r[i][3]=i+1;
        y[i]=complex<double>(i+1,0);
		//x[i]=i+1;
    }
    cdft<N,double>(y,z);
	//fft(x);
    //for(int i=0;i<N;i++)fft(x);
	fft(r,0);
	for(int i=0;i<N;i++){
		t[i]=r[i][3];
		t[i+N]=r[i][7];
		//cout<<t[i]<<","<<t[i+N]<<endl;
	}
	
    //for(int i=0;i<N*50;i++)fft(x);
    dotmul(t,z,N);
    //for(int i=0;i<N;i++) cout<<x[i]<<","<<x[i+N]<<endl;
}
#endif