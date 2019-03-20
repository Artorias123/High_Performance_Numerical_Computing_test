#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
using namespace std;
const constexpr double pi=atan(double(1))*4;
void prt(complex<double> *x, int n)
{
	for (int i = 0; i < n; i++)
		cout << x[i] << '\t';
	cout << endl;
}
void prt(double *x, int n)
{
	for (int i = 0; i < n; i++)
		cout << x[i] << ","<<x[i+n]<<'\t';
	cout << endl;
}
void dotmul(complex<double> *x, complex<double> *y, int n)
{
	double res(0);
	for (int i = 0; i < n; i++)
		res += norm(x[i] - y[i]);
	cout << res << endl;
}
void dotmul(double *x, complex<double> *y, int n)
{
	double res(0);
	for (int i = 0; i < n; i++)
		res += abs(x[i]-y[i].real())+abs(x[i+n]-y[i].imag());
	cout << res << endl;
}
template <typename T>
void Swap(T *x, unsigned i, unsigned j)
{
	T tmp = x[i];
	x[i] = x[j];
	x[j] = tmp;
}
template <int n>
struct BitReverse4
{
	bool judg[n];
	int record[n];
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
			record[i] = k >> 2;
			judg[k >> 2] = !judg[k >> 2];
		}
	}
	void operator()(double *x)
	{
		if (n < 4)
			return;
		for (int i = 0; i < n; i++)
		{
			if (!judg[i])
				Swap(x, i, record[i]);
		}
	}
};

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
template <int n>
struct cfft4
{
	BitReverse4<n> br;
	double *W[3];
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
	cfft4()
	{
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
	~cfft4()
	{
		_mm_free(W[0]);
		_mm_free(W[1]);
		_mm_free(W[2]);
	}
	void operator()(double *in, double *out)
	{
		int td = 1, i, j;
		double r1, r2, r3, r4;
		double *wr1=W[0],*wr2=W[1],*wr3=W[2],*wi1=W[0]+(n-4)/3,*wi2=W[1]+(n-4)/3,*wi3=W[2]+(n-4)/3;
		__m256d mr1,mr2,mr3,mr4,mi1,mi2,mi3,mi4;
		for (i = 0; i < n; i++)
			out[i] = in[i];
		br(out);
		i = 0;
		do
		{
			r1 = out[i];
			r2 = out[i + 1];
			r3 = out[i + 2];
			r4 = out[i + 3];
			out[i]=(r1 + r2 + r3 + r4);
			out[i + 1]=(r1 - r3);
			out[i + 1+n]=(-r2 + r4);
			out[i + 2]=(r1 - r2 + r3 - r4);
			out[i + 3]=out[i + 1];
			out[i + 3+n]=-out[i + 1+n];
			i += 4;
		} while (i < n);
		td <<= 2;
		double *twr1,*twr2,*twr3,*twi1,*twi2,*twi3;
		do
		{
			i = 0;
			twr1=wr1;twr2=wr2;twr3=wr3;twi1=wi1;twi2=wi2;twi3=wi3;
			do
			{
				wr1=twr1;wr2=twr2;wr3=twr3;wi1=twi1;wi2=twi2;wi3=twi3;
				for (j=0; j < td; j+=4)
				{
					mr1 = _mm256_load_pd(&out[i]);
					mi1 = _mm256_load_pd(&out[i+n]);
					mr2 = _mm256_fmsub_pd(_mm256_load_pd(&out[i + td]) , _mm256_load_pd(wr1) , _mm256_load_pd(&out[i + td+n]) * _mm256_load_pd(wi1));
					mi2 = _mm256_fmadd_pd(_mm256_load_pd(&out[i + td]) , _mm256_load_pd(wi1) , _mm256_load_pd(&out[i + td+n]) * _mm256_load_pd(wr1));
					mr3 = _mm256_fmsub_pd(_mm256_load_pd(&out[i + td + td]) , _mm256_load_pd(wr2) , _mm256_load_pd(&out[i + td + td+n]) * _mm256_load_pd(wi2));
					mi3 = _mm256_fmadd_pd(_mm256_load_pd(&out[i + td + td]) , _mm256_load_pd(wi2) , _mm256_load_pd(&out[i + td + td+n]) * _mm256_load_pd(wr2));
					mr4 = _mm256_fmsub_pd(_mm256_load_pd(&out[i + td + td + td]) , _mm256_load_pd(wr3) , _mm256_load_pd(&out[i + td + td + td+n]) * _mm256_load_pd(wi3));
					mi4 = _mm256_fmadd_pd(_mm256_load_pd(&out[i + td + td + td]) , _mm256_load_pd(wi3) , _mm256_load_pd(&out[i + td + td + td+n]) * _mm256_load_pd(wr3));
					_mm256_store_pd(&out[i],mr1 + mr2 + mr3 + mr4);
					_mm256_store_pd(&out[i+n],mi1 + mi2 + mi3 + mi4);
					_mm256_store_pd(&out[i + td],mr1 + mi2 - mr3 - mi4);
					_mm256_store_pd(&out[i + td+n],mi1 - mr2 - mi3 + mr4);
					_mm256_store_pd(&out[i + td + td],mr1 - mr2 + mr3 - mr4);
					_mm256_store_pd(&out[i + td + td+n],mi1 - mi2 + mi3 - mi4);
					_mm256_store_pd(&out[i + td + td + td],mr1 - mi2 - mr3 + mi4);
					_mm256_store_pd(&out[i + td + td + td+n],mi1 + mr2 - mi3 - mr4);
					wr1+=4;
					wr2+=4;
					wr3+=4;
					wi1+=4;
					wi2+=4;
					wi3+=4;
					i+=4;
				}
				i += td + td + td;
			} while (i < n);
			td <<= 2;
		} while (td != n);
	}

	
};

#define N 1024
int main()
{
	srand(clock());
	double *x=(double*)_mm_malloc(2*N*sizeof(double),32),*y=(double*)_mm_malloc(2*N*sizeof(double),32);
	complex<double> d1[N],d2[N];
	cfft4<N> fft;
	for (int i = 0; i < N; i++) {
		x[i]=double(rand())/RAND_MAX;
		x[i+N]=0;
		d1[i]=complex<double>(x[i],0);
	}
//	cdft<N,double>(d1,d2);
//	fft(x,y);
//	prt(y,N);
//	cout<<endl;
//	prt(d2,N);
	//for (int i = 0; i < N; i++)	x[i] = double(rand()) / RAND_MAX;
	for(int i=0;i<N*100;i++) fft(x,y);
	dotmul(y,d2,N);
}
